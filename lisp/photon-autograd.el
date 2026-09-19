;;; photon-autograd.el --- reverse-mode autograd over photon-tensor  -*- lexical-binding: t; -*-

;; A small tape-based reverse-mode automatic differentiation engine on
;; top of the photon-tensor core.  Each `pav' (photon autograd var)
;; holds a value tensor, an accumulated gradient tensor, and a backward
;; closure.  Ops record themselves on a tape during the forward pass;
;; `photon-autograd-backward' walks the tape in reverse, seeding the
;; scalar loss gradient and accumulating gradients into the leaves
;; (parameters).  Because every op delegates to photon-tensor, the heavy
;; ops (matmul/linear/softmax) run on whatever backend is active --
;; including the nelisp-gpu GPU backend, so forward AND backward run on
;; the GPU.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)

(cl-defstruct (pav (:constructor pav--make))
  value grad backward)

(defvar photon-autograd--tape nil
  "List of recorded vars, newest first (= reverse topological order).")

(defun photon-autograd-reset-tape ()
  "Clear the autograd tape before a fresh forward pass."
  (setq photon-autograd--tape nil))

(defun photon-autograd-const (tensor)
  "Wrap TENSOR as a leaf var (e.g. a parameter); its grad accumulates."
  (pav--make :value tensor
             :grad (photon-tensor-create (photon-tensor-shape tensor) 0.0)
             :backward nil))

(defun photon-autograd--record (value backward)
  "Make a var holding VALUE with BACKWARD closure and push it on the tape."
  (let ((v (pav--make :value value
                      :grad (photon-tensor-create (photon-tensor-shape value) 0.0)
                      :backward backward)))
    (push v photon-autograd--tape)
    v))

(defun photon-autograd--addgrad (var delta)
  "Accumulate DELTA (a tensor, same length as VAR's grad) into VAR's grad."
  (let* ((g (photon-tensor-data (pav-grad var)))
         (dd (photon-tensor-data delta)) (n (length g)) (i 0))
    (while (< i n) (aset g i (+ (aref g i) (aref dd i))) (setq i (1+ i)))))

(defun photon-autograd--colsum (g)
  "Sum 2D tensor G (m x n) over rows -> (n) tensor (for bias gradients)."
  (let* ((sh (photon-tensor-shape g)) (m (car sh)) (n (nth 1 sh))
         (d (photon-tensor-data g)) (out (make-vector n 0.0)) (i 0))
    (while (< i m)
      (let ((j 0)) (while (< j n)
                     (aset out j (+ (aref out j) (aref d (+ (* i n) j))))
                     (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list n) out)))

;; --- differentiable ops ----------------------------------------------
(defun photon-autograd-matmul (a b)
  "Autograd matmul A (m x k) by B (k x n)."
  (let ((out (photon-tensor-matmul (pav-value a) (pav-value b))))
    (photon-autograd--record
     out
     (lambda (g)
       (photon-autograd--addgrad
        a (photon-tensor-matmul g (photon-tensor-transpose (pav-value b))))
       (photon-autograd--addgrad
        b (photon-tensor-matmul (photon-tensor-transpose (pav-value a)) g))))))

(defun photon-autograd-linear (x w b)
  "Autograd affine Y = X (m x in) * W^T (W out x in) + B (out)."
  (let ((out (photon-tensor-linear (pav-value x) (pav-value w) (pav-value b))))
    (photon-autograd--record
     out
     (lambda (g)                                ; g : (m x out)
       (photon-autograd--addgrad x (photon-tensor-matmul g (pav-value w)))
       (photon-autograd--addgrad
        w (photon-tensor-matmul (photon-tensor-transpose g) (pav-value x)))
       (photon-autograd--addgrad b (photon-autograd--colsum g))))))

(defun photon-autograd-add (a b)
  "Autograd elementwise sum (same shape)."
  (let ((out (photon-tensor-add (pav-value a) (pav-value b))))
    (photon-autograd--record
     out (lambda (g) (photon-autograd--addgrad a g)
                     (photon-autograd--addgrad b g)))))

(defun photon-autograd--gelu-grad (x)
  "Elementwise derivative of the tanh-approx GELU at tensor X."
  (let* ((d (photon-tensor-data x)) (n (length d)) (out (make-vector n 0.0))
         (c 0.7978845608028654) (i 0))
    (while (< i n)
      (let* ((xv (aref d i))
             (u (* c (+ xv (* 0.044715 xv xv xv))))
             (e (exp (* 2.0 u)))
             (th (/ (- e 1.0) (+ e 1.0)))
             (sech2 (- 1.0 (* th th)))
             (dudx (* c (+ 1.0 (* 0.134145 xv xv)))))
        (aset out i (+ (* 0.5 (+ 1.0 th)) (* 0.5 xv sech2 dudx))))
      (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape x) out)))

(defun photon-autograd-gelu (x)
  "Autograd GELU."
  (let ((out (photon-tensor-gelu (pav-value x))))
    (photon-autograd--record
     out (lambda (g)
           (photon-autograd--addgrad
            x (photon-tensor-hadamard g (photon-autograd--gelu-grad (pav-value x))))))))

(defun photon-autograd-softmax-ce (logits targets)
  "Softmax cross-entropy loss of LOGITS (m x vocab) vs TARGETS (m ints).
Returns a scalar (1 x 1) loss var; backward gives (softmax-onehot)/m."
  (let* ((lv (pav-value logits)) (sh (photon-tensor-shape lv))
         (m (car sh)) (vocab (nth 1 sh))
         (probs (photon-tensor-softmax-rows lv))
         (pd (photon-tensor-data probs))
         (loss 0.0) (i 0))
    (while (< i m)
      (let ((tt (elt targets i)))
        (setq loss (- loss (log (max 1.0e-30 (aref pd (+ (* i vocab) tt)))))))
      (setq i (1+ i)))
    (setq loss (/ loss (float m)))
    (photon-autograd--record
     (photon-tensor (list 1 1) (vector loss))
     (lambda (g)
       (let* ((gv (aref (photon-tensor-data g) 0))
              (dl (make-vector (* m vocab) 0.0)) (ii 0))
         (while (< ii m)
           (let ((tt (elt targets ii)) (jj 0))
             (while (< jj vocab)
               (aset dl (+ (* ii vocab) jj)
                     (* gv (/ (- (aref pd (+ (* ii vocab) jj))
                                 (if (= jj tt) 1.0 0.0))
                              (float m))))
               (setq jj (1+ jj))))
           (setq ii (1+ ii)))
         (photon-autograd--addgrad logits (photon-tensor (list m vocab) dl)))))))

(defun photon-autograd-transpose (a)
  "Autograd 2D transpose."
  (let ((out (photon-tensor-transpose (pav-value a))))
    (photon-autograd--record
     out (lambda (g) (photon-autograd--addgrad a (photon-tensor-transpose g))))))

(defun photon-autograd-scale (a s)
  "Autograd scalar scale of A by S."
  (let ((out (photon-tensor-scale (pav-value a) s)))
    (photon-autograd--record
     out (lambda (g) (photon-autograd--addgrad a (photon-tensor-scale g s))))))

(defun photon-autograd-softmax-rows (a)
  "Autograd row-wise softmax.  Backward: ds = p .* (g - rowsum(p.*g))."
  (let ((p (photon-tensor-softmax-rows (pav-value a))))
    (photon-autograd--record
     p
     (lambda (g)
       (let* ((sh (photon-tensor-shape p)) (m (car sh)) (n (nth 1 sh))
              (pd (photon-tensor-data p)) (gd (photon-tensor-data g))
              (ds (make-vector (* m n) 0.0)) (i 0))
         (while (< i m)
           (let ((base (* i n)) (dot 0.0) (j 0))
             (while (< j n)
               (setq dot (+ dot (* (aref pd (+ base j)) (aref gd (+ base j)))))
               (setq j (1+ j)))
             (setq j 0)
             (while (< j n)
               (aset ds (+ base j)
                     (* (aref pd (+ base j)) (- (aref gd (+ base j)) dot)))
               (setq j (1+ j))))
           (setq i (1+ i)))
         (photon-autograd--addgrad a (photon-tensor (list m n) ds)))))))

(defun photon-autograd-layernorm-rows (x gamma beta &optional eps)
  "Autograd row-wise layernorm with trainable GAMMA, BETA."
  (let* ((xv (pav-value x)) (sh (photon-tensor-shape xv))
         (m (car sh)) (n (nth 1 sh)) (e (or eps 1.0e-5))
         (out (photon-tensor-layernorm-rows xv (pav-value gamma) (pav-value beta) e))
         (xd (photon-tensor-data xv)) (istd (make-vector m 0.0))
         (xhat (make-vector (* m n) 0.0)) (ninv (/ 1.0 (float n))) (i 0))
    (while (< i m)
      (let ((base (* i n)) (mu 0.0) (j 0))
        (while (< j n) (setq mu (+ mu (aref xd (+ base j)))) (setq j (1+ j)))
        (setq mu (* mu ninv))
        (let ((var 0.0) (j2 0))
          (while (< j2 n)
            (let ((c (- (aref xd (+ base j2)) mu))) (setq var (+ var (* c c))))
            (setq j2 (1+ j2)))
          (setq var (* var ninv))
          (let ((is (/ 1.0 (sqrt (+ var e)))) (j3 0))
            (aset istd i is)
            (while (< j3 n)
              (aset xhat (+ base j3) (* (- (aref xd (+ base j3)) mu) is))
              (setq j3 (1+ j3))))))
      (setq i (1+ i)))
    (photon-autograd--record
     out
     (lambda (g)
       (let* ((gd (photon-tensor-data g)) (gam (photon-tensor-data (pav-value gamma)))
              (dgamma (make-vector n 0.0)) (dbeta (make-vector n 0.0))
              (dx (make-vector (* m n) 0.0)) (i2 0))
         (while (< i2 m)
           (let ((base (* i2 n)) (is (aref istd i2)) (dxm 0.0) (dxxh 0.0) (j 0))
             (while (< j n)
               (let* ((dy (aref gd (+ base j))) (xh (aref xhat (+ base j)))
                      (dxh (* dy (aref gam j))))
                 (aset dgamma j (+ (aref dgamma j) (* dy xh)))
                 (aset dbeta j (+ (aref dbeta j) dy))
                 (setq dxm (+ dxm dxh)) (setq dxxh (+ dxxh (* dxh xh))))
               (setq j (1+ j)))
             (setq dxm (* dxm ninv)) (setq dxxh (* dxxh ninv))
             (setq j 0)
             (while (< j n)
               (let* ((xh (aref xhat (+ base j)))
                      (dxh (* (aref gd (+ base j)) (aref gam j))))
                 (aset dx (+ base j) (* is (- dxh dxm (* xh dxxh)))))
               (setq j (1+ j))))
           (setq i2 (1+ i2)))
         (photon-autograd--addgrad x (photon-tensor (list m n) dx))
         (photon-autograd--addgrad gamma (photon-tensor (list n) dgamma))
         (photon-autograd--addgrad beta (photon-tensor (list n) dbeta)))))))

(defun photon-autograd-embedding (wte tokens dim)
  "Autograd embedding lookup of TOKENS from WTE (vocab x DIM)."
  (let ((out (photon-tensor-embedding (pav-value wte) tokens dim)))
    (photon-autograd--record
     out
     (lambda (g)
       (let ((gd (photon-tensor-data g)) (wg (photon-tensor-data (pav-grad wte))) (i 0))
         (dolist (tk tokens)
           (let ((src (* i dim)) (dst (* tk dim)) (k 0))
             (while (< k dim)
               (aset wg (+ dst k) (+ (aref wg (+ dst k)) (aref gd (+ src k))))
               (setq k (1+ k))))
           (setq i (1+ i))))))))

;; --- backward / optimization -----------------------------------------
(defun photon-autograd-backward (loss)
  "Backpropagate from scalar LOSS var through the tape."
  (aset (photon-tensor-data (pav-grad loss)) 0 1.0)
  (dolist (v photon-autograd--tape)
    (when (pav-backward v) (funcall (pav-backward v) (pav-grad v)))))

(defun photon-autograd-zero-grad (vars)
  "Zero the gradients of VARS (parameters) before a backward pass."
  (dolist (v vars)
    (let* ((g (photon-tensor-data (pav-grad v))) (n (length g)) (i 0))
      (while (< i n) (aset g i 0.0) (setq i (1+ i))))))

(defun photon-autograd-sgd (vars lr)
  "In-place SGD step: value -= LR * grad for each var in VARS."
  (dolist (v vars)
    (let* ((val (photon-tensor-data (pav-value v)))
           (gr (photon-tensor-data (pav-grad v))) (n (length val)) (i 0))
      (while (< i n) (aset val i (- (aref val i) (* lr (aref gr i)))) (setq i (1+ i))))))

(provide 'photon-autograd)
;;; photon-autograd.el ends here
