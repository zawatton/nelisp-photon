;;; photon-tensor.el --- Pure-elisp dense tensor core for nelisp-photon  -*- lexical-binding: t; -*-

;; Row-major dense tensors backed by flat float vectors.  Written as
;; plain Emacs Lisp so it byte-/native-compiles for speed on Emacs and
;; stays loadable on the standalone NeLisp runtime (slowly -- the
;; performance substrate is Emacs native-comp; NeLisp is kept only for
;; portability).
;;
;; A tensor is the 2-element vector [SHAPE DATA] where SHAPE is a list of
;; positive integers and DATA is a flat `float' vector in row-major
;; order.  The hot kernels (matmul, linear, softmax, layernorm) operate
;; directly on DATA with `aref'/`aset' in tight `while' loops, which the
;; byte/native compiler turns into fast numeric code.

;;; Code:

(defsubst photon-tensor (shape data) (vector shape data))
(defsubst photon-tensor-shape (tn) (aref tn 0))
(defsubst photon-tensor-data (tn) (aref tn 1))

(defun photon-tensor-rank (tn) (length (photon-tensor-shape tn)))
(defun photon-tensor-size (tn) (length (photon-tensor-data tn)))

(defun photon-tensor--tanh (x)
  "tanh of X using `exp' so the file stays portable to NeLisp."
  (let ((e (exp (* 2.0 x)))) (/ (- e 1.0) (+ e 1.0))))

(defun photon-tensor-create (shape &optional fill)
  "Create a tensor of SHAPE filled with FILL (default 0.0)."
  (let ((size 1))
    (dolist (d shape) (setq size (* size d)))
    (photon-tensor (copy-sequence shape)
                   (make-vector size (float (or fill 0.0))))))

(defun photon-tensor-from-list (shape values)
  "Create a tensor of SHAPE from flat VALUES (a list or vector)."
  (let* ((size 1) (i 0) data)
    (dolist (d shape) (setq size (* size d)))
    (setq data (make-vector size 0.0))
    (if (listp values)
        (dolist (v values) (aset data i (float v)) (setq i (1+ i)))
      (while (< i size) (aset data i (float (aref values i))) (setq i (1+ i))))
    (photon-tensor (copy-sequence shape) data)))

(defun photon-tensor-row (tn i)
  "Return row I of 2D tensor TN as a fresh (1 x cols) tensor."
  (let* ((sh (photon-tensor-shape tn)) (n (nth 1 sh))
         (d (photon-tensor-data tn)) (out (make-vector n 0.0))
         (base (* i n)) (j 0))
    (while (< j n) (aset out j (aref d (+ base j))) (setq j (1+ j)))
    (photon-tensor (list 1 n) out)))

(defun photon-tensor-matmul (a b)
  "Matrix product of 2D tensors A (m x k) and B (k x n) -> (m x n)."
  (let* ((ash (photon-tensor-shape a)) (bsh (photon-tensor-shape b))
         (m (car ash)) (k (nth 1 ash)) (k2 (car bsh)) (n (nth 1 bsh))
         (ad (photon-tensor-data a)) (bd (photon-tensor-data b))
         (out (make-vector (* m n) 0.0)) (i 0))
    (unless (= k k2)
      (error "photon-tensor-matmul: shape mismatch %S %S" ash bsh))
    (while (< i m)
      (let ((j 0) (ai (* i k)) (oi (* i n)))
        (while (< j n)
          (let ((s 0.0) (kk 0))
            (while (< kk k)
              (setq s (+ s (* (aref ad (+ ai kk))
                              (aref bd (+ (* kk n) j)))))
              (setq kk (1+ kk)))
            (aset out (+ oi j) s))
          (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list m n) out)))

(defun photon-tensor-linear (x w &optional b)
  "Affine transform: X (m x in) by weight W (out x in) plus bias B (out).
Returns (m x out).  This is the PyTorch `nn.Linear' convention where the
weight is stored as (out-features x in-features)."
  (let* ((xsh (photon-tensor-shape x)) (wsh (photon-tensor-shape w))
         (m (car xsh)) (in (nth 1 xsh)) (out (car wsh))
         (xd (photon-tensor-data x)) (wd (photon-tensor-data w))
         (bd (and b (photon-tensor-data b)))
         (res (make-vector (* m out) 0.0)) (i 0))
    (while (< i m)
      (let ((o 0) (xi (* i in)) (ri (* i out)))
        (while (< o out)
          (let ((s (if bd (aref bd o) 0.0)) (wo (* o in)) (kk 0))
            (while (< kk in)
              (setq s (+ s (* (aref xd (+ xi kk)) (aref wd (+ wo kk)))))
              (setq kk (1+ kk)))
            (aset res (+ ri o) s))
          (setq o (1+ o))))
      (setq i (1+ i)))
    (photon-tensor (list m out) res)))

(defun photon-tensor-transpose (a)
  "Transpose 2D tensor A (m x n) -> (n x m)."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh))
         (d (photon-tensor-data a)) (out (make-vector (* m n) 0.0)) (i 0))
    (while (< i m)
      (let ((j 0))
        (while (< j n)
          (aset out (+ (* j m) i) (aref d (+ (* i n) j)))
          (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list n m) out)))

(defun photon-tensor-add (a b)
  "Elementwise sum of equal-length tensors A and B (keeps A's shape)."
  (let* ((ad (photon-tensor-data a)) (bd (photon-tensor-data b))
         (len (length ad)) (out (make-vector len 0.0)) (i 0))
    (while (< i len) (aset out i (+ (aref ad i) (aref bd i))) (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

(defun photon-tensor-add-bias (a b)
  "Add row vector B (n) to every row of 2D tensor A (m x n)."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh))
         (ad (photon-tensor-data a)) (bd (photon-tensor-data b))
         (out (make-vector (* m n) 0.0)) (i 0))
    (while (< i m)
      (let ((base (* i n)) (j 0))
        (while (< j n)
          (aset out (+ base j) (+ (aref ad (+ base j)) (aref bd j)))
          (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list m n) out)))

(defun photon-tensor-scale (a s)
  "Multiply every element of A by scalar S."
  (let* ((ad (photon-tensor-data a)) (len (length ad))
         (sf (float s)) (out (make-vector len 0.0)) (i 0))
    (while (< i len) (aset out i (* (aref ad i) sf)) (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

(defun photon-tensor-hadamard (a b)
  "Elementwise product of equal-length tensors A and B."
  (let* ((ad (photon-tensor-data a)) (bd (photon-tensor-data b))
         (len (length ad)) (out (make-vector len 0.0)) (i 0))
    (while (< i len) (aset out i (* (aref ad i) (aref bd i))) (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

(defun photon-tensor-softmax-rows (a)
  "Row-wise softmax over 2D tensor A (m x n)."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh))
         (d (photon-tensor-data a)) (out (make-vector (* m n) 0.0)) (i 0))
    (while (< i m)
      (let ((base (* i n)) (mx -1.0e30) (j 0))
        (while (< j n)
          (let ((v (aref d (+ base j)))) (when (> v mx) (setq mx v)))
          (setq j (1+ j)))
        (let ((sum 0.0) (j2 0))
          (while (< j2 n)
            (let ((e (exp (- (aref d (+ base j2)) mx))))
              (aset out (+ base j2) e) (setq sum (+ sum e)))
            (setq j2 (1+ j2)))
          (let ((inv (/ 1.0 sum)) (j3 0))
            (while (< j3 n)
              (aset out (+ base j3) (* (aref out (+ base j3)) inv))
              (setq j3 (1+ j3))))))
      (setq i (1+ i)))
    (photon-tensor (list m n) out)))

(defun photon-tensor-layernorm-rows (a gamma beta &optional eps)
  "Row-wise layer normalization of A (m x n) with GAMMA, BETA (n).
EPS defaults to 1e-5."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh))
         (d (photon-tensor-data a))
         (g (photon-tensor-data gamma)) (be (photon-tensor-data beta))
         (e (or eps 1.0e-5)) (ninv (/ 1.0 (float n)))
         (out (make-vector (* m n) 0.0)) (i 0))
    (while (< i m)
      (let ((base (* i n)) (mean 0.0) (j 0))
        (while (< j n) (setq mean (+ mean (aref d (+ base j)))) (setq j (1+ j)))
        (setq mean (* mean ninv))
        (let ((var 0.0) (j2 0))
          (while (< j2 n)
            (let ((c (- (aref d (+ base j2)) mean))) (setq var (+ var (* c c))))
            (setq j2 (1+ j2)))
          (setq var (* var ninv))
          (let ((inv (/ 1.0 (sqrt (+ var e)))) (j3 0))
            (while (< j3 n)
              (aset out (+ base j3)
                    (+ (* (* (- (aref d (+ base j3)) mean) inv) (aref g j3))
                       (aref be j3)))
              (setq j3 (1+ j3))))))
      (setq i (1+ i)))
    (photon-tensor (list m n) out)))

(defun photon-tensor-gelu (a)
  "GELU activation (tanh approximation), elementwise over A."
  (let* ((d (photon-tensor-data a)) (len (length d))
         (out (make-vector len 0.0)) (c 0.7978845608028654) (i 0))
    (while (< i len)
      (let* ((x (aref d i))
             (inner (* c (+ x (* 0.044715 x x x)))))
        (aset out i (* 0.5 x (+ 1.0 (photon-tensor--tanh inner)))))
      (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

(defun photon-tensor-relu (a)
  "ReLU activation, elementwise over A."
  (let* ((d (photon-tensor-data a)) (len (length d))
         (out (make-vector len 0.0)) (i 0))
    (while (< i len)
      (let ((x (aref d i))) (aset out i (if (> x 0.0) x 0.0)))
      (setq i (1+ i)))
    (photon-tensor (photon-tensor-shape a) out)))

(defun photon-tensor-embedding (table ids dim)
  "Gather rows of TABLE (vocab x DIM) for token IDS -> (len x DIM)."
  (let* ((td (photon-tensor-data table)) (len (length ids))
         (out (make-vector (* len dim) 0.0)) (i 0))
    (dolist (id ids)
      (let ((src (* id dim)) (dst (* i dim)) (k 0))
        (while (< k dim)
          (aset out (+ dst k) (aref td (+ src k)))
          (setq k (1+ k))))
      (setq i (1+ i)))
    (photon-tensor (list len dim) out)))

(provide 'photon-tensor)
;;; photon-tensor.el ends here
