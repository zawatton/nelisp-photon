;;; photon-tensor-gpu.el --- GPU backend for photon-tensor via nelisp-gpu  -*- lexical-binding: t; -*-

;; Opt-in GPU backend: routes the heavy photon-tensor ops (matmul,
;; linear, softmax-rows, layernorm-rows, gelu) through the nelisp-gpu
;; SPIR-V/Vulkan kernels.  photon-tensor itself stays a pure-elisp,
;; standalone library; this file (which lives in nelisp-photon, the
;; consumer) loads nelisp-gpu and swaps the op function cells when the
;; GPU backend is enabled, so photon-transformer runs on the GPU
;; unchanged.
;;
;;   (require 'photon-tensor-gpu)
;;   (photon-tensor-use-gpu-backend)   ; subsequent ops run on the GPU
;;   ... (photon-transformer-forward ...) ...
;;   (photon-tensor-use-cpu-backend)   ; restore pure-elisp ops

;;; Code:

(require 'photon-tensor)

;; Locate the sibling nelisp-gpu library relative to this file.
(eval-and-compile
  (add-to-list 'load-path
               (expand-file-name
                "../../nelisp-gpu/lisp"
                (file-name-directory (or load-file-name buffer-file-name
                                         default-directory))))
  (require 'nelisp-gpu-run)
  (require 'nelisp-gpu-server))

(setq nelisp-gpu-vkrun
      (expand-file-name
       "../../nelisp-gpu/host/vkrun"
       (file-name-directory (or load-file-name buffer-file-name
                                default-directory))))

(declare-function photon-transformer--slice-cols "photon-transformer")
(declare-function photon-transformer--set-cols "photon-transformer")
(declare-function photon-transformer--causal-mask "photon-transformer")

(defun photon-tensor-matmul-gpu (a b)
  "GPU matmul of 2D tensors A (m x k) and B (k x n)."
  (let* ((ash (photon-tensor-shape a)) (bsh (photon-tensor-shape b))
         (m (car ash)) (k (nth 1 ash)) (n (nth 1 bsh)))
    (photon-tensor
     (list m n)
     (nth 2 (nelisp-gpu-run 'matmul
                            (list (photon-tensor-data a) (photon-tensor-data b)
                                  (make-vector (* m n) 0.0))
                            (list m k n) (/ (+ (* m n) 63) 64))))))

(defun photon-tensor-linear-gpu (x w &optional b)
  "GPU affine: X (m x in) by W (out x in) plus bias B (out).
When the persistent server is up the weight W is kept resident (uploaded
once and referenced by handle), so per-token forwards stop re-encoding and
re-uploading the weight matrix -- the activation X (small) is the only
inline input.  Falls back to the per-op vkrun host when the server is down."
  (let* ((xsh (photon-tensor-shape x)) (wsh (photon-tensor-shape w))
         (m (car xsh)) (in (nth 1 xsh)) (out (car wsh))
         (bd (if b (photon-tensor-data b) (make-vector out 0.0))))
    (photon-tensor
     (list m out)
     (if (nelisp-gpu-server-up-p)
         (car (nelisp-gpu-server-run2
               'linear
               (list (cons 'in (photon-tensor-data x))
                     (list 'res (nelisp-gpu-server-resident (photon-tensor-data w))
                           (* out in))
                     (cons 'in bd)
                     (cons 'out (* m out)))
               (list m in out) (/ (+ (* m out) 63) 64)))
       (nth 3 (nelisp-gpu-run 'linear
                              (list (photon-tensor-data x) (photon-tensor-data w)
                                    bd (make-vector (* m out) 0.0))
                              (list m in out) (/ (+ (* m out) 63) 64)))))))

(defun photon-tensor-softmax-rows-gpu (a)
  "GPU row-wise softmax of A (m x n)."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh)))
    (photon-tensor
     (list m n)
     (nth 1 (nelisp-gpu-run 'softmax
                            (list (photon-tensor-data a) (make-vector (* m n) 0.0))
                            (list m n) (/ (+ m 63) 64))))))

(defun photon-tensor-layernorm-rows-gpu (a gamma beta &optional _eps)
  "GPU row-wise layernorm of A (m x n); GAMMA/BETA applied on the CPU.
The GPU kernel normalizes (eps 1e-5); the affine gamma/beta is a cheap
elementwise pass so arbitrary gamma/beta stay supported."
  (let* ((sh (photon-tensor-shape a)) (m (car sh)) (n (nth 1 sh))
         (norm (nth 1 (nelisp-gpu-run 'layernorm
                                      (list (photon-tensor-data a)
                                            (make-vector (* m n) 0.0))
                                      (list m n) (/ (+ m 63) 64))))
         (g (photon-tensor-data gamma)) (be (photon-tensor-data beta)) (i 0))
    (while (< i m)
      (let ((j 0) (base (* i n)))
        (while (< j n)
          (aset norm (+ base j)
                (+ (* (aref norm (+ base j)) (aref g j)) (aref be j)))
          (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list m n) norm)))

(defun photon-tensor-gelu-gpu (a)
  "GPU GELU over all elements of A (shape preserved)."
  (let* ((d (photon-tensor-data a)) (n (length d)))
    (photon-tensor
     (photon-tensor-shape a)
     (nth 1 (nelisp-gpu-run 'gelu (list d (make-vector n 0.0))
                            (list n) (/ (+ n 63) 64))))))

(defun photon-transformer--ffn-gpu (x layer)
  "Fused feed-forward on the GPU: linear -> gelu -> linear in ONE command
buffer, with the (seq x ff) intermediate kept resident on the GPU instead of
round-tripping to the host.  Weights stay resident across tokens.  Falls back
to the unfused ops (still GPU when the backend is on) if the server is down."
  (if (not (nelisp-gpu-server-up-p))
      (photon-tensor-linear
       (photon-tensor-gelu (photon-tensor-linear x (plist-get layer :w1)
                                                 (plist-get layer :b1)))
       (plist-get layer :w2) (plist-get layer :b2))
    (let* ((xsh (photon-tensor-shape x)) (seq (car xsh)) (dim (nth 1 xsh))
           (w1 (plist-get layer :w1)) (b1 (plist-get layer :b1))
           (w2 (plist-get layer :w2)) (b2 (plist-get layer :b2))
           (ff (car (photon-tensor-shape w1)))
           (hw1 (nelisp-gpu-server-resident (photon-tensor-data w1)))
           (hb1 (nelisp-gpu-server-resident (photon-tensor-data b1)))
           (hw2 (nelisp-gpu-server-resident (photon-tensor-data w2)))
           (hb2 (nelisp-gpu-server-resident (photon-tensor-data b2)))
           (out (car (nelisp-gpu-server-batch
                      (list (cons 'in (photon-tensor-data x))  ; 0 X  (seq x dim)
                            (list 'res hw1 (* ff dim))         ; 1 W1
                            (list 'res hb1 ff)                 ; 2 B1
                            (cons 'tmp (* seq ff))             ; 3 H  (seq x ff)
                            (list 'res hw2 (* dim ff))         ; 4 W2
                            (list 'res hb2 dim)                ; 5 B2
                            (cons 'out (* seq dim)))           ; 6 O  (seq x dim)
                      (list
                       (list 'linear '(0 1 2 3) (list seq dim ff) (/ (+ (* seq ff) 63) 64))
                       (list 'gelu '(3 3) (list (* seq ff)) (/ (+ (* seq ff) 63) 64))
                       (list 'linear '(3 4 5 6) (list seq ff dim) (/ (+ (* seq dim) 63) 64)))))))
      (photon-tensor (list seq dim) out))))

(defun photon-transformer--attention-gpu (x layer heads)
  "Fused causal multi-head self-attention on the GPU.  The whole block --
QKV projections, per-head transpose/scores/scale/causal-mask/softmax/context,
and the output projection -- is recorded into ONE command buffer via OP_BATCH,
keeping every intermediate (q/k/v, per-head scores, context) resident on the
GPU instead of round-tripping to the host.  Weights stay resident across
tokens.  Falls back to the per-op path when the server is down."
  (let* ((xsh (photon-tensor-shape x)) (seq (car xsh)) (dim (nth 1 xsh))
         (hd (/ dim heads)) (scl (/ 1.0 (sqrt (float hd))))
         (wq (plist-get layer :wq)) (wk (plist-get layer :wk))
         (wv (plist-get layer :wv)) (wo (plist-get layer :wo)))
    (if (not (nelisp-gpu-server-up-p))
        (let* ((q (photon-tensor-linear x wq)) (k (photon-tensor-linear x wk))
               (v (photon-tensor-linear x wv))
               (ctx (photon-tensor-create (list seq dim) 0.0)) (h 0))
          (while (< h heads)
            (let* ((c0 (* h hd))
                   (qh (photon-transformer--slice-cols q c0 hd))
                   (kh (photon-transformer--slice-cols k c0 hd))
                   (vh (photon-transformer--slice-cols v c0 hd))
                   (scores (photon-tensor-scale
                            (photon-tensor-matmul qh (photon-tensor-transpose kh)) scl)))
              (photon-transformer--causal-mask scores)
              (photon-transformer--set-cols
               ctx (photon-tensor-matmul (photon-tensor-softmax-rows scores) vh) c0))
            (setq h (1+ h)))
          (photon-tensor-linear ctx wo))
      (let* ((hq (nelisp-gpu-server-resident (photon-tensor-data wq)))
             (hk (nelisp-gpu-server-resident (photon-tensor-data wk)))
             (hv (nelisp-gpu-server-resident (photon-tensor-data wv)))
             (ho (nelisp-gpu-server-resident (photon-tensor-data wo)))
             (sd (* seq dim)) (sh (* seq hd)) (ss (* seq seq))
             (gsd (/ (+ sd 63) 64)) (gsh (/ (+ sh 63) 64)) (gss (/ (+ ss 63) 64))
             (slots (list (cons 'in (photon-tensor-data x))   ; 0  x
                          (list 'res hq (* dim dim))          ; 1  wq
                          (list 'res hk (* dim dim))          ; 2  wk
                          (list 'res hv (* dim dim))          ; 3  wv
                          (list 'res ho (* dim dim))          ; 4  wo
                          (cons 'in (make-vector dim 0.0))    ; 5  zero bias
                          (cons 'tmp sd)                      ; 6  q
                          (cons 'tmp sd)                      ; 7  k
                          (cons 'tmp sd)                      ; 8  v
                          (cons 'in (vector scl))             ; 9  scale scalar
                          (cons 'tmp sd)                      ; 10 ctx
                          (cons 'tmp sh)                      ; 11 qh
                          (cons 'tmp sh)                      ; 12 kh
                          (cons 'tmp sh)                      ; 13 vh
                          (cons 'tmp sh)                      ; 14 kt (hd x seq)
                          (cons 'tmp ss)                      ; 15 scores
                          (cons 'tmp ss)                      ; 16 softmax
                          (cons 'tmp sh)                      ; 17 ctxh
                          (cons 'out sd)))                    ; 18 out
             (disp (list (list 'linear '(0 1 5 6) (list seq dim dim) gsd)
                         (list 'linear '(0 2 5 7) (list seq dim dim) gsd)
                         (list 'linear '(0 3 5 8) (list seq dim dim) gsd)))
             (h 0))
        (while (< h heads)
          (let ((c0 (* h hd)))
            (setq disp
                  (append disp
                          (list (list 'slice-cols '(6 11) (list seq dim hd c0) gsh)
                                (list 'slice-cols '(7 12) (list seq dim hd c0) gsh)
                                (list 'slice-cols '(8 13) (list seq dim hd c0) gsh)
                                (list 'transpose '(12 14) (list seq hd) gsh)
                                (list 'matmul '(11 14 15) (list seq hd seq) gss)
                                (list 'scale '(15 9 15) (list ss) gss)
                                (list 'causal-mask '(15) (list seq) gss)
                                (list 'softmax '(15 16) (list seq seq) (/ (+ seq 63) 64))
                                (list 'matmul '(16 13 17) (list seq seq hd) gsh)
                                (list 'set-cols '(10 17) (list seq dim hd c0) gsh)))))
          (setq h (1+ h)))
        (setq disp (append disp (list (list 'linear '(10 4 5 18) (list seq dim dim) gsd))))
        (photon-tensor (list seq dim) (car (nelisp-gpu-server-batch slots disp)))))))

(defconst photon-tensor-gpu--ops
  '((photon-tensor-matmul          . photon-tensor-matmul-gpu)
    (photon-tensor-linear          . photon-tensor-linear-gpu)
    (photon-tensor-softmax-rows    . photon-tensor-softmax-rows-gpu)
    (photon-tensor-layernorm-rows  . photon-tensor-layernorm-rows-gpu)
    (photon-tensor-gelu            . photon-tensor-gelu-gpu)
    (photon-transformer--ffn       . photon-transformer--ffn-gpu)
    (photon-transformer--attention . photon-transformer--attention-gpu))
  "CPU op -> GPU op overrides installed by the GPU backend.")

(defvar photon-tensor-gpu--saved nil
  "Saved CPU function cells, for restoring with `photon-tensor-use-cpu-backend'.")

(defun photon-tensor-use-gpu-backend ()
  "Route the heavy photon-tensor ops through the nelisp-gpu GPU kernels."
  (unless photon-tensor-gpu--saved
    (setq photon-tensor-gpu--saved
          (mapcar (lambda (c) (cons (car c) (symbol-function (car c))))
                  photon-tensor-gpu--ops)))
  (dolist (c photon-tensor-gpu--ops) (fset (car c) (cdr c)))
  'gpu)

(defun photon-tensor-use-cpu-backend ()
  "Restore the pure-elisp photon-tensor ops."
  (when photon-tensor-gpu--saved
    (dolist (c photon-tensor-gpu--saved) (fset (car c) (cdr c))))
  'cpu)

(provide 'photon-tensor-gpu)
;;; photon-tensor-gpu.el ends here
