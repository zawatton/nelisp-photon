;;; gpu-fusion-verify.el --- verify fused-FFN GPU batch == unfused, CPU == GPU  -*- lexical-binding: t; -*-
;; Run from the nelisp-photon root (needs ../nelisp-gpu/host/vkserver built):
;;   emacs -Q --batch -L lisp -L ../nelisp-gpu/lisp -l test/gpu-fusion-verify.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-gpu/lisp"))
(require 'photon-tensor)
(require 'photon-transformer)
(require 'photon-tensor-gpu)
(require 'nelisp-gpu-server)
(setq nelisp-gpu-server-bin (expand-file-name "../nelisp-gpu/host/vkserver"))

(defvar gf--fail 0)
(defun gf--ck (name ok extra)
  (princ (format "%-44s %s  %s\n" name (if ok "PASS"
                                         (progn (setq gf--fail (1+ gf--fail)) "FAIL"))
                 (or extra ""))))
(defun gf--maxdiff (ta tb)
  (let* ((a (photon-tensor-data ta)) (b (photon-tensor-data tb))
         (n (length a)) (m 0.0) (i 0))
    (while (< i n)
      (let ((e (abs (- (aref a i) (aref b i))))) (when (> e m) (setq m e)))
      (setq i (1+ i)))
    m))

(let* ((cfg '(:vocab 24 :dim 32 :heads 4 :context 16 :layers 3 :ff 64))
       (model (apply #'photon-transformer-create cfg))
       (layer (car (plist-get model :layers)))
       (seq 5) (dim 32)
       (b (photon-tensor (list seq dim)
                         (let ((v (make-vector (* seq dim) 0.0)) (i 0))
                           (while (< i (* seq dim))
                             (aset v i (* 0.1 (- (mod i 9) 4))) (setq i (1+ i)))
                           v)))
       (toks '(1 2 3 4 5))
       (cpu-ref (progn (photon-tensor-use-cpu-backend)
                       (photon-transformer-forward model toks)))
       (cpu-attn (photon-transformer--attention b layer 4)))
  (nelisp-gpu-server-start)
  (photon-tensor-use-gpu-backend)
  ;; warm pipelines / resident weights
  (photon-transformer-forward model toks)
  ;; 1. fused FFN (batch, one command buffer) vs unfused GPU ops (same f32 precision)
  (let* ((fused (photon-transformer--ffn-gpu b layer))
         (unfused (photon-tensor-linear
                   (photon-tensor-gelu
                    (photon-tensor-linear b (plist-get layer :w1) (plist-get layer :b1)))
                   (plist-get layer :w2) (plist-get layer :b2)))
         (d (gf--maxdiff fused unfused)))
    (gf--ck "fused FFN == unfused GPU FFN" (< d 1.0e-3) (format "maxdiff=%.2e" d)))
  ;; 1b. fused attention block (one command buffer) vs CPU attention
  (let* ((gpu-attn (photon-transformer--attention-gpu b layer 4))
         (d (gf--maxdiff gpu-attn cpu-attn)))
    (gf--ck "fused attention ~= CPU attention" (< d 1.0e-2) (format "maxdiff=%.2e" d)))
  ;; 2. full GPU forward (with fused FFN) vs CPU forward
  (let* ((gpu (photon-transformer-forward model toks))
         (d (gf--maxdiff gpu cpu-ref)))
    (gf--ck "GPU forward ~= CPU forward" (< d 5.0e-2) (format "maxdiff=%.2e" d)))
  (photon-tensor-use-cpu-backend)
  (nelisp-gpu-server-stop))

(princ (format "GPU-FUSION-VERIFY %s (%d failures)\n"
               (if (= gf--fail 0) "ALL-PASS" "HAS-FAILURES") gf--fail))
(kill-emacs (if (= gf--fail 0) 0 1))
;;; gpu-fusion-verify.el ends here
