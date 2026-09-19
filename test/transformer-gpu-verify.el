;;; transformer-gpu-verify.el --- CPU vs GPU transformer forward  -*- lexical-binding: t; -*-
;; Run from nelisp-photon root (needs ../nelisp-gpu/host/vkrun built):
;;   emacs -Q --batch -L lisp -l test/transformer-gpu-verify.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-transformer)
(require 'photon-tensor-gpu)

(let* ((m (photon-transformer-create :vocab 16 :dim 8 :heads 2
                                     :context 8 :layers 2 :ff 16))
       (tokens '(1 2 3 4 5))
       (cpu (photon-tensor-data (photon-transformer-forward m tokens))))
  (photon-tensor-use-gpu-backend)
  (let ((gpu (photon-tensor-data (photon-transformer-forward m tokens))))
    (photon-tensor-use-cpu-backend)
    (let ((err 0.0) (i 0) (n (length cpu)))
      (while (< i n)
        (let ((e (abs (- (aref cpu i) (aref gpu i))))) (when (> e err) (setq err e)))
        (setq i (1+ i)))
      (princ (format "transformer forward  cpu-vs-gpu max_err=%.3e  %s\n"
                     err (if (< err 1.0e-3) "PASS" "FAIL"))))))
;;; transformer-gpu-verify.el ends here
