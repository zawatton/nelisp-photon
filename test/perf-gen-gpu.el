;;; perf-gen-gpu.el --- persistent-server GPU generation perf + correctness  -*- lexical-binding: t; -*-
;; Run from nelisp-photon root (needs ../nelisp-gpu/host/vkserver built):
;;   emacs -Q --batch -L lisp -l test/perf-gen-gpu.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-gpu/lisp"))
(require 'photon-transformer)
(require 'photon-tensor-gpu)
(require 'nelisp-gpu-server)
(setq nelisp-gpu-server-bin (expand-file-name "../nelisp-gpu/host/vkserver"))

(nelisp-gpu-server-start)

;; correctness: a matmul via the persistent server vs the photon-tensor oracle
(let* ((M 8) (K 6) (N 5)
       (A (nelisp-gpu-gen (* M K) (lambda (i) (* 0.1 (- (mod i 7) 3)))))
       (B (nelisp-gpu-gen (* K N) (lambda (i) (* 0.1 (- (mod i 5) 2)))))
       (C0 (make-vector (* M N) 0.0))
       (ref (photon-tensor-data
             (photon-tensor-matmul (photon-tensor (list M K) A)
                                   (photon-tensor (list K N) B))))
       (got (nth 2 (nelisp-gpu-run 'matmul (list A B C0)
                                   (list M K N) (/ (+ (* M N) 63) 64))))
       (err 0.0) (i 0))
  (while (< i (length ref))
    (let ((e (abs (- (aref ref i) (aref got i))))) (when (> e err) (setq err e)))
    (setq i (1+ i)))
  (princ (format "server-matmul max_err=%.3e %s\n" err (if (< err 1.0e-3) "OK" "BAD"))))

;; perf: 5-token generation on the GPU backend through the server
(let* ((m (photon-transformer-create :vocab 16 :dim 8 :heads 2
                                     :context 8 :layers 2 :ff 16))
       (prompt '(1 2 3)))
  (photon-tensor-use-gpu-backend)
  ;; one warm-up token so all pipelines are created before timing
  (photon-transformer-generate m prompt 1)
  (let* ((t0 (float-time))
         (gen (photon-transformer-generate m prompt 5))
         (dt (- (float-time) t0)))
    (photon-tensor-use-cpu-backend)
    (princ (format "gpu-gen=%S\ngen5-time=%.3fs\n" gen dt))
    (princ (format "PERF-5TOK=%s\n" (if (< dt 3.0) "PASS" "FAIL")))))

(nelisp-gpu-server-stop)
;;; perf-gen-gpu.el ends here
