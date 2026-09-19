;;; e2e-gpu-bench.el --- end-to-end pipeline on the GPU backend + speed bench  -*- lexical-binding: t; -*-
;; Run from the nelisp-photon root (needs ../nelisp-gpu/host/vkserver built):
;;   emacs -Q --batch -L lisp -l test/e2e-gpu-bench.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-gpu/lisp"))
(require 'photon-tensor)
(require 'photon-transformer)
(require 'photon-bpe)
(require 'photon-spec)
(require 'photon-infer)
(require 'photon-tensor-gpu)
(require 'nelisp-gpu-server)

(setq nelisp-gpu-server-bin (expand-file-name "../nelisp-gpu/host/vkserver"))

(defvar e2e--fail 0)
(defvar e2e--cpu-dt nil)
(defun e2e--ck (name ok)
  (princ (format "%-46s %s\n" name (if ok "PASS"
                                     (progn (setq e2e--fail (1+ e2e--fail)) "FAIL")))))

(let* ((corpus '("the quick brown fox jumps over the lazy dog"
                 "the cat sat on the mat and the dog ran"
                 "to be or not to be that is the question"
                 "all work and no play makes jack a dull boy"))
       (bpe (photon-bpe-train corpus 160))
       (vocab (photon-bpe-size bpe))
       ;; autosize a GPU-friendly model, then bound :context for a quick bench
       (auto (photon-spec-autosize :ram-bytes (* 200 (expt 2 20)) :cores 8 :vocab vocab))
       (cfg (append (list :context 64) (copy-sequence auto)))  ; :context (first) wins
       (model (apply #'photon-transformer-create cfg))
       (prompt "the quick brown")
       (steps 12))
  (princ (format "host: %s\n" (photon-spec-describe)))
  (princ (format "bench cfg: vocab=%d dim=%d heads=%d layers=%d ctx=%d ff=%d (autosize tier %d)\n"
                 vocab (plist-get cfg :dim) (plist-get cfg :heads)
                 (plist-get cfg :layers) (plist-get cfg :context)
                 (plist-get cfg :ff) (plist-get auto :tier)))

  ;; --- CPU baseline ---
  (photon-tensor-use-cpu-backend)
  (let* ((t0 (float-time))
         (txt (photon-infer-generate model bpe prompt steps))
         (dt (- (float-time) t0)))
    (e2e--ck "cpu: output is a string" (stringp txt))
    (e2e--ck "cpu: output starts with prompt" (string-prefix-p prompt txt))
    (setq e2e--cpu-dt dt)
    (princ (format "cpu: %d tok in %.3fs = %.2f tok/s\n" steps dt (/ steps dt))))

  ;; --- GPU (persistent vkserver) ---
  (nelisp-gpu-server-start)
  (photon-tensor-use-gpu-backend)
  (photon-infer-generate model bpe prompt 1)  ; warm-up: build all pipelines
  (let* ((t0 (float-time))
         (txt (photon-infer-generate model bpe prompt steps))
         (dt (- (float-time) t0)))
    (photon-tensor-use-cpu-backend)
    (nelisp-gpu-server-stop)
    (e2e--ck "gpu: output is a string" (stringp txt))
    (e2e--ck "gpu: output starts with prompt" (string-prefix-p prompt txt))
    (princ (format "gpu: %d tok in %.3fs = %.2f tok/s\n" steps dt (/ steps dt)))
    (princ (format "gpu sample: %S\n" (substring txt 0 (min 70 (length txt)))))
    (when e2e--cpu-dt
      (princ (format "speedup (cpu/gpu): %.2fx\n" (/ e2e--cpu-dt dt))))))

(princ (format "E2E-GPU-BENCH %s (%d failures)\n"
               (if (= e2e--fail 0) "ALL-PASS" "HAS-FAILURES") e2e--fail))
(kill-emacs (if (= e2e--fail 0) 0 1))
;;; e2e-gpu-bench.el ends here
