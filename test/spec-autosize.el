;;; spec-autosize.el --- tests for the photon-spec autosizer  -*- lexical-binding: t; -*-
;; Run from the nelisp-photon root:
;;   emacs -Q --batch -L lisp -l test/spec-autosize.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-spec)
(require 'photon-transformer)
(require 'photon-tensor)

(defvar spec--fail 0)
(defun spec--ck (name ok)
  (princ (format "%-44s %s\n" name (if ok "PASS"
                                     (progn (setq spec--fail (1+ spec--fail)) "FAIL")))))

(let ((gib (* 1024 1024 1024)))
  ;; 1. detection returns sane positive numbers for this host
  (let ((hw (photon-spec-detect)))
    (spec--ck "detect: ram>0 and cores>=1"
              (and (> (plist-get hw :ram-bytes) 0)
                   (>= (plist-get hw :cores) 1))))

  ;; 2. a small box gets a smaller model than a big box
  (let ((small (photon-spec-autosize :ram-bytes (* 1 gib)   :cores 64))
        (large (photon-spec-autosize :ram-bytes (* 256 gib) :cores 64)))
    (spec--ck "scale: large dim > small dim"
              (> (plist-get large :dim) (plist-get small :dim)))
    (spec--ck "scale: large layers >= small layers"
              (>= (plist-get large :layers) (plist-get small :layers))))

  ;; 3. dim is non-decreasing as RAM grows (cores held high)
  (let ((prev -1) (ok t))
    (dolist (g '(1 2 4 8 16 32 64 128 256))
      (let ((d (plist-get (photon-spec-autosize :ram-bytes (* g gib) :cores 64) :dim)))
        (when (< d prev) (setq ok nil))
        (setq prev d)))
    (spec--ck "monotonic: dim non-decreasing in RAM" ok))

  ;; 4. dim is non-decreasing as cores grow (RAM held huge)
  (let ((prev -1) (ok t))
    (dolist (c '(1 2 3 4 6 8 12 16 32))
      (let ((d (plist-get (photon-spec-autosize :ram-bytes (* 256 gib) :cores c) :dim)))
        (when (< d prev) (setq ok nil))
        (setq prev d)))
    (spec--ck "monotonic: dim non-decreasing in cores" ok))

  ;; 5. every chosen config is structurally valid and within the RAM budget
  (let ((ok-valid t) (ok-budget t))
    (dolist (g '(1 2 4 8 16 32 64 128 256))
      (let* ((cfg (photon-spec-autosize :ram-bytes (* g gib) :cores 64))
             (dim (plist-get cfg :dim)) (h (plist-get cfg :heads)))
        (unless (and (> dim 0) (>= h 1) (= 0 (mod dim h))
                     (> (plist-get cfg :layers) 0)
                     (> (plist-get cfg :context) 0)
                     (> (plist-get cfg :ff) 0))
          (setq ok-valid nil))
        (when (> (photon-spec--param-bytes cfg photon-spec-default-vocab)
                 (/ (* (* g gib) 0.25) 1.2))
          (setq ok-budget nil))))
    (spec--ck "valid: dim%heads==0 and all positive" ok-valid)
    (spec--ck "budget: weight bytes <= RAM share" ok-budget))

  ;; 6. train mode reserves more headroom, so it never picks a bigger tier
  (let ((inf (photon-spec-autosize :ram-bytes (* 8 gib) :cores 64 :mode 'infer))
        (trn (photon-spec-autosize :ram-bytes (* 8 gib) :cores 64 :mode 'train)))
    (spec--ck "mode: train tier <= infer tier"
              (<= (plist-get trn :tier) (plist-get inf :tier))))

  ;; 7. the autosized config actually builds and runs a forward pass
  (let* ((cfg (photon-spec-autosize :ram-bytes (* 1 gib) :cores 2 :vocab 48))
         (model (apply #'photon-transformer-create cfg))
         (vocab (plist-get cfg :vocab))
         (out (photon-transformer-generate model '(1 2 3) 2)))
    (spec--ck "build: autosized model generates valid tokens"
              (and (= (length out) 5)
                   (catch 'bad
                     (dolist (tk out t)
                       (when (or (< tk 0) (>= tk vocab)) (throw 'bad nil))))))))

(princ (format "SPEC-AUTOSIZE %s (%d failures)\n"
               (if (= spec--fail 0) "ALL-PASS" "HAS-FAILURES") spec--fail))
(kill-emacs (if (= spec--fail 0) 0 1))
;;; spec-autosize.el ends here
