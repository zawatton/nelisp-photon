;;; weights-load.el --- tests for photon-weights save/load determinism  -*- lexical-binding: t; -*-
;; Run from the nelisp-photon root:
;;   emacs -Q --batch -L lisp -l test/weights-load.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-tensor)
(require 'photon-transformer)
(require 'photon-weights)

(defvar wl--fail 0)
(defun wl--ck (name ok)
  (princ (format "%-50s %s\n" name (if ok "PASS"
                                     (progn (setq wl--fail (1+ wl--fail)) "FAIL")))))

(defun wl--bump (tn)
  "Perturb tensor TN in place so the model differs from a fresh build."
  (let* ((d (photon-tensor-data tn)) (n (length d)) (i 0))
    (while (< i n) (aset d i (+ 0.37 (* 1.5 (aref d i)))) (setq i (1+ i)))))

(defun wl--perturb (model)
  "Perturb every weight tensor of MODEL."
  (dolist (k photon-weights--top-keys) (wl--bump (plist-get model k)))
  (dolist (ly (plist-get model :layers))
    (dolist (k photon-weights--layer-keys) (wl--bump (plist-get ly k))))
  model)

(defun wl--maxdiff (ta tb)
  "Max abs difference between two tensors' data vectors."
  (let* ((a (photon-tensor-data ta)) (b (photon-tensor-data tb))
         (n (length a)) (m 0.0) (i 0))
    (while (< i n)
      (let ((e (abs (- (aref a i) (aref b i))))) (when (> e m) (setq m e)))
      (setq i (1+ i)))
    m))

(let* ((cfg '(:vocab 24 :dim 16 :heads 2 :context 8 :layers 2 :ff 32))
       (toks '(1 2 0 3 5))
       (orig (wl--perturb (apply #'photon-transformer-create cfg)))
       (ref  (photon-transformer-forward orig toks))
       (tmp  (make-temp-file "photon-weights" nil ".el")))
  (photon-weights-save orig tmp)
  (let* ((loaded (photon-weights-load tmp))
         (got (photon-transformer-forward loaded toks))
         (fresh (apply #'photon-transformer-create cfg))
         (fresh-out (photon-transformer-forward fresh toks)))
    ;; 1. forward of the loaded model is bit-identical to the original
    (wl--ck "determinism: forward(loaded) == forward(orig)"
            (= 0.0 (wl--maxdiff got ref)))
    ;; 2. a stored weight tensor matches the original exactly
    (wl--ck "weights: loaded wte == orig wte"
            (= 0.0 (wl--maxdiff (plist-get loaded :wte) (plist-get orig :wte))))
    ;; 3. the load really applied file weights (differs from a fresh build)
    (wl--ck "load applied file weights (!= fresh build)"
            (> (wl--maxdiff got fresh-out) 1.0e-6))
    ;; 4. loaded model is independent: mutating orig does not change loaded
    (wl--bump (plist-get orig :wte))
    (let ((got2 (photon-transformer-forward loaded toks)))
      (wl--ck "independence: mutating orig leaves loaded intact"
              (= 0.0 (wl--maxdiff got2 ref)))))
  (delete-file tmp))

(princ (format "WEIGHTS-LOAD %s (%d failures)\n"
               (if (= wl--fail 0) "ALL-PASS" "HAS-FAILURES") wl--fail))
(kill-emacs (if (= wl--fail 0) 0 1))
;;; weights-load.el ends here
