;;; transformer-smoke.el --- forward smoke test for photon-transformer  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -l test/transformer-smoke.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-transformer)

(let* ((m (photon-transformer-create :vocab 16 :dim 8 :heads 2
                                     :context 8 :layers 2 :ff 16))
       (tokens '(1 2 3 4 5))
       (logits (photon-transformer-forward m tokens))
       (sh (photon-tensor-shape logits))
       (d (photon-tensor-data logits))
       (finite t) (i 0))
  (while (< i (length d))
    (unless (< (abs (aref d i)) 1.0e30) (setq finite nil))
    (setq i (1+ i)))
  ;; next-token softmax must sum to 1
  (let* ((vocab (nth 1 sh))
         (last (photon-transformer-next-token-logits m tokens))
         (p (photon-tensor-softmax-rows last))
         (pd (photon-tensor-data p)) (s 0.0) (j 0))
    (while (< j vocab) (setq s (+ s (aref pd j))) (setq j (1+ j)))
    (princ (format "logits-shape=%S finite=%S next-softmax-sum=%.6f\n" sh finite s))
    (princ (format "TRANSFORMER-SMOKE=%s\n"
                   (if (and finite (equal sh '(5 16)) (< (abs (- s 1.0)) 1.0e-4))
                       "PASS" "FAIL")))))
;;; transformer-smoke.el ends here
