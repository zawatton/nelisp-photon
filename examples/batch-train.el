;;; batch-train.el --- Batch training/checkpoint example -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-elisp-vector-backend)

(let* ((dataset (photon-load-batch-dataset-text-file
                 "data/stream-sample.txt" 4))
       (config (photon-make-config
                (length (cdr (assq 'vocab dataset))) 8 4 2))
       (model (photon-make-model config))
       (path "target/photon-batch-checkpoint.el")
       (history-path "target/photon-batch-history.el")
       result
       resumed)
  (make-directory "target" t)
  (setq result
        (photon-train-batch-dataset
         model dataset 2 0.25 8
         (list (cons 'shuffle t)
               (cons 'checkpoint-path path)
               (cons 'history-path history-path)
               (cons 'train-split-numerator 3)
               (cons 'train-split-denominator 4))))
  (setq resumed
        (photon-train-batch-dataset
         model dataset 1 0.10 8
         (list (cons 'resume t)
               (cons 'checkpoint-path path)
               (cons 'history-path history-path)
               (cons 'train-split-numerator 3)
               (cons 'train-split-denominator 4))))
  (prin1
   (list (cons 'checkpoint-exists (file-exists-p path))
         (cons 'history-exists (file-exists-p history-path))
         (cons 'dataset-name (cdr (assq 'dataset-name result)))
         (cons 'token-count (cdr (assq 'token-count result)))
         (cons 'pair-count (cdr (assq 'pair-count result)))
         (cons 'train-pairs (cdr (assq 'train-pairs result)))
         (cons 'eval-pairs (cdr (assq 'eval-pairs result)))
         (cons 'history (cdr (assq 'history result)))
         (cons 'resumed-history (cdr (assq 'history resumed)))))
  (princ "\n"))

;;; batch-train.el ends here
