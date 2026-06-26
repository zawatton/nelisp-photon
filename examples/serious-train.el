;;; serious-train.el --- Configurable longer batch training run -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-vector-backend-name (or (getenv "PHOTON_BACKEND")
                                    "elisp-vector"))

(defun photon-serious-env-int (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-serious-env-float (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-serious-history-tail (history count)
  (let* ((len (length history))
         (drop (max 0 (- len count))))
    (nthcdr drop history)))

(let* ((root (expand-file-name ".." (file-name-directory load-file-name)))
       (corpus-path
        (or (getenv "PHOTON_CORPUS")
            (expand-file-name "data/stream-sample.txt" root)))
       (tokenizer-name (or (getenv "PHOTON_TOKENIZER") "char"))
       (tokenizer (if (string= tokenizer-name "wordpiece") 'wordpiece 'char))
       (context-size (photon-serious-env-int "PHOTON_CONTEXT" 6))
       (hidden-size (photon-serious-env-int "PHOTON_HIDDEN" 16))
       (levels (photon-serious-env-int "PHOTON_LEVELS" 3))
       (epochs (photon-serious-env-int "PHOTON_EPOCHS" 5))
       (batch-size (photon-serious-env-int "PHOTON_BATCH" 16))
       (learning-rate (photon-serious-env-float "PHOTON_LR" 0.18))
       (checkpoint-path
        (or (getenv "PHOTON_CHECKPOINT")
            (expand-file-name "target/photon-serious-checkpoint.el" root)))
       (best-checkpoint-path
        (or (getenv "PHOTON_BEST_CHECKPOINT")
            (expand-file-name "target/photon-serious-best.el" root)))
       (history-path
        (or (getenv "PHOTON_HISTORY")
            (expand-file-name "target/photon-serious-history.el" root)))
       (history-csv-path
        (or (getenv "PHOTON_HISTORY_CSV")
            (expand-file-name "target/photon-serious-history.csv" root)))
       (vocab-path
        (or (getenv "PHOTON_VOCAB")
            (and (string= tokenizer-name "wordpiece")
                 (expand-file-name "target/photon-serious-vocab.el" root))))
       (resume (getenv "PHOTON_RESUME"))
       (dataset (photon-load-batch-dataset-text-file
                 corpus-path context-size
                 (list (cons 'tokenizer tokenizer)
                       (cons 'vocab-path vocab-path)
                       (cons 'max-chars
                             (photon-serious-env-int "PHOTON_MAX_CHARS" 0))
                       (cons 'wordpiece-max-piece-size
                             (photon-serious-env-int "PHOTON_PIECE_SIZE" 4)))))
       (config (photon-make-config
                (length (cdr (assq 'vocab dataset)))
                hidden-size context-size levels))
       (model (photon-make-model config))
       result)
  (make-directory (file-name-directory checkpoint-path) t)
  (make-directory (file-name-directory history-path) t)
  (setq result
        (photon-train-batch-dataset
         model dataset epochs learning-rate batch-size
         (list (cons 'shuffle t)
               (cons 'resume resume)
               (cons 'checkpoint-path checkpoint-path)
               (cons 'best-checkpoint-path best-checkpoint-path)
               (cons 'history-path history-path)
               (cons 'history-csv-path history-csv-path)
               (cons 'checkpoint-every
                     (photon-serious-env-int "PHOTON_CHECKPOINT_EVERY" 1))
               (cons 'progress (getenv "PHOTON_PROGRESS"))
               (cons 'train-split-numerator
                     (photon-serious-env-int "PHOTON_TRAIN_NUM" 4))
               (cons 'train-split-denominator
                     (photon-serious-env-int "PHOTON_TRAIN_DEN" 5)))))
  (prin1
   (list (cons 'corpus-path corpus-path)
         (cons 'tokenizer tokenizer)
         (cons 'backend (photon-vector-backend-name))
         (cons 'source-chars (cdr (assq 'source-chars dataset)))
         (cons 'used-chars (cdr (assq 'used-chars dataset)))
         (cons 'token-diagnostics
               (cdr (assq 'token-diagnostics dataset)))
         (cons 'token-count (cdr (assq 'token-count result)))
         (cons 'pair-count (cdr (assq 'pair-count result)))
         (cons 'train-pairs (cdr (assq 'train-pairs result)))
         (cons 'eval-pairs (cdr (assq 'eval-pairs result)))
         (cons 'epochs-requested epochs)
         (cons 'checkpoint-path checkpoint-path)
         (cons 'best-checkpoint-path best-checkpoint-path)
         (cons 'best-epoch (cdr (assq 'best-epoch result)))
         (cons 'history-path history-path)
         (cons 'history-csv-path history-csv-path)
         (cons 'vocab-path vocab-path)
         (cons 'history-tail
               (photon-serious-history-tail
                (cdr (assq 'history result)) 3))))
  (princ "\n"))

;;; serious-train.el ends here
