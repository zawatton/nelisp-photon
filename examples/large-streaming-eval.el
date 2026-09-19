;;; large-streaming-eval.el --- Larger configurable streaming eval -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun nil)
(setq photon-large-generation-benchmark-autorun nil)

(load "examples/large-generation-benchmark.el")

(defun photon-large-streaming-env-int (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-large-streaming-env-float (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-large-streaming-env-path (name default root)
  (let ((value (getenv name)))
    (expand-file-name
     (if (and value (> (length value) 0)) value default)
     root)))

(defun photon-large-streaming-compact (result)
  (let* ((compact (photon-large-readout-compact-result result))
         (model-path (getenv "PHOTON_MODEL_PATH")))
    (when (and model-path (> (length model-path) 0))
      (make-directory (file-name-directory (expand-file-name model-path)) t)
      (photon-save-model (cdr (assq 'model result))
                         (expand-file-name model-path)))
    (append compact
            (list
             (cons 'train-token-limit photon-large-readout-train-token-limit)
             (cons 'eval-token-limit photon-large-readout-eval-token-limit)
             (cons 'diagnostic-token-limit
                   photon-large-readout-diagnostic-token-limit)
             (cons 'saved-model
                   (and model-path (expand-file-name model-path)))))))

(let* ((root ".")
       (train-path
        (photon-large-streaming-env-path
         "PHOTON_TRAIN_PATH" "data/stream-large-sample.txt" root))
       (eval-path
        (photon-large-streaming-env-path
         "PHOTON_EVAL_PATH" "data/stream-large-eval-sample.txt" root)))
  (setq photon-large-readout-diagnostic-token-limit
        (photon-large-streaming-env-int "PHOTON_DIAGNOSTIC_LIMIT" 160))
  (setq photon-large-readout-full-distill-token-limit
        (photon-large-streaming-env-int "PHOTON_FULL_DISTILL_LIMIT" 160))
  (setq photon-large-readout-train-token-limit
        (photon-large-streaming-env-int "PHOTON_TRAIN_LIMIT" 480))
  (setq photon-large-readout-eval-token-limit
        (photon-large-streaming-env-int "PHOTON_EVAL_LIMIT" 320))
  (setq photon-large-readout-context-size
        (photon-large-streaming-env-int "PHOTON_CONTEXT_SIZE" 4))
  (setq photon-large-readout-hidden-size
        (photon-large-streaming-env-int "PHOTON_HIDDEN_SIZE" 10))
  (setq photon-large-readout-chunk-size
        (photon-large-streaming-env-int "PHOTON_CHUNK_SIZE" 3))
  (setq photon-large-readout-epochs
        (photon-large-streaming-env-int "PHOTON_EPOCHS" 1))
  (setq photon-large-readout-learning-rate
        (photon-large-streaming-env-float "PHOTON_LEARNING_RATE" 0.18))
  (setq photon-large-readout-max-piece-size
        (photon-large-streaming-env-int "PHOTON_MAX_PIECE_SIZE" 4))
  (setq photon-large-readout-distill-candidates
        '(full-scaled start-long-classifier-teacher-scaled
          start-long-ngram-teacher-scaled full-unscaled))
  (setq photon-large-readout-full-distill-selection-after-weight 8.0)
  (setq photon-large-readout-full-distill-start-long-classifier-teacher-weight
        8.0)
  (setq photon-large-readout-full-distill-start-long-ngram-teacher-weight 8.0)
  (let ((result
         (photon-large-generation-train-light
          train-path eval-path
          photon-large-readout-context-size
          photon-large-readout-hidden-size
          photon-large-readout-chunk-size
          photon-large-readout-epochs
          photon-large-readout-learning-rate
          photon-large-readout-max-piece-size)))
    (prin1 (photon-large-streaming-compact result))
    (princ "\n")))

;;; large-streaming-eval.el ends here
