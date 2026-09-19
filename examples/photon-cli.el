;;; photon-cli.el --- Train, save, load, and generate Photon models -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun nil)
(defvar photon-cli-command nil)
(defvar photon-cli-train-path nil)
(defvar photon-cli-eval-path nil)
(defvar photon-cli-model-path nil)
(defvar photon-cli-prompt nil)
(defvar photon-cli-steps nil)
(defvar photon-cli-train-limit nil)
(defvar photon-cli-eval-limit nil)
(defvar photon-cli-diagnostic-limit nil)
(defvar photon-cli-full-distill-limit nil)
(defvar photon-cli-context-size nil)
(defvar photon-cli-hidden-size nil)
(defvar photon-cli-chunk-size nil)
(defvar photon-cli-epochs nil)
(defvar photon-cli-learning-rate nil)
(defvar photon-cli-max-piece-size nil)
(setq photon-large-generation-benchmark-autorun nil)

(load "examples/large-generation-benchmark.el")

(defun photon-cli-env-int (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-cli-option-int (value name default)
  (if value
      (if (stringp value) (string-to-number value) value)
    (photon-cli-env-int name default)))

(defun photon-cli-env-float (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-cli-option-float (value name default)
  (if value
      (if (stringp value) (string-to-number value) value)
    (photon-cli-env-float name default)))

(defun photon-cli-sampling-options ()
  (list (cons 'mode (intern (or (getenv "PHOTON_SAMPLE_MODE") "sample")))
        (cons 'temperature (photon-cli-env-float "PHOTON_TEMPERATURE" 0.85))
        (cons 'top-k (photon-cli-env-int "PHOTON_TOP_K" 5))
        (cons 'repetition-penalty
              (photon-cli-env-float "PHOTON_REPETITION_PENALTY" 1.10))
        (cons 'ngram-bias (photon-cli-env-float "PHOTON_NGRAM_BIAS" 0.25))
        (cons 'repeat-ngram-bias
              (photon-cli-env-float "PHOTON_REPEAT_NGRAM_BIAS" 0.35))
        (cons 'seed (photon-cli-env-int "PHOTON_SEED" 17))))

(defun photon-cli-env-args ()
  (let ((command (or photon-cli-command (getenv "PHOTON_COMMAND"))))
    (when (and command (> (length command) 0))
      (cond
       ((string= command "train")
        (list command
              (or photon-cli-train-path (getenv "PHOTON_TRAIN_PATH") "")
              (or photon-cli-eval-path (getenv "PHOTON_EVAL_PATH") "")
              (or photon-cli-model-path (getenv "PHOTON_MODEL_PATH") "")))
       ((string= command "generate")
        (list command
              (or photon-cli-model-path (getenv "PHOTON_MODEL_PATH") "")
              (or photon-cli-prompt (getenv "PHOTON_PROMPT") "photon stream")
              (or photon-cli-steps (getenv "PHOTON_STEPS") "32")))
       ((string= command "repl")
        (list command
              (or photon-cli-model-path (getenv "PHOTON_MODEL_PATH") "")
              (or photon-cli-steps (getenv "PHOTON_STEPS") "32")))
       ((string= command "eval")
        (list command
              (or photon-cli-model-path (getenv "PHOTON_MODEL_PATH") "")
              (or photon-cli-eval-path (getenv "PHOTON_EVAL_PATH") "")))
       (t (list command))))))

(defun photon-cli-train (args)
  (let* ((root ".")
         (train-path
          (expand-file-name
           (if (and (nth 0 args) (> (length (nth 0 args)) 0))
               (nth 0 args)
             "data/stream-large-sample.txt")
           root))
         (eval-path
          (expand-file-name
           (if (and (nth 1 args) (> (length (nth 1 args)) 0))
               (nth 1 args)
             "data/stream-large-eval-sample.txt")
           root))
         (model-path
          (expand-file-name
           (if (and (nth 2 args) (> (length (nth 2 args)) 0))
               (nth 2 args)
             "target/photon-large-model.el")
           root)))
    (setq photon-large-readout-diagnostic-token-limit
          (photon-cli-option-int
           photon-cli-diagnostic-limit "PHOTON_DIAGNOSTIC_LIMIT" 160))
    (setq photon-large-readout-full-distill-token-limit
          (photon-cli-option-int
           photon-cli-full-distill-limit "PHOTON_FULL_DISTILL_LIMIT" 160))
    (setq photon-large-readout-train-token-limit
          (photon-cli-option-int photon-cli-train-limit
                                 "PHOTON_TRAIN_LIMIT" 480))
    (setq photon-large-readout-eval-token-limit
          (photon-cli-option-int photon-cli-eval-limit
                                 "PHOTON_EVAL_LIMIT" 320))
    (setq photon-large-readout-context-size
          (photon-cli-option-int photon-cli-context-size
                                 "PHOTON_CONTEXT_SIZE" 4))
    (setq photon-large-readout-hidden-size
          (photon-cli-option-int photon-cli-hidden-size
                                 "PHOTON_HIDDEN_SIZE" 10))
    (setq photon-large-readout-chunk-size
          (photon-cli-option-int photon-cli-chunk-size
                                 "PHOTON_CHUNK_SIZE" 3))
    (setq photon-large-readout-epochs
          (photon-cli-option-int photon-cli-epochs "PHOTON_EPOCHS" 1))
    (setq photon-large-readout-learning-rate
          (photon-cli-option-float
           photon-cli-learning-rate "PHOTON_LEARNING_RATE" 0.18))
    (setq photon-large-readout-max-piece-size
          (photon-cli-option-int
           photon-cli-max-piece-size "PHOTON_MAX_PIECE_SIZE" 4))
    (setq photon-large-readout-distill-candidates
          '(full-scaled start-long-classifier-teacher-scaled
            start-long-ngram-teacher-scaled full-unscaled))
    (setq photon-large-readout-full-distill-selection-after-weight 8.0)
    (setq photon-large-readout-full-distill-start-long-classifier-teacher-weight
          8.0)
    (setq photon-large-readout-full-distill-start-long-ngram-teacher-weight 8.0)
    (let* ((result
            (photon-large-generation-train-light
             train-path eval-path
             photon-large-readout-context-size
             photon-large-readout-hidden-size
             photon-large-readout-chunk-size
             photon-large-readout-epochs
             photon-large-readout-learning-rate
             photon-large-readout-max-piece-size))
           (model (cdr (assq 'model result)))
           (compact (photon-large-readout-compact-result result)))
      (make-directory (file-name-directory model-path) t)
      (photon-save-model model model-path)
      (prin1 (append compact (list (cons 'saved-model model-path))))
      (princ "\n"))))

(defun photon-cli-generate (args)
  (let* ((model-path (nth 0 args))
         (prompt (or (nth 1 args) "photon stream"))
         (steps (string-to-number (or (nth 2 args) "32")))
         (model (photon-load-model model-path)))
    (princ (photon-model-generate-text
            model prompt steps (photon-cli-sampling-options)))
    (princ "\n")))

(defun photon-cli-eval (args)
  (let* ((model-path (nth 0 args))
         (eval-path (nth 1 args))
         (context-size (or (photon-model-metadata-get
                            (photon-load-model model-path) 'context-size)
                           4))
         (model (photon-load-model model-path))
         (text (photon-read-text-file eval-path))
         (max-piece-size
          (or (photon-model-metadata-get model 'wordpiece-max-piece-size) 4))
         (vocab (photon-model-metadata-get model 'vocab))
         (tokens (photon-encode-wordpiece-text vocab text max-piece-size)))
    (prin1
     (list
      (cons 'model model-path)
      (cons 'eval-path eval-path)
      (cons 'eval (photon-evaluate model tokens context-size))
      (cons 'fallback
            (photon-evaluate-fallback model tokens context-size))))
    (princ "\n")))

(defun photon-cli-repl (args)
  (let* ((model-path (nth 0 args))
         (steps (string-to-number (or (nth 1 args) "32")))
         (model (photon-load-model model-path))
         line)
    (princ "Photon REPL. Empty line or :quit exits.\n")
    (while (progn
             (princ "> ")
             (setq line (read-string "" nil nil nil))
             (and line (> (length line) 0)
                  (not (string= line ":quit"))))
      (princ
       (photon-model-generate-text
        model line steps (photon-cli-sampling-options)))
      (princ "\n"))))

(defun photon-cli-usage ()
  (princ
   "Usage:
  emacs -Q --batch -L lisp --script examples/photon-cli.el train [TRAIN] [EVAL] [MODEL]
  emacs -Q --batch -L lisp --script examples/photon-cli.el generate MODEL PROMPT [STEPS]
  emacs -Q --batch -L lisp --script examples/photon-cli.el repl MODEL [STEPS]
  emacs -Q --batch -L lisp --script examples/photon-cli.el eval MODEL EVAL

  nelisp --eval '(progn (setq photon-cli-command \"generate\"
                              photon-cli-model-path \"target/photon-large-model.el\"
                              photon-cli-prompt \"photon stream\")
                         (load \"examples/photon-cli.el\"))'

Environment: PHOTON_TRAIN_LIMIT, PHOTON_EVAL_LIMIT, PHOTON_TOP_K,
PHOTON_TEMPERATURE, PHOTON_NGRAM_BIAS, PHOTON_REPEAT_NGRAM_BIAS,
PHOTON_COMMAND, PHOTON_MODEL_PATH, PHOTON_PROMPT, PHOTON_STEPS.
"))

(defun photon-cli-command-args ()
  (or command-line-args-left
      (let ((args command-line-args)
            found)
        (while (and args (not found))
          (if (and (string= (car args) "-scriptload")
                   (cdr args))
              (setq found (cddr args))
            (setq args (cdr args))))
        found)))

(let* ((cli-args (or (photon-cli-env-args)
                     (photon-cli-command-args)))
       (command (car cli-args))
       (args (cdr cli-args)))
  (cond
   ((string= command "train") (photon-cli-train args))
   ((string= command "generate") (photon-cli-generate args))
   ((string= command "repl") (photon-cli-repl args))
   ((string= command "eval") (photon-cli-eval args))
   (t (photon-cli-usage))))

;;; photon-cli.el ends here
