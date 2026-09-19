;;; large-generation-benchmark.el --- Larger stream generation profile probe -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun t)
(defvar photon-large-generation-root "examples/")
(defvar photon-large-readout-diagnostic-token-limit 120)
(defvar photon-large-readout-full-distill-token-limit 120)
(defvar photon-large-readout-long-rare-epochs 1)
(defvar photon-large-readout-prototype-distill-epochs 1)
(defvar photon-large-readout-ngram-long-prototype-distill-epochs 1)
(defvar photon-large-readout-post-ngram-long-prototype-refresh-epochs 1)
(defvar photon-large-readout-long-piece-prototype-refresh-epochs 0)
(defvar photon-large-readout-full-distill-epochs 2)
(defvar photon-large-readout-train-token-limit nil)
(defvar photon-large-readout-eval-token-limit nil)
(defvar photon-large-readout-context-size 4)
(defvar photon-large-readout-hidden-size 8)
(defvar photon-large-readout-chunk-size 3)
(defvar photon-large-readout-epochs 1)
(defvar photon-large-readout-learning-rate 0.18)
(defvar photon-large-readout-max-piece-size 4)
(defvar photon-large-readout-output-head-prototype-limit nil)
(defvar photon-large-readout-output-head-prototype-support-weight nil)
(defvar photon-large-readout-output-head-prototype-long-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-short-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-start-long-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-continuation-long-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-start-short-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-continuation-short-piece-weight nil)
(defvar photon-large-readout-output-head-prototype-negative-weight nil)
(defvar photon-large-readout-output-head-prototype-negative-limit nil)
(defvar photon-large-readout-output-head-prototype-ngram-long-token-mix nil)
(defvar photon-large-readout-output-head-prototype-ngram-long-context-mix nil)
(defvar photon-large-readout-output-head-prototype-ngram-long-anchor-mix nil)
(defvar photon-large-readout-output-head-prototype-ngram-long-contrastive-rate nil)
(defvar photon-large-readout-output-head-prototype-ngram-long-state-path-rate nil)
(defvar photon-large-readout-start-long-positive-distill-weight nil)
(defvar photon-large-readout-start-long-positive-distill-epochs nil)
(defvar photon-large-readout-start-long-positive-distill-top-k nil)
(defvar photon-large-readout-start-long-projected-prototype-distill-weight nil)
(defvar photon-large-readout-start-long-projected-prototype-distill-epochs nil)
(defvar photon-large-readout-start-long-projected-prototype-distill-top-k nil)
(defvar photon-large-readout-start-long-prototype-merge-rate nil)
(defvar photon-large-readout-start-long-prototype-merge-epochs nil)
(defvar photon-large-readout-start-long-prototype-merge-top-k nil)
(defvar photon-large-readout-start-long-prototype-merge-margin nil)
(defvar photon-large-readout-start-long-prototype-merge-target-bias-rate nil)
(defvar photon-large-readout-start-long-prototype-merge-rival-bias-rate nil)
(defvar photon-large-readout-start-long-prototype-merge-score-bonus-rate nil)
(defvar photon-large-readout-start-long-prototype-merge-dominant-rival-bias-rate
  nil)
(defvar photon-large-readout-start-long-prototype-merge-dominant-rival-min-count
  nil)
(defvar photon-large-readout-start-long-prototype-merge-state-separation-rate
  nil)
(defvar photon-large-readout-start-long-prototype-merge-dense-head-rate nil)
(defvar photon-large-readout-start-long-prototype-merge-rival-line-search-step
  nil)
(defvar photon-large-readout-start-long-prototype-merge-rival-line-search-max
  nil)
(defvar photon-large-readout-start-long-prototype-merge-rival-negative-weight nil)
(defvar photon-large-readout-start-long-prototype-merge-rivals nil)
(defvar photon-large-readout-start-long-prototype-merge-pocket-enabled nil)
(defvar photon-large-readout-start-long-prototype-merge-candidate-search-enabled nil)
(defvar photon-large-readout-start-long-prototype-merge-candidate-limit nil)
(defvar photon-large-readout-start-long-prototype-merge-global-tolerance nil)
(defvar photon-large-readout-start-long-projection-separation-rate nil)
(defvar photon-large-readout-start-long-projection-separation-epochs nil)
(defvar photon-large-readout-start-long-projection-separation-top-k nil)
(defvar photon-large-readout-start-long-projection-separation-margin nil)
(defvar photon-large-readout-start-long-projection-separation-score-bonus-rate
  nil)
(defvar photon-large-readout-start-long-projection-separation-rival-bias-rate
  nil)
(defvar photon-large-readout-start-long-projection-separation-pocket-enabled
  nil)
(defvar photon-large-readout-start-long-projection-separation-global-tolerance
  nil)
(defvar photon-large-readout-start-long-rival-negative-distill-weight nil)
(defvar photon-large-readout-start-long-rival-negative-distill-epochs nil)
(defvar photon-large-readout-start-long-rival-negative-distill-top-k nil)
(defvar photon-large-readout-start-long-rival-negative-distill-rivals nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-weight nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-bias-weight nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-epochs nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-top-k nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-rivals nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-margin nil)
(defvar photon-large-readout-start-long-same-shape-margin-distill-adaptive-bias-max nil)
(defvar photon-large-readout-start-long-same-shape-top1-search-enabled nil)
(defvar photon-large-readout-start-long-same-shape-top1-search-score-bonus-max nil)
(defvar photon-large-readout-start-long-same-shape-top1-search-context-bonus-weight nil)
(defvar photon-large-readout-start-long-same-shape-top1-search-global-tolerance nil)
(defvar photon-large-readout-wordpiece-long-teacher-state-path-weight nil)
(defvar photon-large-readout-wordpiece-continuation-teacher-state-path-weight nil)
(defvar photon-large-readout-wordpiece-start-long-teacher-state-path-weight nil)
(defvar photon-large-readout-wordpiece-start-long-teacher-token-mix nil)
(defvar photon-large-readout-wordpiece-start-long-teacher-context-mix nil)
(defvar photon-large-readout-wordpiece-start-long-teacher-anchor-mix nil)
(defvar photon-large-readout-wordpiece-start-long-boundary-readout-enabled nil)
(defvar photon-large-readout-wordpiece-start-long-boundary-readout-weight nil)
(defvar photon-large-readout-wordpiece-start-long-boundary-readout-size nil)
(defvar photon-large-readout-wordpiece-start-long-boundary-ngram-mix nil)
(defvar photon-large-readout-full-distill-start-long-boundary-teacher-weight nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-enabled nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-readout-weight nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates
  nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-candidate-top-k nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-candidate-margin nil)
(defvar photon-large-readout-wordpiece-start-long-classifier-candidate-limit nil)
(defvar photon-large-readout-full-distill-start-long-classifier-teacher-weight nil)
(defvar photon-large-readout-full-distill-start-long-ngram-teacher-weight nil)
(defvar photon-large-readout-full-distill-start-long-scale nil)
(defvar photon-large-readout-full-distill-continuation-long-scale nil)
(defvar photon-large-readout-full-distill-start-short-scale nil)
(defvar photon-large-readout-full-distill-continuation-short-scale nil)
(defvar photon-large-readout-full-distill-selection-start-long-weight nil)
(defvar photon-large-readout-full-distill-selection-after-weight nil)
(defvar photon-large-readout-distill-candidates
  '(full-scaled full-unscaled structural-off-unscaled
    char-structural-bias-off-unscaled))

(load "lisp/photon.el")

(photon-use-elisp-vector-backend)

(defun photon-large-generation-distinct (generated size)
  (let ((index 0)
        (total 0)
        (seen nil))
    (while (<= (+ index size) (length generated))
      (let ((piece (substring generated index (+ index size))))
        (setq total (1+ total))
        (unless (photon-member-equal-p piece seen)
          (setq seen (cons piece seen))))
      (setq index (1+ index)))
    (list (cons 'total total)
          (cons 'distinct (length seen))
          (cons 'distinct-permil
                (if (= total 0) 0 (/ (* 1000 (length seen)) total))))))

(defun photon-large-generation-repeat-metrics (generated prompt)
  (let ((index (max 1 (length prompt)))
        (repeated 0)
        (total 0)
        (current-run 1)
        (max-run 1)
        (newline-run 0)
        (space-run 0)
        (max-newline-run 0)
        (max-space-run 0))
    (while (< index (length generated))
      (let ((char (aref generated index))
            (previous (aref generated (1- index))))
        (setq total (1+ total))
        (when (= char previous)
          (setq repeated (1+ repeated)))
        (if (= char previous)
            (setq current-run (1+ current-run))
          (setq current-run 1))
        (when (> current-run max-run)
          (setq max-run current-run))
        (cond ((= char ?\n)
               (setq newline-run (1+ newline-run))
               (setq space-run 0))
              ((= char ?\s)
               (setq space-run (1+ space-run))
               (setq newline-run 0))
              (t
               (setq newline-run 0)
               (setq space-run 0)))
        (when (> newline-run max-newline-run)
          (setq max-newline-run newline-run))
        (when (> space-run max-space-run)
          (setq max-space-run space-run)))
      (setq index (1+ index)))
    (list (cons 'generated-tail-length total)
          (cons 'repeated-token-permil
                (if (= total 0) 0 (/ (* 1000 repeated) total)))
          (cons 'max-run max-run)
          (cons 'max-newline-run max-newline-run)
          (cons 'max-space-run max-space-run)
          (cons 'collapse-score
                (+ (if (= total 0) 0 (/ (* 1000 repeated) total))
                   (* 100 max-run)
                   (* 100 max-newline-run)
                   (* 100 max-space-run))))))

(defun photon-large-generation-average-field (records path)
  (let ((total 0.0)
        (count 0))
    (while records
      (let ((value (car records))
            (keys path))
        (while keys
          (setq value (cdr (assq (car keys) value)))
          (setq keys (cdr keys)))
        (when (numberp value)
          (setq total (+ total value))
          (setq count (1+ count))))
      (setq records (cdr records)))
    (if (= count 0) 0.0 (/ total count))))

(defun photon-large-generation-run-one (model prompt steps options)
  (let ((generated (photon-model-generate-text model prompt steps options)))
    (list (cons 'prompt prompt)
          (cons 'generated-length (length generated))
          (cons 'collapse-metrics
                (photon-large-generation-repeat-metrics generated prompt))
          (cons 'distinct-1
                (photon-large-generation-distinct generated 1))
          (cons 'distinct-2
                (photon-large-generation-distinct generated 2)))))

(defun photon-large-generation-options-with-seed (options seed)
  (cons (cons 'seed seed)
        (assq-delete-all 'seed (copy-sequence options))))

(defun photon-large-generation-row (name model prompts steps options seeds)
  (let ((rows nil)
        (walk seeds))
    (while walk
      (let ((seed-options
             (photon-large-generation-options-with-seed options (car walk))))
        (setq rows
              (cons
               (mapcar
                (lambda (prompt)
                  (photon-large-generation-run-one
                   model prompt steps seed-options))
                prompts)
               rows)))
      (setq walk (cdr walk)))
    (let ((flat (apply 'append (nreverse rows))))
      (list (cons 'name name)
            (cons 'options options)
            (cons 'seeds seeds)
            (cons 'average-repeated-token-permil
                  (photon-large-generation-average-field
                   flat '(collapse-metrics repeated-token-permil)))
            (cons 'average-collapse-score
                  (photon-large-generation-average-field
                   flat '(collapse-metrics collapse-score)))
            (cons 'average-distinct-1-permil
                  (photon-large-generation-average-field
                   flat '(distinct-1 distinct-permil)))
            (cons 'average-distinct-2-permil
                  (photon-large-generation-average-field
                   flat '(distinct-2 distinct-permil)))
            (cons 'quality-score
                  (- (photon-large-generation-average-field
                      flat '(distinct-2 distinct-permil))
                     (photon-large-generation-average-field
                      flat '(collapse-metrics collapse-score))))))))

(defun photon-large-generation-eval-language-tokens (model tokens context-size)
  (let ((window nil)
        (filled 0)
        (token nil)
        (total 0)
        (correct 0))
    (setq token (car tokens))
    (setq tokens (cdr tokens))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (car tokens))
      (setq tokens (cdr tokens)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (logits (photon-model-readout-logits model state window))
               (prediction (photon-argmax-index logits)))
          (when (= prediction token)
            (setq correct (1+ correct)))
          (setq total (1+ total))
          (setq window (photon-window-slide window token))
          (setq token (car tokens))
          (setq tokens (cdr tokens)))))
    (list (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total))))))

(defun photon-large-generation-distill-candidate
    (name model eval-tokens context-size learning-rate teacher-key scale-enabled)
  (let ((candidate (copy-tree model))
        (after-weight
         (or photon-large-readout-full-distill-selection-after-weight 0.0))
        distill
        after
        selection-score)
    (photon-model-param-set
     candidate 'output-head-full-readout-distill-wordpiece-scale-enabled
     (if scale-enabled 1.0 0.0))
    (setq distill
          (photon-distill-full-readout-to-output-head-token-reader-pocket
           candidate
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-full-distill-token-limit)))
           context-size
           (* learning-rate
              (photon-model-param-get
               candidate 'output-head-full-readout-distill-weight))
           photon-large-readout-full-distill-epochs
           teacher-key))
    (setq after (photon-evaluate candidate eval-tokens context-size))
    (setq selection-score
          (+ (cdr (assq 'best-selection-score distill))
             (* after-weight
                (cdr (assq 'accuracy-permil after)))))
    (list (cons 'name name)
          (cons 'teacher-key teacher-key)
          (cons 'wordpiece-scale-enabled (if scale-enabled 1.0 0.0))
          (cons 'model candidate)
          (cons 'distill distill)
          (cons 'after after)
          (cons 'best-linear-accuracy-permil
                (cdr (assq 'best-linear-accuracy-permil distill)))
          (cons 'best-prototype-accuracy-permil
                (cdr (assq 'best-prototype-accuracy-permil distill)))
          (cons 'best-selection-score
                (cdr (assq 'best-selection-score distill)))
          (cons 'selection-after-weight after-weight)
          (cons 'selection-score selection-score))))

(defun photon-large-generation-public-distill-candidate (candidate)
  (let ((result nil)
        (remaining candidate))
    (while remaining
      (unless (eq (caar remaining) 'model)
        (setq result (cons (car remaining) result)))
      (setq remaining (cdr remaining)))
    (nreverse result)))

(defun photon-large-generation-select-full-readout-distill
    (model eval-tokens context-size learning-rate)
  (let* ((candidate-specs
          '((full-scaled full t)
            (start-long-boundary-teacher-scaled
             start-long-boundary-teacher t)
            (start-long-classifier-teacher-scaled
             start-long-classifier-teacher t)
            (start-long-ngram-teacher-scaled
             start-long-ngram-teacher t)
            (full-unscaled full nil)
            (structural-off-unscaled structural-off nil)
            (char-structural-bias-off-unscaled
             char-structural-bias-off nil)))
         (candidates nil)
         spec
         selected-specs
         (best (car candidates))
         (remaining (cdr candidates)))
    (while candidate-specs
      (setq spec (car candidate-specs))
      (when (memq (car spec) photon-large-readout-distill-candidates)
        (setq selected-specs (cons spec selected-specs)))
      (setq candidate-specs (cdr candidate-specs)))
    (setq selected-specs (nreverse selected-specs))
    (unless selected-specs
      (error "No large readout distill candidates selected"))
    (while selected-specs
      (setq spec (car selected-specs))
      (setq candidates
            (cons
             (photon-large-generation-distill-candidate
              (nth 0 spec) model eval-tokens context-size learning-rate
              (nth 1 spec) (nth 2 spec))
             candidates))
      (setq selected-specs (cdr selected-specs)))
    (setq candidates (nreverse candidates))
    (setq best (car candidates))
    (setq remaining (cdr candidates))
    (while remaining
      (when (> (cdr (assq 'selection-score (car remaining)))
               (cdr (assq 'selection-score best)))
        (setq best (car remaining)))
      (setq remaining (cdr remaining)))
    (list (cons 'selected best)
          (cons 'candidates candidates)
          (cons 'public
                (list
                 (cons 'selected
                       (photon-large-generation-public-distill-candidate best))
                 (cons 'candidates
                       (mapcar 'photon-large-generation-public-distill-candidate
                               candidates)))))))

(defun photon-large-generation-train-light
    (train-path eval-path context-size hidden-size chunk-size epochs
                learning-rate max-piece-size)
  (let* ((train-text (photon-read-text-file train-path))
         (eval-text (photon-read-text-file eval-path))
         (vocab (photon-build-wordpiece-vocab
                 (concat train-text "\n" eval-text)
                 max-piece-size))
         (train-tokens (photon-encode-wordpiece-text
                        vocab train-text max-piece-size))
         (eval-tokens (photon-encode-wordpiece-text
                       vocab eval-text max-piece-size))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         before
         history
         long-rare-readout-rescue
         output-head-prototype-readout-distill
         ngram-long-output-head-prototype-distill
         start-long-positive-distill
         start-long-projected-prototype-distill
         start-long-prototype-merge
         start-long-projection-separation
         start-long-rival-negative-distill
         start-long-same-shape-margin-distill
         start-long-same-shape-top1-search
         start-long-classifier-candidate-search
         post-ngram-long-output-head-prototype-refresh
         long-piece-output-head-prototype-refresh
         full-readout-distill
         full-readout-distill-selection
         start-long-classifier-readout-selection
         after)
    (when photon-large-readout-output-head-prototype-support-weight
      (photon-model-param-set
       model 'output-head-prototype-support-weight
       photon-large-readout-output-head-prototype-support-weight))
    (when photon-large-readout-output-head-prototype-limit
      (photon-model-param-set
       model 'output-head-prototype-limit
       photon-large-readout-output-head-prototype-limit))
    (when photon-large-readout-output-head-prototype-long-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-long-piece-weight
       photon-large-readout-output-head-prototype-long-piece-weight))
    (when photon-large-readout-output-head-prototype-short-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-short-piece-weight
       photon-large-readout-output-head-prototype-short-piece-weight))
    (when photon-large-readout-output-head-prototype-start-long-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-start-long-piece-weight
       photon-large-readout-output-head-prototype-start-long-piece-weight))
    (when photon-large-readout-output-head-prototype-continuation-long-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-continuation-long-piece-weight
       photon-large-readout-output-head-prototype-continuation-long-piece-weight))
    (when photon-large-readout-output-head-prototype-start-short-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-start-short-piece-weight
       photon-large-readout-output-head-prototype-start-short-piece-weight))
    (when photon-large-readout-output-head-prototype-continuation-short-piece-weight
      (photon-model-param-set
       model 'output-head-prototype-continuation-short-piece-weight
       photon-large-readout-output-head-prototype-continuation-short-piece-weight))
    (when photon-large-readout-output-head-prototype-negative-weight
      (photon-model-param-set
       model 'output-head-prototype-negative-weight
       photon-large-readout-output-head-prototype-negative-weight))
    (when photon-large-readout-output-head-prototype-negative-limit
      (photon-model-param-set
       model 'output-head-prototype-negative-limit
       photon-large-readout-output-head-prototype-negative-limit))
    (when photon-large-readout-output-head-prototype-ngram-long-token-mix
      (photon-model-param-set
       model 'output-head-prototype-ngram-long-token-mix
       photon-large-readout-output-head-prototype-ngram-long-token-mix))
    (when photon-large-readout-output-head-prototype-ngram-long-context-mix
      (photon-model-param-set
       model 'output-head-prototype-ngram-long-context-mix
       photon-large-readout-output-head-prototype-ngram-long-context-mix))
    (when photon-large-readout-output-head-prototype-ngram-long-anchor-mix
      (photon-model-param-set
       model 'output-head-prototype-ngram-long-anchor-mix
       photon-large-readout-output-head-prototype-ngram-long-anchor-mix))
    (when photon-large-readout-output-head-prototype-ngram-long-contrastive-rate
      (photon-model-param-set
       model 'output-head-prototype-ngram-long-contrastive-rate
       photon-large-readout-output-head-prototype-ngram-long-contrastive-rate))
    (when photon-large-readout-output-head-prototype-ngram-long-state-path-rate
      (photon-model-param-set
       model 'output-head-prototype-ngram-long-state-path-rate
       photon-large-readout-output-head-prototype-ngram-long-state-path-rate))
    (when photon-large-readout-start-long-positive-distill-weight
      (photon-model-param-set
       model 'output-head-start-long-positive-distill-weight
       photon-large-readout-start-long-positive-distill-weight))
    (when photon-large-readout-start-long-positive-distill-epochs
      (photon-model-param-set
       model 'output-head-start-long-positive-distill-epochs
       photon-large-readout-start-long-positive-distill-epochs))
    (when photon-large-readout-start-long-positive-distill-top-k
      (photon-model-param-set
       model 'output-head-start-long-positive-distill-top-k
       photon-large-readout-start-long-positive-distill-top-k))
    (when photon-large-readout-start-long-projected-prototype-distill-weight
      (photon-model-param-set
       model 'output-head-start-long-projected-prototype-distill-weight
       photon-large-readout-start-long-projected-prototype-distill-weight))
    (when photon-large-readout-start-long-projected-prototype-distill-epochs
      (photon-model-param-set
       model 'output-head-start-long-projected-prototype-distill-epochs
       photon-large-readout-start-long-projected-prototype-distill-epochs))
    (when photon-large-readout-start-long-projected-prototype-distill-top-k
      (photon-model-param-set
       model 'output-head-start-long-projected-prototype-distill-top-k
       photon-large-readout-start-long-projected-prototype-distill-top-k))
    (when photon-large-readout-start-long-prototype-merge-rate
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-rate
       photon-large-readout-start-long-prototype-merge-rate))
    (when photon-large-readout-start-long-prototype-merge-epochs
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-epochs
       photon-large-readout-start-long-prototype-merge-epochs))
    (when photon-large-readout-start-long-prototype-merge-top-k
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-top-k
       photon-large-readout-start-long-prototype-merge-top-k))
    (when photon-large-readout-start-long-prototype-merge-margin
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-margin
       photon-large-readout-start-long-prototype-merge-margin))
    (when photon-large-readout-start-long-prototype-merge-target-bias-rate
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-target-bias-rate
       photon-large-readout-start-long-prototype-merge-target-bias-rate))
    (when photon-large-readout-start-long-prototype-merge-rival-bias-rate
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-rival-bias-rate
       photon-large-readout-start-long-prototype-merge-rival-bias-rate))
    (when photon-large-readout-start-long-prototype-merge-score-bonus-rate
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-score-bonus-rate
       photon-large-readout-start-long-prototype-merge-score-bonus-rate))
    (when photon-large-readout-start-long-prototype-merge-dominant-rival-bias-rate
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-dominant-rival-bias-rate
       photon-large-readout-start-long-prototype-merge-dominant-rival-bias-rate))
    (when photon-large-readout-start-long-prototype-merge-dominant-rival-min-count
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-dominant-rival-min-count
       photon-large-readout-start-long-prototype-merge-dominant-rival-min-count))
    (when photon-large-readout-start-long-prototype-merge-state-separation-rate
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-state-separation-rate
       photon-large-readout-start-long-prototype-merge-state-separation-rate))
    (when photon-large-readout-start-long-prototype-merge-dense-head-rate
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-dense-head-rate
       photon-large-readout-start-long-prototype-merge-dense-head-rate))
    (when photon-large-readout-start-long-prototype-merge-rival-line-search-step
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-rival-line-search-step
       photon-large-readout-start-long-prototype-merge-rival-line-search-step))
    (when photon-large-readout-start-long-prototype-merge-rival-line-search-max
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-rival-line-search-max
       photon-large-readout-start-long-prototype-merge-rival-line-search-max))
    (when photon-large-readout-start-long-prototype-merge-rival-negative-weight
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-rival-negative-weight
       photon-large-readout-start-long-prototype-merge-rival-negative-weight))
    (when photon-large-readout-start-long-prototype-merge-rivals
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-rivals
       photon-large-readout-start-long-prototype-merge-rivals))
    (when photon-large-readout-start-long-prototype-merge-pocket-enabled
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-pocket-enabled
       photon-large-readout-start-long-prototype-merge-pocket-enabled))
    (when photon-large-readout-start-long-prototype-merge-candidate-search-enabled
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-candidate-search-enabled
       photon-large-readout-start-long-prototype-merge-candidate-search-enabled))
    (when photon-large-readout-start-long-prototype-merge-candidate-limit
      (photon-model-param-set
       model
       'output-head-start-long-prototype-merge-candidate-limit
       photon-large-readout-start-long-prototype-merge-candidate-limit))
    (when photon-large-readout-start-long-prototype-merge-global-tolerance
      (photon-model-param-set
       model 'output-head-start-long-prototype-merge-global-tolerance
       photon-large-readout-start-long-prototype-merge-global-tolerance))
    (when photon-large-readout-start-long-projection-separation-rate
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-rate
       photon-large-readout-start-long-projection-separation-rate))
    (when photon-large-readout-start-long-projection-separation-epochs
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-epochs
       photon-large-readout-start-long-projection-separation-epochs))
    (when photon-large-readout-start-long-projection-separation-top-k
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-top-k
       photon-large-readout-start-long-projection-separation-top-k))
    (when photon-large-readout-start-long-projection-separation-margin
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-margin
       photon-large-readout-start-long-projection-separation-margin))
    (when photon-large-readout-start-long-projection-separation-score-bonus-rate
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-score-bonus-rate
       photon-large-readout-start-long-projection-separation-score-bonus-rate))
    (when photon-large-readout-start-long-projection-separation-rival-bias-rate
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-rival-bias-rate
       photon-large-readout-start-long-projection-separation-rival-bias-rate))
    (when photon-large-readout-start-long-projection-separation-pocket-enabled
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-pocket-enabled
       photon-large-readout-start-long-projection-separation-pocket-enabled))
    (when photon-large-readout-start-long-projection-separation-global-tolerance
      (photon-model-param-set
       model 'output-head-start-long-projection-separation-global-tolerance
       photon-large-readout-start-long-projection-separation-global-tolerance))
    (when photon-large-readout-start-long-rival-negative-distill-weight
      (photon-model-param-set
       model 'output-head-start-long-rival-negative-distill-weight
       photon-large-readout-start-long-rival-negative-distill-weight))
    (when photon-large-readout-start-long-rival-negative-distill-epochs
      (photon-model-param-set
       model 'output-head-start-long-rival-negative-distill-epochs
       photon-large-readout-start-long-rival-negative-distill-epochs))
    (when photon-large-readout-start-long-rival-negative-distill-top-k
      (photon-model-param-set
       model 'output-head-start-long-rival-negative-distill-top-k
       photon-large-readout-start-long-rival-negative-distill-top-k))
    (when photon-large-readout-start-long-rival-negative-distill-rivals
      (photon-model-param-set
       model 'output-head-start-long-rival-negative-distill-rivals
       photon-large-readout-start-long-rival-negative-distill-rivals))
    (when photon-large-readout-start-long-same-shape-margin-distill-weight
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-weight
       photon-large-readout-start-long-same-shape-margin-distill-weight))
    (when photon-large-readout-start-long-same-shape-margin-distill-bias-weight
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-bias-weight
       photon-large-readout-start-long-same-shape-margin-distill-bias-weight))
    (when photon-large-readout-start-long-same-shape-margin-distill-epochs
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-epochs
       photon-large-readout-start-long-same-shape-margin-distill-epochs))
    (when photon-large-readout-start-long-same-shape-margin-distill-top-k
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-top-k
       photon-large-readout-start-long-same-shape-margin-distill-top-k))
    (when photon-large-readout-start-long-same-shape-margin-distill-rivals
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-rivals
       photon-large-readout-start-long-same-shape-margin-distill-rivals))
    (when photon-large-readout-start-long-same-shape-margin-distill-margin
      (photon-model-param-set
       model 'output-head-start-long-same-shape-margin-distill-margin
       photon-large-readout-start-long-same-shape-margin-distill-margin))
    (when photon-large-readout-start-long-same-shape-margin-distill-adaptive-bias-max
      (photon-model-param-set
       model
       'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
       photon-large-readout-start-long-same-shape-margin-distill-adaptive-bias-max))
    (when photon-large-readout-start-long-same-shape-top1-search-enabled
      (photon-model-param-set
       model 'output-head-start-long-same-shape-top1-search-enabled
       photon-large-readout-start-long-same-shape-top1-search-enabled))
    (when photon-large-readout-start-long-same-shape-top1-search-score-bonus-max
      (photon-model-param-set
       model
       'output-head-start-long-same-shape-top1-search-score-bonus-max
       photon-large-readout-start-long-same-shape-top1-search-score-bonus-max))
    (when photon-large-readout-start-long-same-shape-top1-search-context-bonus-weight
      (photon-model-param-set
       model
       'output-head-start-long-same-shape-top1-search-context-bonus-weight
       photon-large-readout-start-long-same-shape-top1-search-context-bonus-weight))
    (when photon-large-readout-start-long-same-shape-top1-search-global-tolerance
      (photon-model-param-set
       model
       'output-head-start-long-same-shape-top1-search-global-tolerance
       photon-large-readout-start-long-same-shape-top1-search-global-tolerance))
    (when photon-large-readout-wordpiece-long-teacher-state-path-weight
      (photon-model-param-set
       model 'wordpiece-long-teacher-state-path-weight
       photon-large-readout-wordpiece-long-teacher-state-path-weight))
    (when photon-large-readout-wordpiece-continuation-teacher-state-path-weight
      (photon-model-param-set
       model 'wordpiece-continuation-teacher-state-path-weight
       photon-large-readout-wordpiece-continuation-teacher-state-path-weight))
    (when photon-large-readout-wordpiece-start-long-teacher-state-path-weight
      (photon-model-param-set
       model 'wordpiece-start-long-teacher-state-path-weight
       photon-large-readout-wordpiece-start-long-teacher-state-path-weight))
    (when photon-large-readout-wordpiece-start-long-teacher-token-mix
      (photon-model-param-set
       model 'wordpiece-start-long-teacher-token-mix
       photon-large-readout-wordpiece-start-long-teacher-token-mix))
    (when photon-large-readout-wordpiece-start-long-teacher-context-mix
      (photon-model-param-set
       model 'wordpiece-start-long-teacher-context-mix
       photon-large-readout-wordpiece-start-long-teacher-context-mix))
    (when photon-large-readout-wordpiece-start-long-teacher-anchor-mix
      (photon-model-param-set
       model 'wordpiece-start-long-teacher-anchor-mix
       photon-large-readout-wordpiece-start-long-teacher-anchor-mix))
    (when photon-large-readout-wordpiece-start-long-boundary-readout-enabled
      (photon-model-param-set
       model 'wordpiece-start-long-boundary-readout-enabled
       photon-large-readout-wordpiece-start-long-boundary-readout-enabled))
    (when photon-large-readout-wordpiece-start-long-boundary-readout-weight
      (photon-model-param-set
       model 'wordpiece-start-long-boundary-readout-weight
       photon-large-readout-wordpiece-start-long-boundary-readout-weight))
    (when photon-large-readout-wordpiece-start-long-boundary-readout-size
      (photon-model-param-set
       model 'wordpiece-start-long-boundary-readout-size
       photon-large-readout-wordpiece-start-long-boundary-readout-size))
    (when photon-large-readout-wordpiece-start-long-boundary-ngram-mix
      (photon-model-param-set
       model 'wordpiece-start-long-boundary-ngram-mix
       photon-large-readout-wordpiece-start-long-boundary-ngram-mix))
    (when photon-large-readout-full-distill-start-long-boundary-teacher-weight
      (photon-model-param-set
       model
       'output-head-full-readout-distill-start-long-boundary-teacher-weight
       photon-large-readout-full-distill-start-long-boundary-teacher-weight))
    (when photon-large-readout-wordpiece-start-long-classifier-enabled
      (photon-model-param-set
       model 'wordpiece-start-long-classifier-enabled
       photon-large-readout-wordpiece-start-long-classifier-enabled))
    (when photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled
      (photon-model-param-set
       model 'wordpiece-start-long-classifier-enabled 1.0)
      (photon-model-param-set
       model
       'wordpiece-start-long-classifier-candidate-search-enabled
       photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled))
    (when photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale
      (photon-model-param-set
       model 'wordpiece-start-long-classifier-learning-rate-scale
       photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale))
    (when photon-large-readout-wordpiece-start-long-classifier-readout-weight
      (photon-model-param-set
       model 'wordpiece-start-long-classifier-readout-weight
       photon-large-readout-wordpiece-start-long-classifier-readout-weight))
    (when photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate
      (photon-model-param-set
       model
       'wordpiece-start-long-classifier-candidate-learning-rate
       photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate))
    (when photon-large-readout-wordpiece-start-long-classifier-candidate-top-k
      (photon-model-param-set
       model
       'wordpiece-start-long-classifier-candidate-top-k
       photon-large-readout-wordpiece-start-long-classifier-candidate-top-k))
    (when photon-large-readout-wordpiece-start-long-classifier-candidate-margin
      (photon-model-param-set
       model
       'wordpiece-start-long-classifier-candidate-margin
       photon-large-readout-wordpiece-start-long-classifier-candidate-margin))
    (when photon-large-readout-wordpiece-start-long-classifier-candidate-limit
      (photon-model-param-set
       model
       'wordpiece-start-long-classifier-candidate-limit
       photon-large-readout-wordpiece-start-long-classifier-candidate-limit))
    (when photon-large-readout-full-distill-start-long-classifier-teacher-weight
      (photon-model-param-set
       model
       'output-head-full-readout-distill-start-long-classifier-teacher-weight
       photon-large-readout-full-distill-start-long-classifier-teacher-weight))
    (when photon-large-readout-full-distill-start-long-ngram-teacher-weight
      (photon-model-param-set
       model
       'output-head-full-readout-distill-start-long-ngram-teacher-weight
       photon-large-readout-full-distill-start-long-ngram-teacher-weight))
    (when photon-large-readout-full-distill-start-long-scale
      (photon-model-param-set
       model 'output-head-full-readout-distill-start-long-scale
       photon-large-readout-full-distill-start-long-scale))
    (when photon-large-readout-full-distill-continuation-long-scale
      (photon-model-param-set
       model 'output-head-full-readout-distill-continuation-long-scale
       photon-large-readout-full-distill-continuation-long-scale))
    (when photon-large-readout-full-distill-start-short-scale
      (photon-model-param-set
       model 'output-head-full-readout-distill-start-short-scale
       photon-large-readout-full-distill-start-short-scale))
    (when photon-large-readout-full-distill-continuation-short-scale
      (photon-model-param-set
       model 'output-head-full-readout-distill-continuation-short-scale
       photon-large-readout-full-distill-continuation-short-scale))
    (when photon-large-readout-full-distill-selection-start-long-weight
      (photon-model-param-set
       model 'output-head-full-readout-distill-selection-start-long-weight
       photon-large-readout-full-distill-selection-start-long-weight))
    (when photon-large-readout-train-token-limit
      (setq train-tokens
            (photon-take train-tokens photon-large-readout-train-token-limit)))
    (when photon-large-readout-eval-token-limit
      (setq eval-tokens
            (photon-take eval-tokens photon-large-readout-eval-token-limit)))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (photon-model-metadata-put model 'wordpiece-max-piece-size max-piece-size)
    (photon-model-apply-wordpiece-readout-defaults model)
    (setq before (photon-evaluate model eval-tokens context-size))
    (setq history (photon-train
                   model train-tokens context-size epochs learning-rate))
    (photon-model-apply-wordpiece-final-readout-defaults model)
    (photon-model-param-set model 'wordpiece-readout-bias-enabled 1.0)
    (setq long-rare-readout-rescue
          (photon-wordpiece-long-rare-readout-rescue-token-reader-epochs
           model
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-diagnostic-token-limit)))
           context-size
           (* learning-rate
              (photon-model-param-get
               model 'wordpiece-long-rare-readout-rescue-weight))
           photon-large-readout-long-rare-epochs))
    (setq output-head-prototype-readout-distill
          (photon-distill-readout-to-output-head-prototypes-token-reader-epochs
           model
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-diagnostic-token-limit)))
           context-size
           photon-large-readout-prototype-distill-epochs
           'full))
    (setq ngram-long-output-head-prototype-distill
          (photon-distill-ngram-long-to-output-head-prototypes-token-reader-epochs
           model
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-diagnostic-token-limit)))
           context-size
           photon-large-readout-ngram-long-prototype-distill-epochs))
    (when (> (photon-model-param-get
              model 'output-head-start-long-positive-distill-weight)
             0.0)
      (setq start-long-positive-distill
            (photon-start-long-positive-distill-token-reader-epochs
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-start-long-positive-distill-weight))
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-positive-distill-epochs)))))
    (setq post-ngram-long-output-head-prototype-refresh
          (photon-distill-readout-to-output-head-prototypes-token-reader-epochs
           model
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-diagnostic-token-limit)))
           context-size
           photon-large-readout-post-ngram-long-prototype-refresh-epochs
           'full))
    (setq long-piece-output-head-prototype-refresh
          (photon-distill-readout-to-output-head-prototypes-token-reader-epochs
           model
           (lambda ()
             (photon-make-list-token-reader
              (photon-take eval-tokens
                           photon-large-readout-diagnostic-token-limit)))
           context-size
           photon-large-readout-long-piece-prototype-refresh-epochs
           'full
           'long-piece))
    (when (> (photon-model-param-get
              model
              'wordpiece-start-long-classifier-candidate-search-enabled)
             0.0)
      (setq start-long-classifier-candidate-search
            (photon-start-long-classifier-candidate-search
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size)))
    (let* ((selection
            (photon-large-generation-select-full-readout-distill
             model eval-tokens context-size learning-rate))
           (selected (cdr (assq 'selected selection))))
      (setq model (cdr (assq 'model selected)))
      (setq full-readout-distill (cdr (assq 'distill selected)))
      (setq full-readout-distill-selection (cdr (assq 'public selection)))
      (setq after (cdr (assq 'after selected))))
    (when (> (photon-model-param-get
              model
              'output-head-start-long-projected-prototype-distill-weight)
             0.0)
      (setq start-long-projected-prototype-distill
            (photon-start-long-projected-prototype-distill-token-reader-epochs
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-projected-prototype-distill-epochs))))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (when (> (photon-model-param-get
              model 'output-head-start-long-prototype-merge-rate)
             0.0)
      (setq start-long-prototype-merge
            (cond
             ((and (> (photon-model-param-get
                       model
                       'output-head-start-long-prototype-merge-pocket-enabled)
                      0.0)
                   (> (photon-model-param-get
                       model
                       'output-head-start-long-prototype-merge-candidate-search-enabled)
                      0.0))
              (let ((pocket
                     (photon-start-long-prototype-merge-pocket
                      model
                      (lambda ()
                        (photon-make-list-token-reader
                         (photon-take
                          eval-tokens
                          photon-large-readout-diagnostic-token-limit)))
                      context-size
                      (truncate
                       (photon-model-param-get
                        model 'output-head-start-long-prototype-merge-epochs)))))
                (list
                 (cons 'pocket pocket)
                 (cons
                  'candidate-search
                  (photon-start-long-prototype-merge-candidate-search
                   model
                   (lambda ()
                     (photon-make-list-token-reader
                      (photon-take
                       eval-tokens
                       photon-large-readout-diagnostic-token-limit)))
                   context-size)))))
             ((> (photon-model-param-get
                  model
                  'output-head-start-long-prototype-merge-candidate-search-enabled)
                 0.0)
              (photon-start-long-prototype-merge-candidate-search
               model
               (lambda ()
                 (photon-make-list-token-reader
                  (photon-take eval-tokens
                               photon-large-readout-diagnostic-token-limit)))
               context-size))
             ((> (photon-model-param-get
                  model
                  'output-head-start-long-prototype-merge-pocket-enabled)
                 0.0)
              (photon-start-long-prototype-merge-pocket
               model
               (lambda ()
                 (photon-make-list-token-reader
                  (photon-take eval-tokens
                               photon-large-readout-diagnostic-token-limit)))
               context-size
               (truncate
                (photon-model-param-get
                 model 'output-head-start-long-prototype-merge-epochs))))
             (t
              (photon-start-long-prototype-merge-token-reader-epochs
               model
               (lambda ()
                 (photon-make-list-token-reader
                  (photon-take eval-tokens
                               photon-large-readout-diagnostic-token-limit)))
               context-size
               (truncate
                (photon-model-param-get
                 model 'output-head-start-long-prototype-merge-epochs))))))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (when (> (photon-model-param-get
              model 'output-head-start-long-projection-separation-rate)
             0.0)
      (setq start-long-projection-separation
            (if (> (photon-model-param-get
                    model
                    'output-head-start-long-projection-separation-pocket-enabled)
                   0.0)
                (photon-start-long-projection-separation-pocket
                 model
                 (lambda ()
                   (photon-make-list-token-reader
                    (photon-take eval-tokens
                                 photon-large-readout-diagnostic-token-limit)))
                 context-size
                 (truncate
                  (photon-model-param-get
                   model
                   'output-head-start-long-projection-separation-epochs)))
              (photon-start-long-projection-separation-token-reader-epochs
               model
               (lambda ()
                 (photon-make-list-token-reader
                  (photon-take eval-tokens
                               photon-large-readout-diagnostic-token-limit)))
               context-size
               (truncate
                (photon-model-param-get
                 model
                 'output-head-start-long-projection-separation-epochs)))))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (when (> (photon-model-param-get
              model
              'output-head-start-long-rival-negative-distill-weight)
             0.0)
      (setq start-long-rival-negative-distill
            (photon-start-long-rival-negative-distill-token-reader-epochs
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-rival-negative-distill-epochs))))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (when (or (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-weight)
                 0.0)
              (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-bias-weight)
                 0.0)
              (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-adaptive-bias-max)
                 0.0))
      (setq start-long-same-shape-margin-distill
            (photon-start-long-same-shape-margin-distill-pocket
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model
                 'output-head-start-long-same-shape-margin-distill-weight))
             (truncate
             (photon-model-param-get
               model
               'output-head-start-long-same-shape-margin-distill-epochs))))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (when (> (photon-model-param-get
              model 'output-head-start-long-same-shape-top1-search-enabled)
             0.0)
      (setq start-long-same-shape-top1-search
            (photon-start-long-same-shape-top1-bias-search
             model
             (lambda ()
               (photon-make-list-token-reader
                (photon-take eval-tokens
                             photon-large-readout-diagnostic-token-limit)))
             context-size))
      (setq after (photon-evaluate model eval-tokens context-size)))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'train-tokens train-tokens)
          (cons 'eval-tokens eval-tokens)
          (cons 'before before)
          (cons 'history history)
          (cons 'wordpiece-long-rare-readout-rescue
                long-rare-readout-rescue)
          (cons 'output-head-prototype-readout-distill
                output-head-prototype-readout-distill)
          (cons 'ngram-long-output-head-prototype-distill
                ngram-long-output-head-prototype-distill)
          (cons 'start-long-positive-distill
                start-long-positive-distill)
          (cons 'start-long-projected-prototype-distill
                start-long-projected-prototype-distill)
          (cons 'start-long-prototype-merge
                start-long-prototype-merge)
          (cons 'start-long-projection-separation
                start-long-projection-separation)
          (cons 'start-long-rival-negative-distill
                start-long-rival-negative-distill)
          (cons 'start-long-same-shape-margin-distill
                start-long-same-shape-margin-distill)
          (cons 'start-long-same-shape-top1-search
                start-long-same-shape-top1-search)
          (cons 'start-long-classifier-candidate-search
                start-long-classifier-candidate-search)
          (cons 'post-ngram-long-output-head-prototype-refresh
                post-ngram-long-output-head-prototype-refresh)
          (cons 'long-piece-output-head-prototype-refresh
                long-piece-output-head-prototype-refresh)
          (cons 'full-readout-distill full-readout-distill)
          (cons 'full-readout-distill-selection
                full-readout-distill-selection)
          (cons 'start-long-classifier-readout-selection
                start-long-classifier-readout-selection)
          (cons 'after after))))

(defun photon-large-generation-compact-miss (record)
  (list (cons 'index (cdr (assq 'index record)))
        (cons 'context-pieces (cdr (assq 'context-pieces record)))
        (cons 'target-piece (cdr (assq 'target-piece record)))
        (cons 'target-kind (cdr (assq 'target-kind record)))
        (cons 'fallback-piece (cdr (assq 'fallback-piece record)))
        (cons 'memory-hit (cdr (assq 'memory-hit record)))
        (cons 'margin (cdr (assq 'margin record)))
        (cons 'top-5
              (mapcar
               (lambda (item)
                 (list (cons 'piece (cdr (assq 'piece item)))
                       (cons 'token-kind (cdr (assq 'token-kind item)))
                       (cons 'logit (cdr (assq 'logit item)))))
               (cdr (assq 'top-5 record))))))

(defun photon-large-generation-paths ()
  (let ((root photon-large-generation-root))
    (list
     (cons 'train-path
           (expand-file-name "../data/stream-large-sample.txt" root))
     (cons 'eval-path
           (expand-file-name "../data/stream-large-eval-sample.txt" root)))))

(defun photon-large-readout-result ()
  (let* ((paths (photon-large-generation-paths))
         (train-path (cdr (assq 'train-path paths)))
         (eval-path (cdr (assq 'eval-path paths)))
         (result (photon-large-generation-train-light
                  train-path eval-path
                  photon-large-readout-context-size
                  photon-large-readout-hidden-size
                  photon-large-readout-chunk-size
                  photon-large-readout-epochs
                  photon-large-readout-learning-rate
                  photon-large-readout-max-piece-size))
         (model (cdr (assq 'model result)))
         (eval-tokens (cdr (assq 'eval-tokens result)))
         (heldout (photon-large-generation-eval-language-tokens
                   model eval-tokens photon-large-readout-context-size))
         (diagnostics (photon-wordpiece-diagnostics
                       model
                       (photon-take
                        eval-tokens
                        photon-large-readout-diagnostic-token-limit)
                       photon-large-readout-context-size 3))
         (fallback (cdr (assq 'summary diagnostics))))
    (list
     (cons 'train-path train-path)
     (cons 'eval-path eval-path)
     (cons 'vocab-size (length (cdr (assq 'vocab result))))
     (cons 'before (cdr (assq 'before result)))
     (cons 'wordpiece-long-rare-readout-rescue
           (cdr (assq 'wordpiece-long-rare-readout-rescue result)))
     (cons 'output-head-prototype-readout-distill
           (cdr (assq 'output-head-prototype-readout-distill result)))
     (cons 'ngram-long-output-head-prototype-distill
           (cdr (assq 'ngram-long-output-head-prototype-distill result)))
     (cons 'start-long-positive-distill
           (cdr (assq 'start-long-positive-distill result)))
     (cons 'start-long-projected-prototype-distill
           (cdr (assq 'start-long-projected-prototype-distill result)))
     (cons 'start-long-prototype-merge
           (cdr (assq 'start-long-prototype-merge result)))
     (cons 'start-long-projection-separation
           (cdr (assq 'start-long-projection-separation result)))
     (cons 'start-long-rival-negative-distill
           (cdr (assq 'start-long-rival-negative-distill result)))
     (cons 'start-long-same-shape-margin-distill
           (cdr (assq 'start-long-same-shape-margin-distill result)))
     (cons 'start-long-same-shape-top1-search
           (cdr (assq 'start-long-same-shape-top1-search result)))
     (cons 'start-long-classifier-candidate-search
           (cdr (assq 'start-long-classifier-candidate-search result)))
     (cons 'post-ngram-long-output-head-prototype-refresh
           (cdr (assq 'post-ngram-long-output-head-prototype-refresh result)))
     (cons 'long-piece-output-head-prototype-refresh
           (cdr (assq 'long-piece-output-head-prototype-refresh result)))
     (cons 'full-readout-distill
           (cdr (assq 'full-readout-distill result)))
     (cons 'full-readout-distill-selection
           (cdr (assq 'full-readout-distill-selection result)))
     (cons 'start-long-classifier-readout-selection
           (cdr (assq 'start-long-classifier-readout-selection result)))
     (cons 'after (cdr (assq 'after result)))
     (cons 'heldout-language heldout)
     (cons 'fallback-after fallback)
     (cons 'wordpiece-diagnostics
           (list
            (cons 'summary (cdr (assq 'summary diagnostics)))
            (cons 'readout-component-gaps
                  (cdr (assq 'readout-component-gaps diagnostics)))
            (cons 'readout-component-shape-gaps
                  (cdr (assq 'readout-component-shape-gaps diagnostics)))
            (cons 'prototype-competition-shapes
                  (cdr (assq 'prototype-competition-shapes diagnostics)))
            (cons 'fallback-miss-summary
                  (mapcar
                   'photon-large-generation-compact-miss
                   (photon-take
                    (cdr (assq 'fallback-miss-summary diagnostics))
                    5)))))
            (cons 'model model))))

(defun photon-large-readout-summary-entry (entry)
  (when entry
    (list
     (cons 'name (cdr (assq 'name entry)))
     (cons 'after (cdr (assq 'accuracy-permil
                             (cdr (assq 'after entry)))))
     (cons 'score (cdr (assq 'selection-score entry)))
     (cons 'best-linear
           (cdr (assq 'best-linear-accuracy-permil entry)))
     (cons 'best-prototype
           (cdr (assq 'best-prototype-accuracy-permil entry))))))

(defun photon-large-readout-classifier-readout-summary-entry (entry)
  (when entry
    (list
     (cons 'weight (cdr (assq 'weight entry)))
     (cons 'after (cdr (assq 'accuracy-permil
                             (cdr (assq 'after entry)))))
     (cons 'fallback (cdr (assq 'fallback-accuracy-permil entry)))
     (cons 'top-5 (cdr (assq 'top-5-accuracy-permil entry)))
     (cons 'score (cdr (assq 'score entry))))))

(defun photon-large-readout-summary-shape (entry)
  (when entry
    (list
     (cons 'class (cdr (assq 'class entry)))
     (cons 'total (cdr (assq 'total entry)))
     (cons 'fallback (cdr (assq 'fallback-accuracy-permil entry)))
     (cons 'ngram (cdr (assq 'ngram-accuracy-permil entry)))
     (cons 'raw-output-head
           (cdr (assq 'raw-output-head-accuracy-permil entry)))
     (cons 'output-head-prototype
           (cdr (assq 'output-head-prototype-accuracy-permil entry)))
     (cons 'target-top-1 (cdr (assq 'target-top-1-permil entry)))
     (cons 'target-top-3 (cdr (assq 'target-top-3-permil entry)))
     (cons 'target-top-5 (cdr (assq 'target-top-5-permil entry)))
     (cons 'average-target-rank
           (cdr (assq 'average-target-rank entry)))
     (cons 'top-rival-tokens
           (cdr (assq 'top-rival-tokens entry))))))

(defun photon-large-readout-find-class (class entries)
  (let ((found nil))
    (while (and entries (not found))
      (when (eq (cdr (assq 'class (car entries))) class)
        (setq found (car entries)))
      (setq entries (cdr entries)))
    found))

(defun photon-large-readout-compact-result (&optional result)
  (let* ((readout (or result (photon-large-readout-result)))
         (selection (cdr (assq 'full-readout-distill-selection readout)))
         (classifier-readout-selection
          (cdr (assq 'start-long-classifier-readout-selection readout)))
         (selected (cdr (assq 'selected selection)))
         (fallback (cdr (assq 'fallback-after readout)))
         (diagnostics (cdr (assq 'wordpiece-diagnostics readout)))
         (shape-fallbacks (cdr (assq 'token-shapes fallback)))
         (component-shapes
          (cdr (assq 'readout-component-shape-gaps diagnostics)))
         (prototype-shapes
          (cdr (assq 'prototype-competition-shapes diagnostics))))
    (list
     (cons 'train-path (cdr (assq 'train-path readout)))
     (cons 'eval-path (cdr (assq 'eval-path readout)))
     (cons 'vocab-size (cdr (assq 'vocab-size readout)))
     (cons 'after (cdr (assq 'accuracy-permil (cdr (assq 'after readout)))))
     (cons 'selected (photon-large-readout-summary-entry selected))
     (cons 'candidates
           (mapcar #'photon-large-readout-summary-entry
                   (cdr (assq 'candidates selection))))
     (cons 'classifier-readout-selection
           (when classifier-readout-selection
             (list
              (cons 'selected
                    (photon-large-readout-classifier-readout-summary-entry
                     (cdr (assq 'selected classifier-readout-selection))))
              (cons 'candidates
                    (mapcar
                     #'photon-large-readout-classifier-readout-summary-entry
                     (cdr (assq 'candidates
                                classifier-readout-selection)))))))
     (cons 'fallback
           (list
            (cons 'total (cdr (assq 'total fallback)))
            (cons 'accuracy-permil
                  (cdr (assq 'fallback-accuracy-permil fallback)))
            (cons 'exact-accuracy-permil
                  (cdr (assq 'exact-accuracy-permil fallback)))
            (cons 'top-5-accuracy-permil
                  (cdr (assq 'top-5-accuracy-permil fallback)))
            (cons 'memory-hit-permil
                  (cdr (assq 'memory-hit-permil fallback)))
            (cons 'rare-fallback-accuracy-permil
                  (cdr (assq 'rare-fallback-accuracy-permil fallback)))))
     (cons 'start-long-piece
           (list
            (cons 'fallback
                  (photon-large-readout-summary-shape
                   (photon-large-readout-find-class
                    'start-long-piece shape-fallbacks)))
            (cons 'components
                  (photon-large-readout-summary-shape
                   (photon-large-readout-find-class
                    'start-long-piece component-shapes)))
            (cons 'prototype-competition
                  (photon-large-readout-summary-shape
                   (photon-large-readout-find-class
                    'start-long-piece prototype-shapes))))))))

(defun photon-large-generation-result ()
  (let* ((readout (photon-large-readout-result))
         (model (cdr (assq 'model readout)))
         (train-prompts '("photon stream" "wordpiece" "generation" "context"))
         (eval-prompts '("heldout photon" "rare wordpiece" "sampling profiles"
                         "fallback readout"))
         (seeds '(17))
         (baseline '((mode . sample) (temperature . 0.85) (top-k . 5)
                     (repetition-penalty . 1.10)))
         (recommended '((mode . sample) (temperature . 0.85) (top-k . 5)
                        (repetition-penalty . 1.10)
                        (ngram-bias . 0.25) (repeat-ngram-bias . 0.35)))
         (aggressive '((mode . sample) (temperature . 0.85) (top-k . 5)
                       (repetition-penalty . 1.10)
                       (ngram-bias . 0.40) (repeat-ngram-bias . 0.35))))
    (append
     (assq-delete-all 'model (copy-sequence readout))
     (list
      (cons 'train
            (list
             (photon-large-generation-row
              'baseline model train-prompts 24 baseline seeds)
             (photon-large-generation-row
              'recommended model train-prompts 24 recommended seeds)
             (photon-large-generation-row
              'aggressive model train-prompts 24 aggressive seeds)))
      (cons 'eval
            (list
             (photon-large-generation-row
              'baseline model eval-prompts 24 baseline seeds)
             (photon-large-generation-row
              'recommended model eval-prompts 24 recommended seeds)
             (photon-large-generation-row
              'aggressive model eval-prompts 24 aggressive seeds)))))))

(when photon-large-generation-benchmark-autorun
  (prin1 (photon-large-generation-result))
  (princ "\n"))

;;; large-generation-benchmark.el ends here
