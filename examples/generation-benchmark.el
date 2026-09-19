;;; generation-benchmark.el --- Tiny text generation quality probe -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-elisp-vector-backend)

(defun photon-generation-benchmark-count-substring (needle haystack)
  (let ((start 0)
        (count 0))
    (while (and (> (length needle) 0)
                (string-match (regexp-quote needle) haystack start))
      (setq count (1+ count))
      (setq start (1+ (match-beginning 0))))
    count))

(defun photon-generation-benchmark-ngram-score (generated corpus size)
  (let ((index 0)
        (total 0)
        (hits 0))
    (while (<= (+ index size) (length generated))
      (let ((piece (substring generated index (+ index size))))
        (setq total (1+ total))
        (when (> (photon-generation-benchmark-count-substring piece corpus) 0)
          (setq hits (1+ hits))))
      (setq index (1+ index)))
    (list (cons 'total total)
          (cons 'hits hits)
          (cons 'hit-permil
                (if (= total 0) 0 (/ (* 1000 hits) total))))))

(defun photon-generation-benchmark-distinct (generated size)
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

(defun photon-generation-benchmark-logsumexp (logits)
  (let ((max-logit (apply 'max logits))
        (sum 0.0)
        (walk logits))
    (while walk
      (setq sum (+ sum (exp (- (car walk) max-logit))))
      (setq walk (cdr walk)))
    (+ max-logit (log sum))))

(defun photon-generation-benchmark-eval-language (model path vocab
                                                        context-size
                                                        chunk-bytes)
  (let ((reader (photon-make-file-token-reader path vocab chunk-bytes))
        (window nil)
        (filled 0)
        (token nil)
        (total 0)
        (correct 0)
        (nll-sum 0.0))
    (setq token (funcall reader))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall reader)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (logits (photon-model-readout-logits model state window))
               (prediction (photon-argmax-index logits))
               (log-z (photon-generation-benchmark-logsumexp logits))
               (target-logit (nth token logits)))
          (when (= prediction token)
            (setq correct (1+ correct)))
          (setq nll-sum (+ nll-sum (- log-z target-logit)))
          (setq total (1+ total))
          (setq window (photon-window-slide window token))
          (setq token (funcall reader)))))
    (list (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'average-nll
                (if (= total 0) 0.0 (/ nll-sum total)))
          (cons 'perplexity
                (if (= total 0) 0.0 (exp (/ nll-sum total)))))))

(defun photon-generation-benchmark-eval-language-tokens (model tokens
                                                               context-size)
  (let ((window nil)
        (filled 0)
        (token nil)
        (total 0)
        (correct 0)
        (nll-sum 0.0))
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
               (prediction (photon-argmax-index logits))
               (log-z (photon-generation-benchmark-logsumexp logits))
               (target-logit (nth token logits)))
          (when (= prediction token)
            (setq correct (1+ correct)))
          (setq nll-sum (+ nll-sum (- log-z target-logit)))
          (setq total (1+ total))
          (setq window (photon-window-slide window token))
          (setq token (car tokens))
          (setq tokens (cdr tokens)))))
    (list (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'average-nll
                (if (= total 0) 0.0 (/ nll-sum total)))
          (cons 'perplexity
                (if (= total 0) 0.0 (exp (/ nll-sum total)))))))

(defun photon-generation-benchmark-repeat-metrics (generated prompt)
  (let ((index (max 1 (length prompt)))
        (repeated 0)
        (total 0)
        (current-run 1)
        (max-run 1)
        (newline-count 0)
        (space-count 0)
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
               (setq newline-count (1+ newline-count))
               (setq newline-run (1+ newline-run))
               (setq space-run 0))
              ((= char ?\s)
               (setq space-count (1+ space-count))
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
          (cons 'repeated-adjacent repeated)
          (cons 'repeated-token-permil
                (if (= total 0) 0 (/ (* 1000 repeated) total)))
          (cons 'max-run max-run)
          (cons 'newline-count newline-count)
          (cons 'newline-permil
                (if (= total 0) 0 (/ (* 1000 newline-count) total)))
          (cons 'space-count space-count)
          (cons 'space-permil
                (if (= total 0) 0 (/ (* 1000 space-count) total)))
          (cons 'max-newline-run max-newline-run)
          (cons 'max-space-run max-space-run)
          (cons 'collapse-score
                (+ (if (= total 0) 0 (/ (* 1000 repeated) total))
                   (* 100 max-run)
                   (* 100 max-newline-run)
                   (* 100 max-space-run))))))

(defun photon-generation-benchmark-compact-contribution (item)
  (list (cons 'component (cdr (assq 'component item)))
        (cons 'prediction-char (cdr (assq 'prediction-char item)))
        (cons 'target-margin (cdr (assq 'target-margin item)))))

(defun photon-generation-benchmark-compact-trace (trace)
  (mapcar
   (lambda (record)
     (list (cons 'step (cdr (assq 'step record)))
           (cons 'context-text (cdr (assq 'context-text record)))
           (cons 'prediction-char (cdr (assq 'prediction-char record)))
           (cons 'top-5
                 (mapcar
                  (lambda (item)
                    (list (cons 'char (cdr (assq 'char item)))
                          (cons 'logit (cdr (assq 'logit item)))))
                  (cdr (assq 'top-5 record))))
           (cons 'raw-top-5
                 (mapcar
                  (lambda (item)
                    (list (cons 'char (cdr (assq 'char item)))
                          (cons 'logit (cdr (assq 'logit item)))))
                  (cdr (assq 'raw-top-5 record))))
           (cons 'contributions
                 (mapcar
                  'photon-generation-benchmark-compact-contribution
                  (cdr (assq 'readout-contributions record))))))
   trace))

(defun photon-generation-benchmark-run-one (model corpus prompt steps)
  (let* ((result (photon-model-generate-text-with-trace
                  model prompt steps))
         (generated (cdr (assq 'generated result)))
         (trace (cdr (assq 'trace result))))
    (list (cons 'prompt prompt)
          (cons 'generated generated)
          (cons 'generated-length (length generated))
          (cons 'collapse-metrics
                (photon-generation-benchmark-repeat-metrics
                 generated prompt))
          (cons 'bigram-score
                (photon-generation-benchmark-ngram-score
                 generated corpus 2))
          (cons 'trigram-score
                (photon-generation-benchmark-ngram-score
                 generated corpus 3))
          (cons 'distinct-1
                (photon-generation-benchmark-distinct generated 1))
          (cons 'distinct-2
                (photon-generation-benchmark-distinct generated 2))
          (cons 'top-k-trace
                (photon-generation-benchmark-compact-trace trace)))))

(defun photon-generation-benchmark-run-one-summary
    (model corpus prompt steps &optional options)
  (let* ((generated (photon-model-generate-text model prompt steps options)))
    (list (cons 'prompt prompt)
          (cons 'generated generated)
          (cons 'generated-length (length generated))
          (cons 'collapse-metrics
                (photon-generation-benchmark-repeat-metrics
                 generated prompt))
          (cons 'bigram-score
                (photon-generation-benchmark-ngram-score
                 generated corpus 2))
          (cons 'trigram-score
                (photon-generation-benchmark-ngram-score
                 generated corpus 3))
          (cons 'distinct-1
                (photon-generation-benchmark-distinct generated 1))
          (cons 'distinct-2
                (photon-generation-benchmark-distinct generated 2)))))

(defun photon-generation-benchmark-sampling-row
    (name model corpus prompts steps options)
  (let ((generations
         (mapcar
          (lambda (prompt)
            (photon-generation-benchmark-run-one-summary
             model corpus prompt steps options))
          prompts)))
    (list (cons 'name name)
          (cons 'options options)
          (cons 'average-repeated-token-permil
                (photon-generation-benchmark-average-field
                 generations '(collapse-metrics repeated-token-permil)))
          (cons 'average-collapse-score
                (photon-generation-benchmark-average-field
                 generations '(collapse-metrics collapse-score)))
          (cons 'average-distinct-1-permil
                (photon-generation-benchmark-average-field
                 generations '(distinct-1 distinct-permil)))
          (cons 'average-distinct-2-permil
                (photon-generation-benchmark-average-field
                 generations '(distinct-2 distinct-permil)))
          (cons 'generations generations))))

(defun photon-generation-benchmark-options-with-seed (options seed)
  (cons (cons 'seed seed)
        (assq-delete-all 'seed (copy-sequence options))))

(defun photon-generation-benchmark-sampling-row-summary (row)
  (list (cons 'name (cdr (assq 'name row)))
        (cons 'options (cdr (assq 'options row)))
        (cons 'average-repeated-token-permil
              (cdr (assq 'average-repeated-token-permil row)))
        (cons 'average-collapse-score
              (cdr (assq 'average-collapse-score row)))
        (cons 'average-distinct-1-permil
              (cdr (assq 'average-distinct-1-permil row)))
        (cons 'average-distinct-2-permil
              (cdr (assq 'average-distinct-2-permil row)))))

(defun photon-generation-benchmark-sampling-row-seeds
    (name model corpus prompts steps options seeds)
  (let ((rows nil)
        (walk seeds))
    (while walk
      (setq rows
            (cons
             (photon-generation-benchmark-sampling-row-summary
              (photon-generation-benchmark-sampling-row
               name model corpus prompts steps
               (photon-generation-benchmark-options-with-seed
                options (car walk))))
             rows))
      (setq walk (cdr walk)))
    (setq rows (nreverse rows))
    (list (cons 'name name)
          (cons 'seeds seeds)
          (cons 'options options)
          (cons 'average-repeated-token-permil
                (photon-generation-benchmark-average-field
                 rows '(average-repeated-token-permil)))
          (cons 'average-collapse-score
                (photon-generation-benchmark-average-field
                 rows '(average-collapse-score)))
          (cons 'average-distinct-1-permil
                (photon-generation-benchmark-average-field
                 rows '(average-distinct-1-permil)))
          (cons 'average-distinct-2-permil
                (photon-generation-benchmark-average-field
                 rows '(average-distinct-2-permil)))
          (cons 'seed-summaries rows))))

(defun photon-generation-benchmark-sampling-quality-score (row)
  (- (cdr (assq 'average-distinct-2-permil row))
     (cdr (assq 'average-collapse-score row))))

(defun photon-generation-benchmark-bias-options (ngram-weight repeat-weight)
  (let ((options '((mode . sample) (temperature . 0.85) (top-k . 5)
                   (repetition-penalty . 1.10))))
    (when (> ngram-weight 0.0)
      (setq options (append options
                            (list (cons 'ngram-bias ngram-weight)))))
    (when (> repeat-weight 0.0)
      (setq options (append options
                            (list (cons 'repeat-ngram-bias
                                        repeat-weight)))))
    options))

(defun photon-generation-benchmark-sampling-weight-sweep
    (model corpus prompts steps seeds ngram-weights repeat-weights)
  (let ((rows nil)
        (ngram-walk ngram-weights))
    (while ngram-walk
      (let ((repeat-walk repeat-weights)
            (ngram-weight (car ngram-walk)))
        (while repeat-walk
          (let* ((repeat-weight (car repeat-walk))
                 (row (photon-generation-benchmark-sampling-row-seeds
                       (intern (format "ngram-%.2f-repeat-%.2f"
                                       ngram-weight repeat-weight))
                       model corpus prompts steps
                       (photon-generation-benchmark-bias-options
                        ngram-weight repeat-weight)
                       seeds)))
            (setq row
                  (append row
                          (list
                           (cons 'ngram-bias ngram-weight)
                           (cons 'repeat-ngram-bias repeat-weight)
                           (cons 'quality-score
                                 (photon-generation-benchmark-sampling-quality-score
                                  row)))))
            (setq rows (cons row rows)))
          (setq repeat-walk (cdr repeat-walk))))
      (setq ngram-walk (cdr ngram-walk)))
    (let* ((ordered (sort (copy-sequence rows)
                          (lambda (left right)
                            (> (cdr (assq 'quality-score left))
                               (cdr (assq 'quality-score right)))))))
      (list (cons 'seeds seeds)
            (cons 'ngram-weights ngram-weights)
            (cons 'repeat-ngram-weights repeat-weights)
            (cons 'best (car ordered))
            (cons 'rows ordered)))))

(defun photon-generation-benchmark-sampling-profile-validation
    (model train-corpus eval-corpus train-prompts eval-prompts steps seeds)
  (let ((profiles
         (list
          (list (cons 'name 'baseline)
                (cons 'options
                      '((mode . sample) (temperature . 0.85) (top-k . 5)
                        (repetition-penalty . 1.10))))
          (list (cons 'name 'recommended)
                (cons 'options
                      '((mode . sample) (temperature . 0.85) (top-k . 5)
                        (repetition-penalty . 1.10)
                        (ngram-bias . 0.25)
                        (repeat-ngram-bias . 0.35))))
          (list (cons 'name 'aggressive)
                (cons 'options
                      '((mode . sample) (temperature . 0.85) (top-k . 5)
                        (repetition-penalty . 1.10)
                        (ngram-bias . 0.40)
                        (repeat-ngram-bias . 0.35)))))))
    (list
     (cons 'seeds seeds)
     (cons 'train-prompts train-prompts)
     (cons 'eval-prompts eval-prompts)
     (cons
      'train
      (mapcar
       (lambda (profile)
         (let ((row (photon-generation-benchmark-sampling-row-seeds
                     (cdr (assq 'name profile))
                     model train-corpus train-prompts steps
                     (cdr (assq 'options profile))
                     seeds)))
           (append row
                   (list
                    (cons 'quality-score
                          (photon-generation-benchmark-sampling-quality-score
                           row))))))
       profiles))
     (cons
      'eval
      (mapcar
       (lambda (profile)
         (let ((row (photon-generation-benchmark-sampling-row-seeds
                     (cdr (assq 'name profile))
                     model eval-corpus eval-prompts steps
                     (cdr (assq 'options profile))
                     seeds)))
           (append row
                   (list
                    (cons 'quality-score
                          (photon-generation-benchmark-sampling-quality-score
                           row))))))
       profiles)))))

(defun photon-generation-benchmark-average-field (records path)
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

(defun photon-generation-benchmark-level-row
    (level heldout fallback generations)
  (list (cons 'level level)
        (cons 'heldout-accuracy-permil
              (cdr (assq 'accuracy-permil heldout)))
        (cons 'fallback-accuracy-permil
              (cdr (assq 'accuracy-permil fallback)))
        (cons 'average-nll (cdr (assq 'average-nll heldout)))
        (cons 'perplexity (cdr (assq 'perplexity heldout)))
        (cons 'average-repeated-token-permil
              (photon-generation-benchmark-average-field
               generations '(collapse-metrics repeated-token-permil)))
        (cons 'average-collapse-score
              (photon-generation-benchmark-average-field
               generations '(collapse-metrics collapse-score)))
        (cons 'average-distinct-1-permil
              (photon-generation-benchmark-average-field
               generations '(distinct-1 distinct-permil)))
        (cons 'average-distinct-2-permil
              (photon-generation-benchmark-average-field
               generations '(distinct-2 distinct-permil)))))

(defun photon-generation-benchmark-tokenizer-preview (corpus prompts)
  (let ((vocab (photon-build-wordpiece-vocab
                (concat corpus "\n" (mapconcat 'identity prompts "\n"))
                4))
        (result nil))
    (while prompts
      (let* ((prompt (car prompts))
             (tokens (photon-encode-wordpiece-text vocab prompt 4)))
        (setq result
              (cons (list (cons 'prompt prompt)
                          (cons 'char-length (length prompt))
                          (cons 'wordpiece-token-count (length tokens))
                          (cons 'round-trip
                                (photon-decode-wordpiece-tokens
                                 vocab tokens)))
                    result)))
      (setq prompts (cdr prompts)))
    (list (cons 'vocab-size (length vocab))
          (cons 'prompts (nreverse result)))))

(let* ((root (file-name-directory load-file-name))
       (train-path (expand-file-name "../data/stream-sample.txt" root))
       (eval-path (expand-file-name "../data/stream-eval-sample.txt" root))
       (corpus (photon-read-text-file train-path))
       (eval-corpus (photon-read-text-file eval-path))
       (context-size 6)
       (chunk-bytes 16)
       (result (photon-train-text-file-stream-split
                train-path eval-path context-size 8 3 2 0.20 chunk-bytes))
       (model (cdr (assq 'model result)))
       (char-heldout
        (photon-generation-benchmark-eval-language
         model eval-path (cdr (assq 'vocab result))
         context-size chunk-bytes))
       (prompts '("abc" "ph" "ne" "stream"))
       (eval-prompts '("reader" "heldout" "accuracy" "checked"))
       (char-generations
        (mapcar
         (lambda (prompt)
           (photon-generation-benchmark-run-one
            model corpus prompt 24))
         prompts))
       (wordpiece-context-size 3)
       (wordpiece-result
        (photon-train-wordpiece-text-file-split
         train-path eval-path wordpiece-context-size 8 3 2 0.20 4))
       (wordpiece-model (cdr (assq 'model wordpiece-result)))
       (wordpiece-eval-tokens (cdr (assq 'eval-tokens wordpiece-result)))
       (wordpiece-heldout
        (photon-generation-benchmark-eval-language-tokens
         wordpiece-model wordpiece-eval-tokens wordpiece-context-size))
       (wordpiece-fallback (cdr (assq 'fallback-after wordpiece-result)))
       (wordpiece-diagnostics
        (photon-wordpiece-diagnostics
         wordpiece-model wordpiece-eval-tokens wordpiece-context-size 3))
       (wordpiece-generations
        (mapcar
         (lambda (prompt)
           (photon-generation-benchmark-run-one-summary
            wordpiece-model corpus prompt 24))
         prompts))
       (sampling-comparison
        (list
         (photon-generation-benchmark-sampling-row
          'greedy wordpiece-model corpus prompts 24 nil)
         (photon-generation-benchmark-sampling-row
          'temperature-top-k wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (seed . 17)))
         (photon-generation-benchmark-sampling-row
          'temperature-top-p wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.95) (top-p . 0.85)
            (repetition-penalty . 1.20) (seed . 17)))))
       (sampling-bias-ablation
        (list
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-baseline wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-ngram-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (ngram-bias . 0.40))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-gated-ngram-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (ngram-bias . 0.40)
            (ngram-bias-gate-margin . 0.0))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-adaptive-ngram-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (ngram-bias . 0.40)
            (ngram-bias-repeat-scale . 0.0))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-repeat-ngram-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (repeat-ngram-bias . 0.35))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-adaptive-repeat-ngram-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (repeat-ngram-bias . 0.35)
            (repeat-ngram-bias-min-count . 2.0))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'top-k-ngram-and-repeat-bias wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10) (ngram-bias . 0.40)
            (repeat-ngram-bias . 0.35))
          '(11 17 23))))
       (sampling-weight-sweep
        (photon-generation-benchmark-sampling-weight-sweep
         wordpiece-model corpus prompts 24 '(11 17 23)
         '(0.0 0.25 0.40) '(0.0 0.20 0.35)))
       (sampling-profile-validation
        (photon-generation-benchmark-sampling-profile-validation
         wordpiece-model corpus eval-corpus prompts eval-prompts
         24 '(11 17 23)))
       (sampling-seed-comparison
        (list
         (photon-generation-benchmark-sampling-row-seeds
          'temperature-top-k wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.85) (top-k . 5)
            (repetition-penalty . 1.10))
          '(11 17 23))
         (photon-generation-benchmark-sampling-row-seeds
          'temperature-top-p wordpiece-model corpus prompts 24
          '((mode . sample) (temperature . 0.95) (top-p . 0.85)
            (repetition-penalty . 1.20))
          '(11 17 23)))))
  (prin1
   (list (cons 'train-path train-path)
         (cons 'eval-path eval-path)
         (cons 'after (cdr (assq 'after result)))
         (cons 'fallback-after (cdr (assq 'fallback-after result)))
         (cons 'heldout-language char-heldout)
         (cons 'level-comparison
               (list
                (photon-generation-benchmark-level-row
                 'char char-heldout (cdr (assq 'fallback-after result))
                 char-generations)
                (photon-generation-benchmark-level-row
                 'wordpiece wordpiece-heldout wordpiece-fallback
                 wordpiece-generations)))
         (cons 'tokenizer-preview
               (photon-generation-benchmark-tokenizer-preview
                corpus prompts))
         (cons 'wordpiece-heldout-language wordpiece-heldout)
         (cons 'wordpiece-fallback-after wordpiece-fallback)
         (cons 'wordpiece-diagnostics
               (list (cons 'summary
                           (cdr (assq 'summary wordpiece-diagnostics)))))
         (cons 'sampling-comparison sampling-comparison)
         (cons 'sampling-bias-ablation sampling-bias-ablation)
         (cons 'sampling-weight-sweep sampling-weight-sweep)
         (cons 'sampling-profile-validation sampling-profile-validation)
         (cons 'sampling-seed-comparison sampling-seed-comparison)
         (cons 'wordpiece-generations wordpiece-generations)
         (cons 'generations char-generations)))
  (princ "\n"))

;;; generation-benchmark.el ends here
