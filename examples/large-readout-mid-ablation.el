;;; large-readout-mid-ablation.el --- Mid-size start-long readout ablation -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun nil)
(defvar photon-large-readout-mid-ablation-autorun t)

(setq photon-large-generation-benchmark-autorun nil)
(load-file (expand-file-name
            "large-generation-benchmark.el"
            (file-name-directory load-file-name)))

(defun photon-large-readout-mid-ablation-reset ()
  (setq photon-large-generation-benchmark-autorun nil)
  (setq photon-large-readout-diagnostic-token-limit 80)
  (setq photon-large-readout-full-distill-token-limit 80)
  (setq photon-large-readout-long-rare-epochs 1)
  (setq photon-large-readout-prototype-distill-epochs 1)
  (setq photon-large-readout-ngram-long-prototype-distill-epochs 1)
  (setq photon-large-readout-post-ngram-long-prototype-refresh-epochs 1)
  (setq photon-large-readout-full-distill-epochs 1)
  (setq photon-large-readout-train-token-limit 240)
  (setq photon-large-readout-eval-token-limit 160)
  (setq photon-large-readout-context-size 4)
  (setq photon-large-readout-hidden-size 8)
  (setq photon-large-readout-chunk-size 3)
  (setq photon-large-readout-epochs 1)
  (setq photon-large-readout-learning-rate 0.18)
  (setq photon-large-readout-max-piece-size 4)
  (setq photon-large-readout-distill-candidates
        '(full-scaled start-long-classifier-teacher-scaled
          start-long-ngram-teacher-scaled full-unscaled))
  (setq photon-large-readout-full-distill-selection-after-weight 8.0)
  (setq photon-large-readout-full-distill-start-long-classifier-teacher-weight
        8.0)
  (setq photon-large-readout-full-distill-start-long-ngram-teacher-weight 8.0)
  (setq photon-large-readout-start-long-prototype-merge-rate nil)
  (setq photon-large-readout-start-long-prototype-merge-epochs nil)
  (setq photon-large-readout-start-long-prototype-merge-top-k nil)
  (setq photon-large-readout-start-long-prototype-merge-margin nil)
  (setq photon-large-readout-start-long-prototype-merge-target-bias-rate nil)
  (setq photon-large-readout-start-long-prototype-merge-rival-bias-rate nil)
  (setq photon-large-readout-start-long-prototype-merge-score-bonus-rate nil)
  (setq photon-large-readout-start-long-prototype-merge-dominant-rival-bias-rate
        nil)
  (setq photon-large-readout-start-long-prototype-merge-dominant-rival-min-count
        nil)
  (setq photon-large-readout-start-long-prototype-merge-state-separation-rate
        nil)
  (setq photon-large-readout-start-long-prototype-merge-dense-head-rate nil)
  (setq photon-large-readout-start-long-prototype-merge-rival-line-search-step
        nil)
  (setq photon-large-readout-start-long-prototype-merge-rival-line-search-max
        nil)
  (setq photon-large-readout-start-long-prototype-merge-rival-negative-weight
        nil)
  (setq photon-large-readout-start-long-prototype-merge-rivals nil)
  (setq photon-large-readout-start-long-prototype-merge-pocket-enabled nil)
  (setq photon-large-readout-start-long-prototype-merge-candidate-search-enabled
        nil)
  (setq photon-large-readout-start-long-prototype-merge-candidate-limit nil)
  (setq photon-large-readout-start-long-prototype-merge-global-tolerance nil)
  (setq photon-large-readout-start-long-projection-separation-rate nil)
  (setq photon-large-readout-start-long-projection-separation-epochs nil)
  (setq photon-large-readout-start-long-projection-separation-top-k nil)
  (setq photon-large-readout-start-long-projection-separation-margin nil)
  (setq photon-large-readout-start-long-projection-separation-score-bonus-rate
        nil)
  (setq photon-large-readout-start-long-projection-separation-rival-bias-rate
        nil)
  (setq photon-large-readout-start-long-projection-separation-pocket-enabled
        nil)
  (setq photon-large-readout-start-long-projection-separation-global-tolerance
        nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-weight nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-bias-weight
        nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-epochs nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-top-k nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-rivals nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-margin nil)
  (setq photon-large-readout-start-long-same-shape-margin-distill-adaptive-bias-max
        nil)
  (setq photon-large-readout-start-long-same-shape-top1-search-enabled nil)
  (setq photon-large-readout-start-long-same-shape-top1-search-score-bonus-max
        nil)
  (setq photon-large-readout-start-long-same-shape-top1-search-context-bonus-weight
        nil)
  (setq photon-large-readout-start-long-same-shape-top1-search-global-tolerance
        nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-enabled nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale
        nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-readout-weight nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates
        nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled
        nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate
        nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-candidate-top-k nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-candidate-margin nil)
  (setq photon-large-readout-wordpiece-start-long-classifier-candidate-limit nil))

(defun photon-large-readout-mid-ablation-apply (settings)
  (while settings
    (set (caar settings) (cdar settings))
    (setq settings (cdr settings))))

(defun photon-large-readout-mid-ablation-candidate-score (summary)
  (+ (* 8 (cdr (assq 'after summary)))
     (cdr (assq 'accuracy-permil (cdr (assq 'fallback summary))))))

(defun photon-large-readout-mid-ablation-weight-summary (weight summary)
  (let ((fallback (cdr (assq 'fallback summary))))
    (list
     (cons 'weight weight)
     (cons 'after (cdr (assq 'after summary)))
     (cons 'fallback (cdr (assq 'accuracy-permil fallback)))
     (cons 'top-5 (cdr (assq 'top-5-accuracy-permil fallback)))
     (cons 'score
           (photon-large-readout-mid-ablation-candidate-score summary)))))

(defun photon-large-readout-mid-ablation-select-readout-weight
    (name settings weights)
  (let ((candidates nil)
        (remaining weights)
        best)
    (while remaining
      (photon-large-readout-mid-ablation-reset)
      (photon-large-readout-mid-ablation-apply
       (cons
        (cons 'photon-large-readout-wordpiece-start-long-classifier-readout-weight
              (car remaining))
        (assq-delete-all
         'photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates
         (copy-sequence settings))))
      (setq candidates
            (cons
             (cons (cons 'weight (car remaining))
                   (photon-large-readout-compact-result))
             candidates))
      (setq remaining (cdr remaining)))
    (setq candidates (nreverse candidates))
    (setq best (car candidates))
    (setq remaining (cdr candidates))
    (while remaining
      (let* ((candidate (car remaining))
             (candidate-score
              (photon-large-readout-mid-ablation-candidate-score
               (cdr candidate)))
             (best-score
              (photon-large-readout-mid-ablation-candidate-score
               (cdr best))))
        (when (> candidate-score best-score)
          (setq best candidate)))
      (setq remaining (cdr remaining)))
    (let ((summary (copy-sequence (cdr best)))
          (selection
           (list
            (cons 'selected
                  (photon-large-readout-mid-ablation-weight-summary
                   (cdr (assq 'weight best))
                   (cdr best)))
            (cons 'candidates
                  (mapcar
                   (lambda (candidate)
                     (photon-large-readout-mid-ablation-weight-summary
                      (cdr (assq 'weight candidate))
                      (cdr candidate)))
                   candidates)))))
      (setq summary (assq-delete-all 'classifier-readout-selection summary))
      (cons (cons 'name name)
            (cons (cons 'classifier-readout-selection selection)
                  summary)))))

(defun photon-large-readout-mid-ablation-row (name settings)
  (photon-large-readout-mid-ablation-reset)
  (photon-large-readout-mid-ablation-apply settings)
  (let ((weights
         photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates))
    (if weights
        (photon-large-readout-mid-ablation-select-readout-weight
         name settings weights)
      (let ((summary (photon-large-readout-compact-result)))
        (cons (cons 'name name) summary)))))

(defun photon-large-readout-mid-ablation-result ()
  (list
   (photon-large-readout-mid-ablation-row 'baseline nil)
   (photon-large-readout-mid-ablation-row
    'prototype-merge-pocket
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-epochs . 1)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-pocket-enabled . 1.0)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-pocket-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-epochs . 1)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-pocket-enabled . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-rival-bias-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-rival-bias-rate . 0.05)
      (photon-large-readout-start-long-prototype-merge-rivals . 3)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-rival-line-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-rivals . 3)
      (photon-large-readout-start-long-prototype-merge-rival-line-search-step
       . 0.02)
      (photon-large-readout-start-long-prototype-merge-rival-line-search-max
       . 0.12)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-score-bonus-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-score-bonus-rate . 0.01)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-dominant-rival-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-rivals . 3)
      (photon-large-readout-start-long-prototype-merge-dominant-rival-bias-rate
       . 0.03)
      (photon-large-readout-start-long-prototype-merge-dominant-rival-min-count
       . 3)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-state-separation-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-rivals . 3)
      (photon-large-readout-start-long-prototype-merge-state-separation-rate
       . 0.25)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'prototype-merge-dense-head-search
    '((photon-large-readout-start-long-prototype-merge-rate . 1.0)
      (photon-large-readout-start-long-prototype-merge-top-k . 10)
      (photon-large-readout-start-long-prototype-merge-margin . 1.0)
      (photon-large-readout-start-long-prototype-merge-rivals . 3)
      (photon-large-readout-start-long-prototype-merge-dense-head-rate . 0.02)
      (photon-large-readout-start-long-prototype-merge-candidate-search-enabled
       . 1.0)
      (photon-large-readout-start-long-prototype-merge-candidate-limit . 16)))
   (photon-large-readout-mid-ablation-row
    'start-long-classifier-readout-selector
    '((photon-large-readout-wordpiece-start-long-classifier-enabled . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates
       . (0.0 0.05 0.10 0.25))
      (photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate
       . 0.20)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-top-k
       . 10)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-margin
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-limit
       . 16)))
   (photon-large-readout-mid-ablation-row
    'start-long-classifier-readout-selector-top1-search
    '((photon-large-readout-wordpiece-start-long-classifier-enabled . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-learning-rate-scale
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-readout-weight-candidates
       . (0.0 0.05 0.10 0.25))
      (photon-large-readout-wordpiece-start-long-classifier-candidate-search-enabled
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-learning-rate
       . 0.20)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-top-k
       . 10)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-margin
       . 1.0)
      (photon-large-readout-wordpiece-start-long-classifier-candidate-limit
       . 16)
      (photon-large-readout-start-long-same-shape-top1-search-enabled
       . 1.0)
      (photon-large-readout-start-long-same-shape-top1-search-score-bonus-max
       . 0.10)
      (photon-large-readout-start-long-same-shape-top1-search-context-bonus-weight
       . 0.05)
      (photon-large-readout-start-long-same-shape-top1-search-global-tolerance
       . 0)
      (photon-large-readout-start-long-same-shape-margin-distill-top-k . 10)
      (photon-large-readout-start-long-same-shape-margin-distill-rivals . 2)
      (photon-large-readout-start-long-same-shape-margin-distill-margin . 1.0)
      (photon-large-readout-start-long-same-shape-margin-distill-bias-weight
       . 0.02)
      (photon-large-readout-start-long-same-shape-margin-distill-adaptive-bias-max
       . 0.05)))
   (photon-large-readout-mid-ablation-row
    'projection-separation-pocket
    '((photon-large-readout-start-long-projection-separation-rate . 0.05)
      (photon-large-readout-start-long-projection-separation-epochs . 1)
      (photon-large-readout-start-long-projection-separation-top-k . 10)
      (photon-large-readout-start-long-projection-separation-margin . 0.90)
      (photon-large-readout-start-long-projection-separation-score-bonus-rate
       . 0.05)
      (photon-large-readout-start-long-projection-separation-rival-bias-rate
       . 0.05)
      (photon-large-readout-start-long-projection-separation-pocket-enabled
       . 1.0)))))

(when photon-large-readout-mid-ablation-autorun
  (prin1 (photon-large-readout-mid-ablation-result))
  (princ "\n"))

;;; large-readout-mid-ablation.el ends here
