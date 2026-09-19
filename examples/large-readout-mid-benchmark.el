;;; large-readout-mid-benchmark.el --- Mid-size larger-stream readout probe -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun nil)
(defvar photon-large-readout-diagnostic-token-limit 80)
(defvar photon-large-readout-full-distill-token-limit 80)
(defvar photon-large-readout-long-rare-epochs 1)
(defvar photon-large-readout-prototype-distill-epochs 1)
(defvar photon-large-readout-ngram-long-prototype-distill-epochs 1)
(defvar photon-large-readout-post-ngram-long-prototype-refresh-epochs 1)
(defvar photon-large-readout-full-distill-epochs 1)
(defvar photon-large-readout-train-token-limit 240)
(defvar photon-large-readout-eval-token-limit 160)
(defvar photon-large-readout-context-size 4)
(defvar photon-large-readout-hidden-size 8)
(defvar photon-large-readout-chunk-size 3)
(defvar photon-large-readout-epochs 1)
(defvar photon-large-readout-learning-rate 0.18)
(defvar photon-large-readout-max-piece-size 4)
(defvar photon-large-readout-distill-candidates
  '(full-scaled start-long-classifier-teacher-scaled
    start-long-ngram-teacher-scaled full-unscaled))
(defvar photon-large-readout-full-distill-selection-after-weight 8.0)
(defvar photon-large-readout-full-distill-start-long-classifier-teacher-weight
  8.0)
(defvar photon-large-readout-full-distill-start-long-ngram-teacher-weight 8.0)

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

(load-file (expand-file-name
            "large-generation-benchmark.el"
            (file-name-directory load-file-name)))

(prin1 (photon-large-readout-compact-result))
(princ "\n")

;;; large-readout-mid-benchmark.el ends here
