;;; large-readout-benchmark.el --- Fast larger stream readout probe -*- lexical-binding: t; -*-

(defvar photon-large-generation-benchmark-autorun nil)
(defvar photon-large-readout-distill-candidates
  '(full-scaled start-long-classifier-teacher-scaled
    start-long-ngram-teacher-scaled full-unscaled))
(defvar photon-large-readout-full-distill-selection-after-weight 8.0)
(defvar photon-large-readout-full-distill-start-long-classifier-teacher-weight
  8.0)
(defvar photon-large-readout-full-distill-start-long-ngram-teacher-weight 8.0)

(setq photon-large-generation-benchmark-autorun nil)
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

(prin1 (assq-delete-all 'model (photon-large-readout-result)))
(princ "\n")

;;; large-readout-benchmark.el ends here
