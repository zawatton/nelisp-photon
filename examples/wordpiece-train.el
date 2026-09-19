;;; wordpiece-train.el --- Wordpiece training path example -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))
(require 'seq)

(photon-use-elisp-vector-backend)

(defun photon-wordpiece-example-compact-top (items)
  (mapcar (lambda (item)
            (list (cons 'piece (cdr (assq 'piece item)))
                  (cons 'token-kind (cdr (assq 'token-kind item)))
                  (cons 'logit (cdr (assq 'logit item)))))
          items))

(defun photon-wordpiece-example-compact-contribution (item)
  (list (cons 'component (cdr (assq 'component item)))
        (cons 'prediction-piece (cdr (assq 'prediction-piece item)))
        (cons 'prediction-kind (cdr (assq 'prediction-kind item)))
        (cons 'target-minus-fallback
              (cdr (assq 'target-minus-fallback item)))
        (cons 'target-margin (cdr (assq 'target-margin item)))))

(defun photon-wordpiece-example-interesting-contribution-p (item)
  (let ((component (cdr (assq 'component item))))
    (or (eq component 'output-head)
        (eq component 'raw-output-head)
        (eq component 'dense-prototype-readout)
        (eq component 'prototype)
        (eq component 'ngram-anchor)
        (eq component 'token-readout)
        (eq component 'linear-output-head)
        (eq component 'full))))

(defun photon-wordpiece-example-compact-record (record)
  (list (cons 'index (cdr (assq 'index record)))
        (cons 'context-pieces (cdr (assq 'context-pieces record)))
        (cons 'target-piece (cdr (assq 'target-piece record)))
        (cons 'target-kind (cdr (assq 'target-kind record)))
        (cons 'fallback-piece (cdr (assq 'fallback-piece record)))
        (cons 'memory-hit (cdr (assq 'memory-hit record)))
        (cons 'margin (cdr (assq 'margin record)))
        (cons 'top-5
              (photon-wordpiece-example-compact-top
               (cdr (assq 'top-5 record))))
        (cons 'readout-contributions
              (mapcar
               'photon-wordpiece-example-compact-contribution
               (seq-filter
                'photon-wordpiece-example-interesting-contribution-p
                (cdr (assq 'readout-contributions record)))))))

(let* ((root (file-name-directory load-file-name))
       (train-path (expand-file-name "../data/stream-sample.txt" root))
       (eval-path (expand-file-name "../data/stream-eval-sample.txt" root))
       (result (photon-train-wordpiece-text-file-split
                train-path eval-path 3 8 3 2 0.20 4))
       (model (cdr (assq 'model result)))
       (diagnostics
        (photon-wordpiece-diagnostics
         model (cdr (assq 'eval-tokens result)) 3 5)))
  (prin1
   (list (cons 'vocab-size (length (cdr (assq 'vocab result))))
         (cons 'before (cdr (assq 'before result)))
         (cons 'cluster-distill (cdr (assq 'cluster-distill result)))
         (cons 'output-head-compress
               (cdr (assq 'output-head-compress result)))
         (cons 'wordpiece-final-ngram-rescue
               (cdr (assq 'wordpiece-final-ngram-rescue result)))
         (cons 'full-readout-distill
               (cdr (assq 'full-readout-distill result)))
         (cons 'after (cdr (assq 'after result)))
         (cons 'fallback-after (cdr (assq 'fallback-after result)))
         (cons 'wordpiece-diagnostics
               (list
                (cons 'summary (cdr (assq 'summary diagnostics)))
                (cons 'fallback-miss-summary
                      (mapcar
                       'photon-wordpiece-example-compact-record
                       (photon-take
                        (cdr (assq 'fallback-miss-summary diagnostics))
                        5)))
                (cons 'fallback-misses
                      (mapcar
                       'photon-wordpiece-example-compact-record
                       (cdr (assq 'fallback-misses diagnostics))))))
         (cons 'generated
               (photon-model-generate-text model "photon" 8))))
  (princ "\n"))

;;; wordpiece-train.el ends here
