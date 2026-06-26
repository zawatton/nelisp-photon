;;; stream-diagnostics.el --- Stream fallback benchmark diagnostics -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-elisp-vector-backend)

(defun photon-stream-diagnostics-compact-top (items)
  (mapcar (lambda (item)
            (list (cons 'char (cdr (assq 'char item)))
                  (cons 'logit (cdr (assq 'logit item)))))
          items))

(defun photon-stream-diagnostics-compact-contribution (item)
  (list (cons 'component (cdr (assq 'component item)))
        (cons 'prediction-char (cdr (assq 'prediction-char item)))
        (cons 'target-minus-fallback
              (cdr (assq 'target-minus-fallback item)))
        (cons 'target-margin (cdr (assq 'target-margin item)))))

(defun photon-stream-diagnostics-compact-record (record)
  (list (cons 'count (or (cdr (assq 'count record)) 1))
        (cons 'context-text (cdr (assq 'context-text record)))
        (cons 'target-char (cdr (assq 'target-char record)))
        (cons 'target-class (cdr (assq 'target-class record)))
        (cons 'fallback-char (cdr (assq 'fallback-char record)))
        (cons 'exact-char (cdr (assq 'exact-char record)))
        (cons 'memory-hit (cdr (assq 'memory-hit record)))
        (cons 'margin (cdr (assq 'margin record)))
        (cons 'top-3
              (photon-stream-diagnostics-compact-top
               (cdr (assq 'top-3 record))))
        (cons 'contributions
              (mapcar
               'photon-stream-diagnostics-compact-contribution
               (cdr (assq 'readout-contributions record))))))

(let* ((train-path (expand-file-name "../data/stream-sample.txt"
                                     (file-name-directory load-file-name)))
       (eval-path (expand-file-name "../data/stream-eval-sample.txt"
                                    (file-name-directory load-file-name)))
       (context-size 6)
       (chunk-bytes 16)
       (result (photon-train-text-file-stream-split
                train-path eval-path context-size 8 3 2 0.20 chunk-bytes))
       (model (cdr (assq 'model result)))
       (vocab (cdr (assq 'vocab result)))
       (diagnostics
        (photon-stream-file-diagnostics
         model eval-path vocab context-size chunk-bytes 10)))
  (prin1
   (list (cons 'vocab-size (length vocab))
         (cons 'train-path train-path)
         (cons 'eval-path eval-path)
         (cons 'before (cdr (assq 'before result)))
         (cons 'cluster-distill (cdr (assq 'cluster-distill result)))
         (cons 'output-head-compress
               (cdr (assq 'output-head-compress result)))
         (cons 'after (cdr (assq 'after result)))
         (cons 'fallback-after (cdr (assq 'fallback-after result)))
         (cons 'summary (cdr (assq 'summary diagnostics)))
         (cons 'miss-summary
               (mapcar
                'photon-stream-diagnostics-compact-record
                (photon-take
                 (cdr (assq 'fallback-miss-summary diagnostics))
                 8)))
         (cons 'ngram-head-gap-summary
               (mapcar
                'photon-stream-diagnostics-compact-record
                (photon-take
                 (cdr (assq 'ngram-head-gap-summary diagnostics))
                 8)))
         (cons 'ngram-head-gaps
               (mapcar
                'photon-stream-diagnostics-compact-record
                (cdr (assq 'ngram-head-gaps diagnostics))))
         (cons 'newline-targets
               (mapcar
                'photon-stream-diagnostics-compact-record
                (cdr (assq 'newline-targets diagnostics))))))
  (princ "\n"))

;;; stream-diagnostics.el ends here
