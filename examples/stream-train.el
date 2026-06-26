(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-elisp-vector-backend)

(let* ((train-path (expand-file-name "../data/stream-sample.txt"
                                     (file-name-directory load-file-name)))
       (eval-path (expand-file-name "../data/stream-eval-sample.txt"
                                    (file-name-directory load-file-name)))
       (result (photon-train-text-file-stream-split
                train-path eval-path 6 8 3 2 0.20 16))
       (model (cdr (assq 'model result)))
       (summary
        (list (cons 'vocab-size (length (cdr (assq 'vocab result))))
              (cons 'stream-chunk-bytes
                    (photon-model-metadata-get model 'stream-chunk-bytes))
              (cons 'train-path
                    (photon-model-metadata-get model 'train-path))
              (cons 'eval-path
                    (photon-model-metadata-get model 'eval-path))
              (cons 'before (cdr (assq 'before result)))
              (cons 'history (cdr (assq 'history result)))
              (cons 'after (cdr (assq 'after result)))
              (cons 'fallback-after
                    (cdr (assq 'fallback-after result))))))
  (prin1 summary)
  (princ "\n"))
