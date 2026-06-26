(load "lisp/photon.el")

(photon-use-elisp-vector-backend)

(let* ((config (photon-make-config 8 6 3 2))
       (model (photon-make-model config))
       (context '(1 2 3))
       (reset-stats (photon-model-reset-cache-stats model))
       (first (photon-predict-one model context))
       (cache-after-first (length (photon-model-cache model)))
       (second (photon-predict-one model context))
       (cache-after-second (length (photon-model-cache model)))
       (extended-states (photon-forward-model model '(1 2 3 4)))
       (repeated-chunk-state (photon-model-chunker model context))
       (stats-after-second (photon-model-cache-stats model))
       (corpus '(1 2 3 1 2 3 1 2 3))
       (history (photon-train model corpus 3 2 0.20))
       (cache-after-train (length (photon-model-cache model))))
  (list (cons 'backend (photon-vector-backend-name))
        (cons 'first-prediction (cdr (assq 'prediction first)))
        (cons 'second-prediction (cdr (assq 'prediction second)))
        (cons 'cache-after-first cache-after-first)
        (cons 'cache-after-second cache-after-second)
        (cons 'extended-state-count (length extended-states))
        (cons 'repeated-chunk-size (length repeated-chunk-state))
        (cons 'prediction-window-hits
              (photon-cache-stats-kind-get stats-after-second
                                           'prediction-window 'hits))
        (cons 'prediction-window-misses
              (photon-cache-stats-kind-get stats-after-second
                                           'prediction-window 'misses))
        (cons 'chunk-state-hits
              (photon-cache-stats-kind-get stats-after-second
                                           'chunk-state 'hits))
        (cons 'chunk-state-misses
              (photon-cache-stats-kind-get stats-after-second
                                           'chunk-state 'misses))
        (cons 'incremental-forward-hits
              (photon-cache-stats-kind-get stats-after-second
                                           'incremental-forward 'hits))
        (cons 'projection-puts
              (photon-cache-stats-kind-get stats-after-second
                                           'projection 'puts))
        (cons 'cache-limit (photon-model-cache-limit model))
        (cons 'cache-after-train cache-after-train)
        (cons 'history history)
        (cons 'generated (photon-model-generate model context 3))))
