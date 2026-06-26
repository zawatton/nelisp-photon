(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(photon-use-elisp-vector-backend)

(defun photon-benchmark-loss (result field)
  (cdr (assq field (cdr (assq 'loss result)))))

(defun photon-benchmark-cache-total (stats)
  (+ (photon-cache-stats-get stats 'hits)
     (photon-cache-stats-get stats 'misses)
     (photon-cache-stats-get stats 'puts)))

(defun photon-benchmark-fallback-prediction (model context)
  (let* ((states (photon-forward-model model context))
         (state (photon-prediction-state states context)))
    (photon-argmax-index (photon-model-readout-logits model state context))))

(defun photon-benchmark-readout-metrics (model tokens context-size)
  (let ((pairs (photon-training-pairs tokens context-size))
        (memory-hits 0)
        (memory-correct 0)
        (fallback-correct 0)
        (total 0))
    (while pairs
      (let* ((context (car (car pairs)))
             (target (cdr (car pairs)))
             (remembered (photon-model-next-token-memory-get model context))
             (fallback (photon-benchmark-fallback-prediction model context)))
        (when (photon-model-next-token-memory-hit-p model context)
          (setq memory-hits (1+ memory-hits))
          (when (= remembered target)
            (setq memory-correct (1+ memory-correct))))
        (when (= fallback target)
          (setq fallback-correct (1+ fallback-correct)))
        (setq total (1+ total)))
      (setq pairs (cdr pairs)))
    (list (cons 'memory-size (length (photon-model-next-token-memory model)))
          (cons 'memory-hits memory-hits)
          (cons 'memory-hit-permil
                (if (= total 0) 0 (/ (* 1000 memory-hits) total)))
          (cons 'memory-accuracy-permil
                (if (= memory-hits 0) 0 (/ (* 1000 memory-correct)
                                           memory-hits)))
          (cons 'fallback-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 fallback-correct) total))))))

(defun photon-benchmark-apply-weights (model weights)
  (photon-model-param-set model 'next-token-weight
                          (cdr (assq 'next-token-weight weights)))
  (photon-model-param-set model 'reconstruction-weight
                          (cdr (assq 'reconstruction-weight weights)))
  (photon-model-param-set model 'next-context-weight
                          (cdr (assq 'next-context-weight weights))))

(defun photon-benchmark-apply-readout (model readout-case)
  (photon-model-param-set model 'prototype-weight
                          (cdr (assq 'prototype-weight readout-case)))
  (photon-model-param-set model 'context-anchor-enabled
                          (cdr (assq 'context-anchor-enabled readout-case)))
  (photon-model-param-set model 'context-anchor-ngram-enabled
                          (cdr (assq 'context-anchor-ngram-enabled
                                     readout-case)))
  (photon-model-param-set model 'char-class-bias-enabled
                          (cdr (assq 'char-class-bias-enabled
                                     readout-case)))
  (photon-model-param-set model 'char-boundary-bias-enabled
                          (cdr (assq 'char-boundary-bias-enabled
                                     readout-case)))
  (photon-model-param-set model 'structural-boundary-bias-enabled
                          (cdr (assq 'structural-boundary-bias-enabled
                                     readout-case)))
  (photon-model-param-set model 'structural-tie-breaker-enabled
                          (cdr (assq 'structural-tie-breaker-enabled
                                     readout-case)))
  (photon-model-param-set model 'context-anchor-weight
                          (cdr (assq 'context-anchor-weight readout-case))))

(defun photon-benchmark-metrics (model eval-tokens context-size
                                       before after train-steps)
  (let* ((stats (photon-model-cache-stats model))
         (cache-total (photon-benchmark-cache-total stats)))
    (append
     (list (cons 'before-accuracy (cdr (assq 'accuracy-permil before)))
           (cons 'after-accuracy (cdr (assq 'accuracy-permil after)))
           (cons 'before-total-loss (photon-benchmark-loss before 'total-loss))
           (cons 'after-total-loss (photon-benchmark-loss after 'total-loss))
           (cons 'before-reconstruction-loss
                 (photon-benchmark-loss before 'reconstruction-loss-permil))
           (cons 'after-reconstruction-loss
                 (photon-benchmark-loss after 'reconstruction-loss-permil)))
     (photon-benchmark-readout-metrics model eval-tokens context-size)
     (list (cons 'cache-hits (photon-cache-stats-get stats 'hits))
           (cons 'cache-misses (photon-cache-stats-get stats 'misses))
           (cons 'cache-puts (photon-cache-stats-get stats 'puts))
           (cons 'cache-ops cache-total)
           (cons 'train-steps train-steps)
           (cons 'elapsed-ish-steps (+ train-steps cache-total))))))

(defun photon-benchmark-run-case (corpus-case loss-case cache-case
                                              memory-case readout-case)
  (let* ((train-tokens (cdr (assq 'train-tokens corpus-case)))
         (eval-tokens (cdr (assq 'eval-tokens corpus-case)))
         (context-size (cdr (assq 'context-size corpus-case)))
         (epochs (cdr (assq 'epochs corpus-case)))
         (config (photon-make-config 8 4 2 2))
         (model (photon-make-model config))
         (pairs (length (photon-training-pairs train-tokens context-size)))
         (train-steps (* epochs pairs)))
    (photon-benchmark-apply-weights model (cdr (assq 'weights loss-case)))
    (photon-benchmark-apply-readout model readout-case)
    (photon-model-param-set model 'next-token-memory-enabled
                            (cdr (assq 'enabled memory-case)))
    (photon-model-set-cache-limit model (cdr (assq 'limit cache-case)))
    (photon-model-reset-cache-stats model)
    (let* ((before (photon-evaluate model eval-tokens context-size))
           (history (photon-train model train-tokens context-size epochs 0.20))
           (after (photon-evaluate model eval-tokens context-size)))
      (append
       (list (cons 'corpus (cdr (assq 'name corpus-case)))
             (cons 'train-tokens train-tokens)
             (cons 'eval-tokens eval-tokens)
             (cons 'loss-mode (cdr (assq 'name loss-case)))
             (cons 'cache-mode (cdr (assq 'name cache-case)))
             (cons 'cache-limit (cdr (assq 'limit cache-case)))
             (cons 'memory-mode (cdr (assq 'name memory-case)))
             (cons 'readout-mode (cdr (assq 'name readout-case)))
             (cons 'history history))
       (photon-benchmark-metrics model eval-tokens context-size
                                 before after train-steps)))))

(defun photon-benchmark-run-all ()
  (let ((corpora
        (list
         (list (cons 'name 'alternating)
                (cons 'train-tokens '(1 2 1 2))
                (cons 'eval-tokens '(2 1 2 1))
                (cons 'context-size 2)
                (cons 'epochs 1))
         (list (cons 'name 'triad)
                (cons 'train-tokens '(1 2 3 1 2))
                (cons 'eval-tokens '(2 3 1 2 3))
                (cons 'context-size 3)
                (cons 'epochs 1))))
        (loss-cases
         (list
          (list (cons 'name 'next-token-only)
                (cons 'weights
                      (list (cons 'next-token-weight 1.0)
                            (cons 'reconstruction-weight 0.0)
                            (cons 'next-context-weight 0.0))))
          (list (cons 'name 'next-token+reconstruction)
                (cons 'weights
                      (list (cons 'next-token-weight 1.0)
                            (cons 'reconstruction-weight 0.10)
                            (cons 'next-context-weight 0.0))))
          (list (cons 'name 'all-losses)
                (cons 'weights
                      (list (cons 'next-token-weight 1.0)
                            (cons 'reconstruction-weight 0.10)
                            (cons 'next-context-weight 0.02))))))
        (cache-cases
         (list
          (list (cons 'name 'small-cache) (cons 'limit 8))
          (list (cons 'name 'default-cache) (cons 'limit 512))))
        (memory-cases
         (list
          (list (cons 'name 'memory-on) (cons 'enabled 1.0))
          (list (cons 'name 'memory-off) (cons 'enabled 0.0))))
        (readout-cases
         (list
          (list (cons 'name 'full-readout)
                (cons 'prototype-weight 0.75)
                (cons 'context-anchor-enabled 1.0)
                (cons 'context-anchor-ngram-enabled 1.0)
                (cons 'char-class-bias-enabled 1.0)
                (cons 'char-boundary-bias-enabled 1.0)
                (cons 'structural-boundary-bias-enabled 1.0)
                (cons 'structural-tie-breaker-enabled 1.0)
                (cons 'context-anchor-weight 0.0))
          (list (cons 'name 'prototype-off)
                (cons 'prototype-weight 0.0)
                (cons 'context-anchor-enabled 1.0)
                (cons 'context-anchor-ngram-enabled 1.0)
                (cons 'char-class-bias-enabled 1.0)
                (cons 'char-boundary-bias-enabled 1.0)
                (cons 'structural-boundary-bias-enabled 1.0)
                (cons 'structural-tie-breaker-enabled 1.0)
                (cons 'context-anchor-weight 0.0))
          (list (cons 'name 'structural-off)
                (cons 'prototype-weight 0.75)
                (cons 'context-anchor-enabled 1.0)
                (cons 'context-anchor-ngram-enabled 1.0)
                (cons 'char-class-bias-enabled 0.0)
                (cons 'char-boundary-bias-enabled 0.0)
                (cons 'structural-boundary-bias-enabled 0.0)
                (cons 'structural-tie-breaker-enabled 0.0)
                (cons 'context-anchor-weight 0.0))
          (list (cons 'name 'output-head-only)
                (cons 'prototype-weight 0.0)
                (cons 'context-anchor-enabled 0.0)
                (cons 'context-anchor-ngram-enabled 0.0)
                (cons 'char-class-bias-enabled 0.0)
                (cons 'char-boundary-bias-enabled 0.0)
                (cons 'structural-boundary-bias-enabled 0.0)
                (cons 'structural-tie-breaker-enabled 0.0)
                (cons 'context-anchor-weight 0.0))))
        (results nil))
    (while corpora
      (let ((loss-walk loss-cases))
        (while loss-walk
          (let ((cache-walk cache-cases))
            (while cache-walk
              (let ((memory-walk memory-cases))
                (while memory-walk
                  (let ((readout-walk readout-cases))
                    (while readout-walk
                      (setq results
                            (cons (photon-benchmark-run-case
                                   (car corpora) (car loss-walk)
                                   (car cache-walk) (car memory-walk)
                                   (car readout-walk))
                                  results))
                      (setq readout-walk (cdr readout-walk))))
                  (setq memory-walk (cdr memory-walk))))
              (setq cache-walk (cdr cache-walk))))
          (setq loss-walk (cdr loss-walk))))
      (setq corpora (cdr corpora)))
    (nreverse results)))

(defun photon-benchmark-state-records (model pairs)
  (let ((records nil))
    (while pairs
      (let* ((context (car (car pairs)))
             (states (photon-forward-model model context))
             (state (photon-prediction-state states context)))
        (setq records
              (cons (list (cons 'context context)
                          (cons 'target (cdr (car pairs)))
                          (cons 'state state))
                    records)))
      (setq pairs (cdr pairs)))
    (nreverse records)))

(defun photon-benchmark-nearest-state-distance (state records)
  (let ((best nil))
    (while records
      (let ((distance (photon-context-state-distance
                       state (cdr (assq 'state (car records))))))
        (when (or (null best) (< distance (cdr (assq 'distance best))))
          (setq best
                (list (cons 'context (cdr (assq 'context (car records))))
                      (cons 'target (cdr (assq 'target (car records))))
                      (cons 'distance distance)))))
      (setq records (cdr records)))
    best))

(defun photon-benchmark-token-vocab (model)
  (mapcar 'number-to-string
          (photon-range
           (photon-config-get (photon-model-config model) 'vocab-size))))

(defun photon-benchmark-pair-diagnostics (model train-records eval-pair)
  (let* ((context (car eval-pair))
         (target (cdr eval-pair))
         (states (photon-forward-model model context))
         (state (photon-prediction-state states context))
         (components (photon-model-readout-components model state context))
         (base-logits (cdr (assq 'token-readout components)))
         (raw-anchor-logits (photon-model-context-anchor-logits model context))
         (readout-logits (photon-model-readout-logits model state context))
         (prediction (photon-argmax-index readout-logits))
         (rival (photon-best-non-target-index readout-logits target))
         (raw-anchor-rival
          (photon-best-non-target-index raw-anchor-logits target))
         (vocab (photon-benchmark-token-vocab model)))
    (list (cons 'context context)
          (cons 'target target)
          (cons 'prediction prediction)
          (cons 'rival rival)
          (cons 'target-margin (photon-target-margin readout-logits
                                                     target rival))
          (cons 'base-prediction (photon-argmax-index base-logits))
          (cons 'raw-anchor-prediction
                (photon-argmax-index raw-anchor-logits))
          (cons 'raw-anchor-margin
                (photon-target-margin raw-anchor-logits
                                      target raw-anchor-rival))
          (cons 'readout-logits readout-logits)
          (cons 'component-contributions
                (photon-readout-contribution-debug
                 vocab components target prediction))
          (cons 'nearest-train-state
                (photon-benchmark-nearest-state-distance
                 state train-records)))))

(defun photon-benchmark-heldout-triad-diagnostics ()
  (let* ((train-tokens '(1 2 3 1 2))
         (eval-tokens '(2 3 1 2 3))
         (context-size 3)
         (config (photon-make-config 8 4 2 2))
         (model (photon-make-model config))
         (train-pairs (photon-training-pairs train-tokens context-size))
         (eval-pairs (photon-training-pairs eval-tokens context-size))
         (diagnostics nil))
    (photon-model-param-set model 'next-token-memory-enabled 0.0)
    (photon-model-param-set model 'prototype-weight 0.0)
    (photon-model-param-set model 'context-anchor-enabled 0.0)
    (photon-model-param-set model 'context-anchor-ngram-enabled 0.0)
    (photon-model-param-set model 'char-class-bias-enabled 0.0)
    (photon-model-param-set model 'char-boundary-bias-enabled 0.0)
    (photon-model-param-set model 'structural-boundary-bias-enabled 0.0)
    (photon-model-param-set model 'structural-tie-breaker-enabled 0.0)
    (photon-train model train-tokens context-size 1 0.20)
    (let ((train-records (photon-benchmark-state-records model train-pairs)))
      (while eval-pairs
        (setq diagnostics
              (cons (photon-benchmark-pair-diagnostics
                     model train-records (car eval-pairs))
                    diagnostics))
        (setq eval-pairs (cdr eval-pairs))))
    (list (cons 'case 'held-out-triad-output-head-only)
          (cons 'diagnostics (nreverse diagnostics)))))

(let ((result (photon-benchmark-run-all)))
  (prin1 (list (cons 'results result)
               (cons 'diagnostics
                     (list (photon-benchmark-heldout-triad-diagnostics)))))
  (princ "\n"))
