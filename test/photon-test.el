;;; photon-test.el --- Tests for photon.el -*- lexical-binding: t; -*-

(require 'ert)
(add-to-list 'load-path "lisp")
(require 'photon)

(ert-deftest photon-chunk-tokens-preserves-order ()
  (should (equal (photon-chunk-tokens '(1 2 3 4 5) 2)
                 '((1 2) (3 4) (5)))))

(ert-deftest photon-forward-has-bounded-local-shape ()
  (let* ((config (photon-make-config 12 6 3 2))
         (states (photon-forward config '(1 2 3 4 5))))
    ;; Two chunks at chunk-size 3 decode into six local positions.  The final
    ;; partial chunk is padded only inside the bounded local decoder.
    (should (= (length states) 6))
    (should (= (length (car states)) 6))))

(ert-deftest photon-prediction-state-uses-last-real-token ()
  (let* ((config (photon-make-config 12 6 2 2))
         (tokens '(1 2 3))
         (states (photon-forward config tokens)))
    (should (eq (photon-prediction-state states tokens)
                (nth 2 states)))
    (should-not (eq (photon-prediction-state states tokens)
                    (car (last states))))))

(ert-deftest photon-generation-is-deterministic ()
  (let* ((config (photon-make-config 16 8 4 2))
         (left (photon-generate config '(1 3 5) 3))
         (right (photon-generate config '(1 3 5) 3)))
    (should (equal left right))
    (should (= (length left) 6))))

(ert-deftest photon-training-pairs-use-fixed-context ()
  (should (equal (photon-training-pairs '(1 2 3 4 5) 3)
                 '(((1 2 3) . 4)
                   ((2 3 4) . 5)))))

(ert-deftest photon-training-pairs-handle-short-input ()
  (should (equal (photon-training-pairs '(1 2) 3) nil)))

(ert-deftest photon-vector-backend-defaults-to-list ()
  (let ((photon-vector-backend (photon-make-list-vector-backend)))
    (should (eq (photon-vector-backend-name) 'list-vector))
    (should (equal (photon-vector-to-list
                    (photon-vector-add (photon-vector-from-list '(1.0 2.0))
                                       (photon-vector-from-list '(3.0 4.0))))
                   '(4.0 6.0)))
    (should (= (photon-vector-dot (photon-vector-from-list '(1.0 2.0))
                                  (photon-vector-from-list '(3.0 4.0)))
               11.0))))

(ert-deftest photon-vector-backend-supports-elementwise-mul ()
  (let ((photon-vector-backend (photon-make-elisp-vector-backend)))
    (should (equal (photon-vector-to-list
                    (photon-vector-mul (photon-vector-from-list '(2.0 3.0))
                                       (photon-vector-from-list '(4.0 5.0))))
                   '(8.0 15.0)))))

(ert-deftest photon-vector-backend-selects-by-name ()
  (let ((photon-vector-backend (photon-make-list-vector-backend)))
    (photon-use-vector-backend-name "elisp-vector")
    (should (eq (photon-vector-backend-name) 'elisp-vector))
    (photon-use-vector-backend-name 'elisp-tensor)
    (should (eq (photon-vector-backend-name) 'elisp-tensor))
    (photon-use-vector-backend-name "list-vector")
    (should (eq (photon-vector-backend-name) 'list-vector))
    (should-error (photon-use-vector-backend-name "missing-backend"))))

(ert-deftest photon-vector-library-supports-matrix-vector-dot ()
  (let ((photon-vector-backend (photon-make-elisp-vector-backend)))
    (let ((matrix (photon-matrix-from-rows
                   (list (photon-vector-from-list '(1.0 2.0))
                         (photon-vector-from-list '(3.0 4.0)))))
          (vec (photon-vector-from-list '(10.0 20.0))))
      (should (equal (photon-matrix-shape matrix) '(2 2)))
      (should (equal (photon-vector-to-list
                      (photon-matrix-vector-dot matrix vec))
                     '(50.0 110.0))))))

(ert-deftest photon-vector-library-reports-tensor-shape ()
  (let ((photon-vector-backend (photon-make-elisp-vector-backend)))
    (should (photon-vector-backend-supports-p 'tensor))
    (should (equal (photon-tensor-shape
                    (list (photon-vector-from-list '(1.0 2.0))
                          (photon-vector-from-list '(3.0 4.0))))
                   '(2 2)))))

(ert-deftest photon-elisp-tensor-backend-supports-matrix-and-tensor-ops ()
  (let ((photon-vector-backend (photon-make-elisp-tensor-backend)))
    (should (eq (photon-vector-backend-name) 'elisp-tensor))
    (should (photon-vector-backend-supports-p 'tensor))
    (let* ((matrix (photon-matrix-from-rows
                    (list (photon-vector-from-list '(1.0 2.0))
                          (photon-vector-from-list '(3.0 4.0)))))
           (vec (photon-vector-from-list '(10.0 20.0)))
           (tensor (photon-tensor-zeros '(2 2)))
           (scaled (photon-tensor-scale 2.0
                                        (photon-tensor-add tensor tensor))))
      (should (vectorp matrix))
      (should (equal (photon-matrix-shape matrix) '(2 2)))
      (should (equal (photon-vector-to-list
                      (photon-matrix-vector-dot matrix vec))
                     '(50.0 110.0)))
      (should (equal (photon-tensor-shape scaled) '(2 2))))))

(defun photon-test-external-matrix-shape (_matrix)
  '(external 2 3))

(defun photon-test-external-tensor-shape (_tensor)
  '(external-tensor 2 3 4))

(ert-deftest photon-vector-external-backend-can-override-shape-ops ()
  (let ((photon-vector-backend
         (photon-make-external-vector-backend
          'external-test
          '(vector matrix tensor)
          (list (cons 'add 'photon-list-vector-add)
                (cons 'mul 'photon-list-vector-mul)
                (cons 'matvec 'photon-list-matrix-vector-dot)
                (cons 'scale 'photon-list-vector-scale)
                (cons 'dot 'photon-list-vector-dot)
                (cons 'squash 'photon-list-vector-squash)
                (cons 'from-list 'photon-list-vector-from-list)
                (cons 'to-list 'photon-list-vector-to-list)
                (cons 'zeros 'photon-list-vector-zeros)
                (cons 'matrix-shape
                      'photon-test-external-matrix-shape)
                (cons 'tensor-shape
                      'photon-test-external-tensor-shape)))))
    (should (eq (photon-vector-backend-name) 'external-test))
    (should (equal (photon-matrix-shape 'ignored) '(external 2 3)))
    (should (equal (photon-tensor-shape 'ignored)
                   '(external-tensor 2 3 4)))))

(ert-deftest photon-vector-fingerprint-is-stable-and-compact ()
  (let ((photon-vector-backend (photon-make-elisp-vector-backend)))
    (let ((left (photon-vector-from-list '(1.0 2.0 3.0 4.0)))
          (right (photon-vector-from-list '(1.0 2.0 3.0 5.0))))
      (should (equal (photon-vector-fingerprint left)
                     (photon-vector-fingerprint left)))
      (should-not (equal (photon-vector-fingerprint left)
                         (photon-vector-to-list left)))
      (should-not (equal (photon-vector-fingerprint left)
                         (photon-vector-fingerprint right))))))

(ert-deftest photon-vector-backend-supports-elisp-vector ()
  (let ((photon-vector-backend (photon-make-elisp-vector-backend)))
    (should (eq (photon-vector-backend-name) 'elisp-vector))
    (should (photon-vector-backend-supports-p 'vector))
    (should (photon-vector-backend-supports-p 'matrix))
    (should (photon-vector-backend-supports-p 'tensor))
    (should (photon-vector-backend-supports-p 'elementwise))
    (should (photon-vector-backend-supports-p 'fingerprint))
    (should (equal (photon-vector-to-list
                    (photon-vector-add (photon-vector-from-list '(1.0 2.0))
                                       (photon-vector-from-list '(3.0 4.0))))
                   '(4.0 6.0)))
    (should (= (photon-vector-dot (photon-vector-from-list '(1.0 2.0))
                                  (photon-vector-from-list '(3.0 4.0)))
               11.0))))

(ert-deftest photon-stream-training-matches-pair-evaluation-counts ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model-a (photon-make-model config))
         (model-b (photon-make-model config))
         (tokens '(1 2 3 4 1 2 3 4)))
    (should (equal (photon-train-epoch model-a
                                       (photon-training-pairs tokens 4)
                                       0.0)
                   (photon-train-epoch-stream model-b tokens 4 0.0)))))

(ert-deftest photon-minibatch-training-improves-toy-corpus ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4))
         (before (cdr (assq 'correct (photon-evaluate model corpus 4)))))
    (photon-train-minibatch model corpus 4 4 0.35 4)
    (let ((after (cdr (assq 'correct (photon-evaluate model corpus 4)))))
      (should (>= after before)))))

(ert-deftest photon-model-has-trainable-embedding-table ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (embedding (nth 1 (photon-model-embedding-table model))))
    (should (= (length (photon-model-embedding-table model)) 8))
    (should (= (length embedding) 8))))

(ert-deftest photon-model-has-target-prototype-memory ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config)))
    (should (= (length (photon-model-target-prototypes model)) 8))
    (should (= (length (car (photon-model-target-prototypes model))) 8))
    (should (= (length (photon-model-target-counts model)) 8))
    (should-not (photon-model-target-state-clusters model))
    (should (photon-model-next-token-memory-enabled-p model))
    (should (= (photon-model-next-token-memory-limit model) 512))
    (photon-model-remember-next-token model '(1 2 3 4) 5)
    (should (photon-model-next-token-memory-hit-p model '(1 2 3 4)))
    (should (= (photon-model-next-token-memory-get model '(1 2 3 4)) 5))
    (photon-model-param-set model 'next-token-memory-enabled 0.0)
    (should-not (photon-model-next-token-memory-enabled-p model))
    (should-not (photon-model-next-token-memory-hit-p model '(1 2 3 4)))
    (should-not (photon-model-next-token-memory-get model '(1 2 3 4)))))

(ert-deftest photon-training-rebuilds-target-state-clusters ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4)))
    (photon-train model corpus 4 2 0.20)
    (let ((clusters (photon-model-target-state-clusters model)))
      (should (consp clusters))
      (should (assq 1 clusters))
      (should (<= (length (cdr (assq 1 clusters)))
                  (photon-model-target-state-cluster-limit model))))))

(ert-deftest photon-context-anchor-weight-learns-from-context ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 4 2 2))
         (model (photon-make-model config)))
    (photon-model-param-set model 'next-token-memory-enabled 0.0)
    (should (= (photon-model-param-get model 'context-anchor-weight) 0.0))
    (photon-train model '(1 2 3 1 2) 3 1 0.20)
    (should (> (photon-model-param-get model 'context-anchor-weight) 0.0))
    (should (= (cdr (assq 'accuracy-permil
                          (photon-evaluate model '(2 3 1 2 3) 3)))
               1000))))

(ert-deftest photon-training-updates-embedding-table ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4))
         (before (photon-vector-to-list
                  (nth 1 (photon-model-embedding-table model)))))
    (photon-train model corpus 4 2 0.20)
    (let ((after (photon-vector-to-list
                  (nth 1 (photon-model-embedding-table model)))))
      (should-not (equal before after)))))

(ert-deftest photon-model-has-architecture-params ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config)))
    (should (= (photon-model-param-get model 'chunk-scale) 1.0))
    (should (= (photon-model-param-get model 'context-current) 0.72))
    (should (= (photon-model-param-get model 'decoder-prev) 0.35))
    (should (= (photon-model-param-get model 'decoder-token-mix) 1.0))
    (should (= (photon-model-param-get model 'context-anchor-recency) 0.70))
    (should (= (photon-model-param-get model 'context-anchor-prefix-weight) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-context-anchor-ngram-weight) 6.0))
    (should (= (photon-model-param-get model 'wordpiece-final-context-anchor-ngram-weight) 16.0))
    (should (= (photon-model-param-get model 'readout-logit-temperature) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-final-readout-temperature) 8.0))
    (should (= (photon-model-param-get model 'wordpiece-long-token-update-weight) 0.35))
    (should (= (photon-model-param-get model 'wordpiece-rare-token-update-weight) 0.50))
    (should (= (photon-model-param-get model 'wordpiece-update-scale-max) 2.50))
    (should (= (photon-model-param-get model 'wordpiece-readout-bias-enabled) 0.0))
    (should (= (photon-model-param-get model 'wordpiece-long-readout-bias-weight) 0.75))
    (should (= (photon-model-param-get model 'wordpiece-rare-readout-bias-weight) 0.50))
    (should (= (photon-model-param-get model 'wordpiece-token-readout-bias-weight) 0.35))
    (should (= (photon-model-param-get model 'wordpiece-readout-bias-max) 2.50))
    (should (= (photon-model-param-get model 'wordpiece-ngram-rescue-weight) 1.00))
    (should (= (photon-model-param-get model 'wordpiece-ngram-rescue-epochs) 4.0))
    (should (= (photon-model-param-get model 'wordpiece-final-ngram-rescue-weight) 1.00))
    (should (= (photon-model-param-get model 'wordpiece-final-ngram-rescue-epochs) 3.0))
    (should (= (photon-model-param-get model 'wordpiece-long-rare-readout-rescue-weight) 0.75))
    (should (= (photon-model-param-get model 'wordpiece-long-rare-readout-rescue-epochs) 2.0))
    (should (= (photon-model-param-get
                model 'wordpiece-long-teacher-state-path-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-continuation-teacher-state-path-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-teacher-state-path-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-teacher-token-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-teacher-context-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-teacher-anchor-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-boundary-readout-enabled)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-boundary-readout-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-boundary-readout-size)
               4.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-boundary-ngram-mix)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-full-readout-distill-start-long-boundary-teacher-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-classifier-enabled)
               0.0))
    (should (= (photon-model-param-get
                model 'wordpiece-start-long-classifier-learning-rate-scale)
               1.0))
    (should (= (photon-model-param-get
                model
                'wordpiece-start-long-classifier-candidate-search-enabled)
               0.0))
    (should (= (photon-model-param-get
                model
                'wordpiece-start-long-classifier-candidate-learning-rate)
               0.20))
    (should (= (photon-model-param-get
                model
                'wordpiece-start-long-classifier-candidate-top-k)
               10.0))
    (should (= (photon-model-param-get
                model
                'wordpiece-start-long-classifier-candidate-margin)
               0.10))
    (should (= (photon-model-param-get
                model
                'wordpiece-start-long-classifier-candidate-limit)
               64.0))
    (should (= (photon-model-param-get
                model
                'output-head-full-readout-distill-start-long-classifier-teacher-weight)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-full-readout-distill-start-long-ngram-teacher-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-start-long-positive-distill-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-start-long-positive-distill-epochs)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-start-long-positive-distill-top-k)
               3.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-projected-prototype-distill-weight)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-projected-prototype-distill-epochs)
               1.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-projected-prototype-distill-top-k)
               1.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-prototype-merge-candidate-search-enabled)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-prototype-merge-candidate-limit)
               64.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-rival-negative-distill-weight)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-rival-negative-distill-epochs)
               1.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-rival-negative-distill-top-k)
               3.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-rival-negative-distill-rivals)
               1.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-weight)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-bias-weight)
               0.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-epochs)
               1.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-top-k)
               3.0))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-margin)
               0.10))
    (should (= (photon-model-param-get
                model
                'output-head-start-long-same-shape-margin-distill-adaptive-bias-max)
               0.0))
    (should (= (photon-model-param-get model 'context-anchor-ngram-contrastive-weight) 0.25))
    (should (= (photon-model-param-get model 'context-anchor-weight-max) 1.0))
    (should (= (photon-model-param-get model 'char-class-bias-weight) 0.75))
    (should (= (photon-model-param-get model 'char-boundary-bias-weight) 1.0))
    (should (= (photon-model-param-get model 'structural-boundary-bias-weight) 0.10))
    (should (= (photon-model-param-get model 'structural-bias-letter-gate-scale) 0.0))
    (should (= (photon-model-param-get model 'structural-bias-space-gate-scale) 0.75))
    (should (= (photon-model-param-get model 'structural-tie-breaker-weight) 2.0))
    (should (= (photon-model-param-get model 'newline-boundary-bonus) 4.20))
    (should (= (photon-model-param-get model 'generation-repeat-bias-weight) 1.25))
    (should (= (photon-model-param-get model 'generation-newline-run-bias-weight) 3.0))
    (should (= (photon-model-param-get model 'generation-space-run-bias-weight) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-repeat-bias-weight) 2.50))
    (should (= (photon-model-param-get model 'wordpiece-generation-continuation-run-bias-weight) 1.50))
    (should (= (photon-model-param-get model 'wordpiece-generation-recent-token-bias-weight) 0.40))
    (should (= (photon-model-param-get model 'wordpiece-generation-recent-token-window) 8.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-ngram-bias-weight) 0.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-ngram-bias-gate-margin) -1.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-ngram-bias-repeat-scale) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-ngram-bias-repeat-window) 24.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-repeat-ngram-bias-weight) 0.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-repeat-ngram-bias-min-count) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-generation-repeat-ngram-window) 24.0))
    (should (= (photon-model-param-get model 'output-head-aux-update-weight) 0.50))
    (should (= (photon-model-param-get model 'output-head-anchor-distill-weight) 2.50))
    (should (= (photon-model-param-get model 'output-head-cluster-distill-weight) 1.00))
    (should (= (photon-model-param-get model 'output-head-cluster-distill-epochs) 5.0))
    (should (= (photon-model-param-get model 'output-head-prototype-weight) 8.0))
    (should (= (photon-model-param-get model 'output-head-prototype-limit) 32.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-support-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-support-limit)
               3.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-negative-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-negative-limit)
               32.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-long-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-short-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-special-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-start-long-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-continuation-long-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-start-short-piece-weight)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-continuation-short-piece-weight)
               1.0))
    (should (= (photon-model-param-get model 'output-head-prototype-diverse-trim-enabled) 0.0))
    (should (= (photon-model-param-get model 'output-head-prototype-projection-enabled) 1.0))
    (should (= (photon-model-param-get model 'wordpiece-final-output-head-prototype-projection-enabled) 0.0))
    (should (= (photon-model-param-get model 'output-head-linear-readout-weight) 0.0))
    (should (= (photon-model-param-get model 'wordpiece-final-output-head-linear-readout-weight) 1.0))
    (should (= (photon-model-param-get model 'output-head-compress-weight) 0.40))
    (should (= (photon-model-param-get model 'output-head-compress-epochs) 12.0))
    (should (= (photon-model-param-get model 'output-head-token-readout-distill-weight) 0.50))
    (should (= (photon-model-param-get model 'output-head-token-readout-distill-epochs) 3.0))
    (should (= (photon-model-param-get model 'output-head-full-readout-distill-weight) 2.00))
    (should (= (photon-model-param-get model 'output-head-full-readout-distill-epochs) 8.0))
    (should (= (photon-model-param-get model 'output-head-full-readout-distill-wordpiece-scale-enabled) 0.0))
    (should (= (photon-model-param-get
                model 'output-head-full-readout-distill-start-long-scale)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-full-readout-distill-continuation-long-scale)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-full-readout-distill-start-short-scale)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-full-readout-distill-continuation-short-scale)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-full-readout-distill-selection-start-long-weight)
               0.0))
    (should (= (photon-model-param-get model 'output-head-prototype-readout-distill-epochs) 1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-distill-epochs)
               1.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-bias-weight)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-token-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-context-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-anchor-mix)
               0.0))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-contrastive-rate)
               0.02))
    (should (= (photon-model-param-get
                model 'output-head-prototype-ngram-long-state-path-rate)
               0.0))
    (should (= (photon-model-param-get model 'output-head-ngram-distill-weight) 0.50))
    (should (= (photon-model-param-get model 'prototype-ngram-distill-weight) 1.00))
    (should (= (photon-model-param-get model 'output-head-state-cluster-readout-weight) 1.0))
    (should (= (photon-model-param-get model 'state-cluster-readout-weight) 1.0))
    (should (= (photon-model-param-get model 'state-cluster-limit) 64.0))
    (should (= (photon-model-param-get model 'next-token-weight) 1.0))
    (should (= (photon-model-param-get model 'reconstruction-weight) 0.05))
    (should (= (photon-model-param-get model 'next-context-weight) 0.02))))

(ert-deftest photon-training-updates-architecture-params ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4))
         (before (copy-tree (photon-model-architecture-params model))))
    (photon-train model corpus 4 2 0.20)
    (should-not (equal before (photon-model-architecture-params model)))))

(ert-deftest photon-model-has-layer-gain-vectors ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (gain (photon-model-gain-get model 'chunk-gain)))
    (should (= (length (photon-model-layer-gains model)) 5))
    (should (= (length gain) 8))))

(ert-deftest photon-model-has-layer-projection-matrices ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (projection (photon-model-projection-get model 'chunk-proj)))
    (should (= (length (photon-model-layer-projections model)) 4))
    (should (equal (photon-projection-shape projection) '(8 8)))
    (should (eq (cdr (assq 'kind projection)) 'low-rank))))

(ert-deftest photon-model-has-output-head-projection ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (projection (photon-model-output-head-projection model)))
    (should (equal (photon-projection-shape projection) '(8 8)))
    (should (eq (cdr (assq 'kind projection)) 'low-rank))))

(ert-deftest photon-training-updates-layer-gain-vectors ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4))
         (before (photon-vector-to-list
                  (photon-model-gain-get model 'chunk-gain))))
    (photon-train model corpus 4 2 0.20)
    (let ((after (photon-vector-to-list
                  (photon-model-gain-get model 'chunk-gain))))
      (should-not (equal before after)))))

(ert-deftest photon-training-updates-layer-projections ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4))
         (before (copy-tree (photon-model-projection-get model 'chunk-proj))))
    (photon-train model corpus 4 2 0.20)
    (should-not (equal before (photon-model-projection-get model 'chunk-proj)))))

(ert-deftest photon-model-project-caches-results ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (vec (photon-vector-from-list '(1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))))
    (photon-model-project model 'chunk-proj vec)
    (should (= (length (photon-model-cache model)) 1))
    (photon-model-project model 'chunk-proj vec)
    (should (= (length (photon-model-cache model)) 1))
    (photon-update-layer-projections model 1 2 0.1)
    (should (= (length (photon-model-cache model)) 0))))

(ert-deftest photon-model-token-embedding-caches-by-token ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (key (photon-token-cache-key 1)))
    (photon-model-clear-cache model)
    (photon-model-token-embedding model 1)
    (should (photon-model-cache-get model key))
    (should (= (length (photon-model-cache model)) 1))
    (photon-model-token-embedding model 1)
    (should (= (length (photon-model-cache model)) 1))
    (photon-update-embedding-table model '(1) 1 2 0.1)
    (should (= (length (photon-model-cache model)) 0))))

(ert-deftest photon-forward-model-caches-by-window ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (context '(1 2 3 4))
         (key (photon-window-cache-key context)))
    (photon-model-clear-cache model)
    (photon-forward-model model context)
    (should (photon-model-cache-get model key))
    (let ((cache-size (length (photon-model-cache model))))
      (photon-forward-model model context)
      (should (= (length (photon-model-cache model)) cache-size)))
    (photon-model-param-set model 'chunk-scale 1.1)
    (should (= (length (photon-model-cache model)) 0))))

(ert-deftest photon-predict-one-caches-by-window ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (context '(1 2 3 4))
         (key (photon-prediction-cache-key context)))
    (photon-model-clear-cache model)
    (photon-predict-one model context)
    (should (photon-model-cache-get model key))
    (let ((cache-size (length (photon-model-cache model))))
      (photon-predict-one model context)
      (should (= (length (photon-model-cache model)) cache-size)))
    (photon-update-output-head model
                               (cdr (assq 'state (photon-predict-one model context)))
                               1 2 0.1)
    (should (= (length (photon-model-cache model)) 0))))

(ert-deftest photon-cache-stats-track-hit-miss-by-kind ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (context '(1 2 3 4)))
    (photon-model-clear-cache model)
    (photon-model-reset-cache-stats model)
    (photon-predict-one model context)
    (photon-predict-one model context)
    (let ((stats (photon-model-cache-stats model)))
      (should (> (photon-cache-stats-get stats 'hits) 0))
      (should (> (photon-cache-stats-get stats 'misses) 0))
      (should (= (photon-cache-stats-kind-get stats 'prediction-window 'misses) 1))
      (should (= (photon-cache-stats-kind-get stats 'prediction-window 'hits) 1))
      (should (> (photon-cache-stats-kind-get stats 'projection 'puts) 0)))
    (photon-model-clear-cache model)
    (should (= (photon-cache-stats-get (photon-model-cache-stats model)
                                       'clears)
               1))))

(ert-deftest photon-model-chunker-caches-chunk-state ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (chunk '(1 2 3 4)))
    (photon-model-clear-cache model)
    (photon-model-reset-cache-stats model)
    (photon-model-chunker model chunk)
    (photon-model-chunker model chunk)
    (let ((stats (photon-model-cache-stats model)))
      (should (= (photon-cache-stats-kind-get stats 'chunk-state 'misses) 1))
      (should (= (photon-cache-stats-kind-get stats 'chunk-state 'hits) 1)))))

(ert-deftest photon-forward-model-reuses-incremental-prefix ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 3 2))
         (incremental-model (photon-make-model config))
         (full-model (photon-make-model config))
         (short '(1 2 3))
         (extended '(1 2 3 4)))
    (photon-forward-model incremental-model short)
    (let ((incremental-states (photon-forward-model incremental-model extended))
          (full-states (photon-forward-model full-model extended))
          (last-info (photon-model-cache-get
                      incremental-model
                      (photon-incremental-forward-cache-key))))
      (should (equal incremental-states full-states))
      (should (cdr (assq 'incremental last-info)))
      (should (= (cdr (assq 'reused-chunks last-info)) 1)))))

(ert-deftest photon-model-cache-limit-trims-old-entries ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 16 8 4 2))
         (model (photon-make-model config))
         (i 0))
    (photon-model-set-cache-limit model 5)
    (while (< i 10)
      (photon-model-token-embedding model i)
      (setq i (1+ i)))
    (should (<= (length (photon-model-cache model)) 5))))

(ert-deftest photon-loss-report-separates-training-signals ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (report (photon-loss-report model '(1 2 3 4) 1)))
    (should (assq 'next-token-loss report))
    (should (assq 'memory-hit report))
    (should (assq 'reconstruction-loss report))
    (should (assq 'next-context-loss report))
    (should (>= (cdr (assq 'next-token-loss report)) 0))
    (should (>= (cdr (assq 'reconstruction-loss report)) 0))
    (should (>= (cdr (assq 'next-context-loss report)) 0.0))))

(ert-deftest photon-next-context-loss-counts-adjacent-chunks ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 2 2))
         (model (photon-make-model config))
         (report (photon-loss-report model '(1 2 3 4) 1)))
    (should (= (cdr (assq 'next-context-total report)) 1))
    (should (>= (cdr (assq 'next-context-loss report)) 0))))

(ert-deftest photon-evaluate-returns-loss-breakdown ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4))
         (result (photon-evaluate model corpus 4))
         (loss (cdr (assq 'loss result))))
    (should (= (cdr (assq 'total result)) 4))
    (should (assq 'next-token-loss loss))
    (should (assq 'reconstruction-loss-permil loss))
    (should (assq 'next-context-loss-permil loss))))

(ert-deftest photon-loss-breakdown-includes-weighted-total ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (result (photon-evaluate model '(1 2 3 4 1 2 3 4) 4))
         (loss (cdr (assq 'loss result))))
    (should (assq 'weighted-next-token-loss loss))
    (should (assq 'weighted-reconstruction-loss loss))
    (should (assq 'weighted-next-context-loss loss))
    (should (assq 'total-loss loss))
    (should (= (cdr (assq 'total-loss loss))
               (+ (cdr (assq 'weighted-next-token-loss loss))
                  (cdr (assq 'weighted-reconstruction-loss loss))
                  (cdr (assq 'weighted-next-context-loss loss)))))))

(ert-deftest photon-reconstruction-update-changes-reconstruction-head ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (zero-head nil)
         (i 0))
    (while (< i 8)
      (setq zero-head (cons (photon-vector-zeros 8) zero-head))
      (setq i (1+ i)))
    (photon-model-set-reconstruction-head model (nreverse zero-head))
    (let* ((report (photon-loss-report model '(1 2 3 4) 1))
           (before-output-head (copy-tree (photon-model-output-head model)))
           (before-head (copy-tree (photon-model-reconstruction-head model)))
           (before-mix (photon-model-param-get model 'decoder-token-mix))
           (before-gain (photon-vector-to-list
                         (photon-model-gain-get model 'decoder-prev-gain))))
      (photon-update-reconstruction-from-report model '(1 2 3 4) report 0.20)
      (should (equal before-output-head (photon-model-output-head model)))
      (should-not (equal before-head (photon-model-reconstruction-head model)))
      (should-not (= before-mix (photon-model-param-get model 'decoder-token-mix)))
      (should-not (equal before-gain
                         (photon-vector-to-list
                          (photon-model-gain-get model 'decoder-prev-gain)))))))

(ert-deftest photon-next-context-update-changes-context-parameters ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (config (photon-make-config 8 8 2 2))
         (model (photon-make-model config))
         (report (photon-loss-report model '(1 2 3 4) 1))
         (before-gain (photon-vector-to-list
                       (photon-model-gain-get model 'context-current-gain)))
         (before-proj (copy-tree (photon-model-projection-get model
                                                              'context-proj))))
    (photon-update-next-context-from-report model report 0.20)
    (should-not (equal before-gain
                       (photon-vector-to-list
                        (photon-model-gain-get model 'context-current-gain))))
    (should-not (equal before-proj
                       (photon-model-projection-get model 'context-proj)))))

(ert-deftest photon-model-training-improves-toy-corpus ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (corpus '(1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4))
         (before (cdr (assq 'correct (photon-evaluate model corpus 4)))))
    (photon-train model corpus 4 10 0.20)
    (let ((after (cdr (assq 'correct (photon-evaluate model corpus 4)))))
      (should (>= after before)))))

(ert-deftest photon-char-vocab-encodes-and-decodes-text ()
  (let* ((vocab (photon-build-char-vocab "abac"))
         (tokens (photon-encode-text vocab "cab")))
    (should (equal vocab '("a" "b" "c")))
    (should (equal (photon-decode-tokens vocab tokens) "cab"))))

(ert-deftest photon-wordpiece-tokenizer-round-trips-text ()
  (let* ((text "photon stream\nnext photon")
         (vocab (photon-build-wordpiece-vocab text 3))
         (tokens (photon-encode-wordpiece-text vocab text 3)))
    (should (equal (photon-take vocab 4)
                   (photon-wordpiece-special-tokens)))
    (should (member "pho" vocab))
    (should (member "##ton" vocab))
    (should (< (length tokens) (length text)))
    (should (equal (photon-decode-wordpiece-tokens vocab tokens) text))))

(ert-deftest photon-wordpiece-tokenizer-handles-specials-and-unknown ()
  (let* ((vocab (photon-build-wordpiece-vocab "known" 3))
         (tokens (photon-encode-wordpiece-text vocab "zzz" 3 t)))
    (should (= (car tokens) (photon-vocab-index vocab "[BOS]")))
    (should (= (car (last tokens)) (photon-vocab-index vocab "[EOS]")))
    (should (photon-member-equal-p (photon-vocab-index vocab "[UNK]")
                                   tokens))))

(ert-deftest photon-wordpiece-vocab-save-load-round-trips ()
  (let* ((vocab (photon-build-wordpiece-vocab "photon stream" 4))
         (path (make-temp-file "photon-vocab-" nil ".el")))
    (unwind-protect
        (progn
          (photon-save-vocab vocab path '((tokenizer . wordpiece)))
          (should (equal vocab (photon-load-vocab path))))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-wordpiece-dataset-loader-reuses-vocab-path ()
  (let ((train-path (make-temp-file "photon-wp-train-" nil ".txt"
                                    "photon stream"))
        (eval-path (make-temp-file "photon-wp-eval-" nil ".txt"
                                   "stream photon"))
        (vocab-path (make-temp-file "photon-wp-vocab-" nil ".el")))
    (delete-file vocab-path)
    (unwind-protect
        (let* ((train (photon-load-batch-dataset-text-file
                       train-path 3
                       (list (cons 'tokenizer 'wordpiece)
                             (cons 'vocab-path vocab-path))))
               (eval (photon-load-batch-dataset-text-file
                      eval-path 3
                      (list (cons 'tokenizer 'wordpiece)
                            (cons 'vocab-path vocab-path)))))
	          (should (file-exists-p vocab-path))
	          (should (equal (cdr (assq 'vocab train))
	                         (cdr (assq 'vocab eval))))
                  (should (assq 'unknown-token-count
                                (cdr (assq 'token-diagnostics eval))))
                  (should (assq 'rare-token-count
                                (cdr (assq 'token-diagnostics eval))))
	          (should (> (cdr (assq 'token-count eval)) 0)))
      (when (file-exists-p train-path)
        (delete-file train-path))
      (when (file-exists-p eval-path)
        (delete-file eval-path))
      (when (file-exists-p vocab-path)
        (delete-file vocab-path)))))

(ert-deftest photon-batch-dataset-loader-can-limit-source-chars ()
  (let ((data-path (make-temp-file "photon-limited-data-" nil ".txt"
                                   "abcdefghij")))
    (unwind-protect
        (let ((dataset
               (photon-load-batch-dataset-text-file
                data-path 2 '((max-chars . 4)))))
          (should (= (cdr (assq 'source-chars dataset)) 10))
          (should (= (cdr (assq 'used-chars dataset)) 4))
          (should (= (cdr (assq 'token-count dataset)) 4)))
      (when (file-exists-p data-path)
        (delete-file data-path)))))

(ert-deftest photon-wordpiece-token-kind-classifies-pieces ()
  (let ((vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "p" "pho" "##ton")))
    (should (eq (photon-wordpiece-token-kind vocab 1) 'unknown))
    (should (eq (photon-wordpiece-token-kind vocab 2) 'special))
    (should (eq (photon-wordpiece-token-kind vocab 4) 'short-piece))
    (should (eq (photon-wordpiece-token-kind vocab 5) 'long-piece))
    (should (eq (photon-wordpiece-token-kind vocab 6) 'long-piece))
    (should (photon-wordpiece-token-continuation-p vocab 6))
    (should (eq (photon-wordpiece-token-shape vocab 5)
                'start-long-piece))
    (should (eq (photon-wordpiece-token-shape vocab 6)
                'continuation-long-piece))
    (should
     (photon-wordpiece-start-long-teacher-target-p
      (let ((model (photon-make-model (photon-make-config 7 2 2 1))))
        (photon-model-metadata-put model 'vocab vocab)
        (photon-model-metadata-put model 'tokenizer 'wordpiece)
        model)
      5))
    (should (= (photon-wordpiece-token-length vocab 6) 3))))

(ert-deftest photon-wordpiece-update-scale-boosts-long-rare-tokens ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (short-token 4)
         (long-token 5))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "p" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (should (> (photon-wordpiece-update-scale model long-token)
               (photon-wordpiece-update-scale model short-token)))
    (photon-model-metadata-put model 'tokenizer 'char)
    (should (= (photon-wordpiece-update-scale model long-token) 1.0))))

(ert-deftest photon-wordpiece-long-teacher-state-path-updates-training-state ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (report nil)
         (before-prototypes nil)
         (before-head nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-long-teacher-state-path-weight 0.50)
    (photon-update-context-anchor-ngram-weighted model context target 2.0)
    (setq report (photon-loss-report model context target))
    (setq before-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
    (setq before-head (copy-tree (photon-model-output-head model)))
    (should
     (photon-update-wordpiece-long-teacher-state-path
      model context target report 0.10))
    (should-not
     (equal before-prototypes
            (photon-model-output-head-prototypes model)))
    (should-not (equal before-head (photon-model-output-head model)))))

(ert-deftest photon-wordpiece-continuation-teacher-state-path-updates-training-state ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (report nil)
         (before-prototypes nil)
         (before-head nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "pho" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-continuation-teacher-state-path-weight 0.50)
    (photon-update-context-anchor-ngram-weighted model context target 2.0)
    (setq report (photon-loss-report model context target))
    (setq before-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
    (setq before-head (copy-tree (photon-model-output-head model)))
    (should
     (photon-update-wordpiece-continuation-teacher-state-path
      model context target report 0.10))
    (should-not
     (equal before-prototypes
            (photon-model-output-head-prototypes model)))
    (should-not (equal before-head (photon-model-output-head model)))))

(ert-deftest photon-wordpiece-start-long-teacher-state-path-updates-training-state ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (report nil)
         (before-prototypes nil)
         (before-head nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-start-long-teacher-state-path-weight 0.50)
    (photon-update-context-anchor-ngram-weighted model context target 2.0)
    (setq report (photon-loss-report model context target))
    (setq before-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
    (setq before-head (copy-tree (photon-model-output-head model)))
    (should
     (photon-update-wordpiece-start-long-teacher-state-path
      model context target report 0.10))
    (should-not
     (equal before-prototypes
            (photon-model-output-head-prototypes model)))
    (should-not (equal before-head (photon-model-output-head model)))))

(ert-deftest photon-wordpiece-start-long-teacher-state-can-mix-boundary-targets ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (report nil)
         (state nil)
         (baseline nil)
         (mixed nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (setq report (photon-loss-report model context target))
    (setq state (cdr (assq 'state report)))
    (setq baseline
          (photon-wordpiece-start-long-teacher-state
           model context state target))
    (photon-model-param-set
     model 'wordpiece-start-long-teacher-token-mix 0.50)
    (photon-model-param-set
     model 'wordpiece-start-long-teacher-context-mix 0.25)
    (photon-model-param-set
     model 'wordpiece-start-long-teacher-anchor-mix 0.25)
    (setq mixed
          (photon-wordpiece-start-long-teacher-state
           model context state target))
    (should-not (equal baseline mixed))))

(ert-deftest photon-start-long-boundary-readout-learns-start-long-token ()
  (let* ((config (photon-make-config 8 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (start-target 6)
         (continuation-target 7)
         (logits nil)
         (components nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-start-long-boundary-readout-enabled 1.0)
    (photon-model-param-set
     model 'wordpiece-start-long-boundary-readout-weight 4.0)
    (photon-model-param-set
     model 'wordpiece-start-long-boundary-ngram-mix 1.0)
    (photon-update-start-long-boundary-readout model context start-target)
    (photon-update-start-long-boundary-readout
     model context continuation-target)
    (photon-update-context-anchor-ngram-weighted
     model context start-target 2.0)
    (setq logits
          (photon-start-long-boundary-readout-logits model context))
    (should (> (nth start-target logits) 0.0))
    (should (= (nth continuation-target logits) 0.0))
    (setq components
          (photon-model-readout-components
           model
           (cdr (assq 'state (photon-loss-report model context start-target)))
           context))
    (should (assq 'start-long-boundary-readout components))
    (should (> (nth start-target
                    (cdr (assq 'start-long-boundary-readout components)))
               0.0))))

(ert-deftest photon-start-long-boundary-teacher-is-distill-only ()
  (let* ((config (photon-make-config 8 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (state nil)
         (components nil)
         (full nil)
         (teacher nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-start-long-boundary-readout-enabled 1.0)
    (photon-model-param-set
     model 'wordpiece-start-long-boundary-ngram-mix 1.0)
    (photon-model-param-set
     model
     'output-head-full-readout-distill-start-long-boundary-teacher-weight
     4.0)
    (photon-update-start-long-boundary-readout model context target)
    (photon-update-context-anchor-ngram-weighted model context target 2.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (setq components (photon-model-readout-components model state context))
    (setq full (cdr (assq 'full components)))
    (setq teacher (cdr (assq 'start-long-boundary-teacher components)))
    (should (assq 'start-long-boundary-teacher components))
    (should (> (nth target teacher) (nth target full)))))

(ert-deftest photon-start-long-ngram-teacher-is-distill-only ()
  (let* ((config (photon-make-config 8 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (continuation 7)
         (state nil)
         (components nil)
         (full nil)
         (teacher nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model
     'output-head-full-readout-distill-start-long-ngram-teacher-weight
     4.0)
    (photon-update-context-anchor-ngram-weighted model context target 2.0)
    (photon-update-context-anchor-ngram-weighted
     model context continuation 2.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (setq components (photon-model-readout-components model state context))
    (setq full (cdr (assq 'full components)))
    (setq teacher (cdr (assq 'start-long-ngram-teacher components)))
    (should (assq 'start-long-ngram-teacher components))
    (should (> (nth target teacher) (nth target full)))
    (should (= (nth continuation teacher) (nth continuation full)))))

(ert-deftest photon-start-long-classifier-learns-start-long-token ()
  (let* ((config (photon-make-config 9 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 6)
         (state nil)
         (before nil)
         (after nil)
         (components nil)
         (teacher nil)
         (full nil)
         (readout nil)
         (unmixed-full nil)
         (mixed-full nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "photo" "world" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-start-long-classifier-enabled 1.0)
    (photon-model-param-set
     model
     'output-head-full-readout-distill-start-long-classifier-teacher-weight
     4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (setq before (photon-model-start-long-classifier-logits model state))
    (photon-update-start-long-classifier
     model state 7 target 0.20)
    (setq after (photon-model-start-long-classifier-logits model state))
    (should (> (nth target after) (nth target before)))
    (should (= (nth 8 after) -1000.0))
    (setq components (photon-model-readout-components model state context))
    (setq full (cdr (assq 'full components)))
    (setq teacher (cdr (assq 'start-long-classifier-teacher components)))
    (should (assq 'start-long-classifier components))
    (should (assq 'start-long-classifier-readout components))
    (should (assq 'start-long-classifier-teacher components))
    (should (> (nth target teacher) (nth target full)))
    (setq unmixed-full full)
    (photon-model-param-set
     model 'wordpiece-start-long-classifier-readout-weight 0.25)
    (setq components (photon-model-readout-components model state context))
    (setq mixed-full (cdr (assq 'full components)))
    (setq readout (cdr (assq 'start-long-classifier-readout components)))
    (should (> (nth target readout) 0.0))
    (should (= (nth 8 readout) 0.0))
    (should (> (nth target mixed-full)
               (nth target unmixed-full)))))

(ert-deftest photon-start-long-classifier-candidate-search-accepts-improvement ()
  (let* ((config (photon-make-config 9 6 2 1))
         (model (photon-make-model config))
         (context '(4 5))
         (target 7)
         (rival 6)
         (tokens (append context (list target)))
         (state nil)
         (before nil)
         (after nil)
         (result nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "photo" "world" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'wordpiece-start-long-classifier-enabled 1.0)
    (photon-model-param-set
     model 'wordpiece-start-long-classifier-candidate-learning-rate 0.50)
    (photon-model-param-set
     model 'wordpiece-start-long-classifier-candidate-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-start-long-classifier model state target rival 0.50)
    (setq before (photon-model-start-long-classifier-logits model state))
    (setq result
          (photon-start-long-classifier-candidate-search
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)))
    (setq after (photon-model-start-long-classifier-logits model state))
    (should (= (cdr (assq 'candidate-count result)) 1))
    (should (= (cdr (assq 'tested result)) 1))
    (should (= (cdr (assq 'accepted result)) 1))
    (should (= (cdr (assq 'updated result)) 1))
    (should (> (nth target after) (nth target before)))
    (should (> (cdr (assq 'best-start-long-classifier-accuracy-permil
                          result))
               (cdr (assq 'accuracy-permil
                          (cdr (assq 'initial-start-long-classifier
                                     result))))))))

(ert-deftest photon-start-long-positive-distill-uses-ngram-hit ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (tokens (append context (list target)))
         (state nil)
         (before-linear nil)
         (after-linear nil)
         (before-prototype nil)
         (after-prototype nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-start-long-positive-distill-top-k 3.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (setq before-linear (photon-model-output-head-logits model state))
    (setq before-prototype
          (photon-model-output-head-prototype-logits model state))
    (setq report
          (photon-start-long-positive-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)
           0.25))
    (setq after-linear (photon-model-output-head-logits model state))
    (setq after-prototype
          (photon-model-output-head-prototype-logits model state))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'prototype-updated report)) 1))
    (should (> (nth target after-linear) (nth target before-linear)))
    (should (> (nth target after-prototype) (nth target before-prototype)))))

(ert-deftest photon-start-long-projected-prototype-distill-stores-projected-state ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (tokens (append context (list target)))
         (state nil)
         (before nil)
         (after nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-start-long-projected-prototype-distill-top-k 1.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-projection model state 0.10)
    (setq before (photon-model-output-head-prototype-logits model state))
    (setq report
          (photon-start-long-projected-prototype-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)))
    (setq after (photon-model-output-head-prototype-logits model state))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'projected-prototype-updated report)) 1))
    (should (> (nth target after) (nth target before)))))

(ert-deftest photon-start-long-prototype-merge-updates-target-prototype ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before-prototypes nil)
         (after-prototypes nil)
         (before-logits nil)
         (after-logits nil)
         (before-target-bias nil)
         (after-target-bias nil)
         (before-rival-bias nil)
         (after-rival-bias nil)
         (negative-prototypes nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rate 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-top-k 10.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-margin 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-target-bias-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-bias-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-score-bonus-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-line-search-step
     0.05)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-line-search-max
     0.10)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-negative-weight 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rivals 1.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype
     model
     (photon-vector-squash
      (photon-vector-add state
                         (photon-vector-scale
                          0.40
                          (photon-model-token-embedding model rival))))
     target)
    (photon-update-output-head-prototype
     model
     (photon-vector-squash
      (photon-vector-add state
                         (photon-vector-scale
                          0.30
                          (photon-model-token-embedding model rival))))
     rival)
    (setq before-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
    (setq before-logits
          (photon-model-output-head-prototype-logits model state))
    (setq before-target-bias
          (or (nth target
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (setq before-rival-bias
          (or (nth rival
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (setq report
          (photon-start-long-prototype-merge-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)))
    (setq after-prototypes
          (photon-model-output-head-prototypes model))
    (setq after-logits
          (photon-model-output-head-prototype-logits model state))
    (setq after-target-bias
          (or (nth target
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (setq after-rival-bias
          (or (nth rival
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (setq negative-prototypes
          (photon-model-output-head-negative-prototypes model))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'merged report)) 1))
    (should (= (cdr (assq 'target-bias-updated report)) 1))
    (should (= (cdr (assq 'rival-bias-updated report)) 1))
    (should (= (cdr (assq 'rival-negative-updated report)) 1))
    (should (> (length (cdr (assq target after-prototypes)))
               (length (cdr (assq target before-prototypes)))))
    (should (consp (cdr (assq rival negative-prototypes))))
    (should (> after-target-bias before-target-bias))
    (should (< after-rival-bias before-rival-bias))
    (should (> (nth target after-logits) (nth target before-logits)))))

(ert-deftest photon-start-long-prototype-merge-candidate-search-accepts-improvement ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before-output-head nil)
         (result nil)
         (negative-prototypes nil)
         (target-prototypes nil)
         (after-output-head nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rate 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-top-k 10.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-margin 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-target-bias-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-bias-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-score-bonus-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-dominant-rival-bias-rate
     0.20)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-dominant-rival-min-count
     1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-state-separation-rate
     0.25)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-dense-head-rate
     0.02)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-line-search-step
     0.05)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-line-search-max
     0.10)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rival-negative-weight 1.0)
    (photon-model-param-set
     model 'output-head-start-long-prototype-merge-rivals 1.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq before-output-head (copy-tree (photon-model-output-head model)))
    (setq result
          (photon-start-long-prototype-merge-candidate-search
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)))
    (setq negative-prototypes
          (photon-model-output-head-negative-prototypes model))
    (setq target-prototypes
          (cdr (assq target
                     (photon-model-output-head-prototypes model))))
    (setq after-output-head (photon-model-output-head model))
    (should (= (cdr (assq 'candidate-count result)) 1))
    (should (= (cdr (assq 'tested result)) 1))
    (should (= (cdr (assq 'accepted result)) 1))
    (should (= (cdr (assq 'target-bias-updated result)) 1))
    (should (= (cdr (assq 'rival-bias-updated result)) 1))
    (should (= (cdr (assq 'score-bonus-updated result)) 1))
    (should (= (cdr (assq 'dominant-rival-bias-updated result)) 1))
    (should (= (cdr (assq 'state-separation-updated result)) 1))
    (should (= (cdr (assq 'dense-head-updated result)) 1))
    (should (= (cdr (assq 'rival-negative-updated result)) 1))
    (should (assq rival (cdr (assq 'dominant-rivals result))))
    (should (> (cdr (assq 'line-search-attempted result)) 0))
    (should (consp (cdr (assq rival negative-prototypes))))
    (should (> (length target-prototypes) 1))
    (should-not (equal before-output-head after-output-head))
    (should (> (cdr (assq 'best-start-long-prototype-accuracy-permil result))
               (cdr (assq 'accuracy-permil
                          (cdr (assq 'initial-start-long-prototype
                                     result))))))))

(ert-deftest photon-start-long-projection-separation-updates-projection ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before-projection nil)
         (after-projection nil)
         (before-logits nil)
         (after-logits nil)
         (before-target-score-bonus nil)
         (after-target-score-bonus nil)
         (before-rival-bias nil)
         (after-rival-bias nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-rate 0.05)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-top-k 10.0)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-margin 1.00)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-score-bonus-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-rival-bias-rate 0.20)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state target)
    (photon-update-output-head-prototype
     model
     (photon-vector-squash
      (photon-vector-add state
                         (photon-vector-scale
                          0.30
                          (photon-model-token-embedding model rival))))
     rival)
    (setq before-logits
          (photon-model-output-head-prototype-logits model state))
    (setq before-target-score-bonus
          (or (nth target
                   (photon-model-output-head-prototype-score-bonus model))
              0.0))
    (setq before-rival-bias
          (or (nth rival
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (setq before-projection (copy-tree
                             (photon-model-output-head-projection model)))
    (setq report
          (photon-start-long-projection-separation-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)))
    (setq after-projection (photon-model-output-head-projection model))
    (setq after-logits
          (photon-model-output-head-prototype-logits model state))
    (setq after-target-score-bonus
          (or (nth target
                   (photon-model-output-head-prototype-score-bonus model))
              0.0))
    (setq after-rival-bias
          (or (nth rival
                   (photon-model-output-head-prototype-bias model))
              0.0))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'updated report)) 1))
    (should (= (cdr (assq 'score-bonus-updated report)) 1))
    (should (= (cdr (assq 'rival-bias-updated report)) 1))
    (should (> after-target-score-bonus before-target-score-bonus))
    (should (< after-rival-bias before-rival-bias))
    (should (> (nth target after-logits) (nth target before-logits)))
    (should (< (nth rival after-logits) (nth rival before-logits)))
    (should (not (equal before-projection after-projection)))))

(ert-deftest photon-start-long-projection-separation-pocket-restores-rejected-state ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before-projection nil)
         (before-score-bonus nil)
         (before-bias nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-rate 0.05)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-top-k 10.0)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-margin 1.00)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-score-bonus-rate 0.20)
    (photon-model-param-set
     model 'output-head-start-long-projection-separation-rival-bias-rate 0.20)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state target)
    (photon-update-output-head-prototype
     model
     (photon-vector-squash
      (photon-vector-add state
                         (photon-vector-scale
                          0.30
                          (photon-model-token-embedding model rival))))
     rival)
    (setq before-projection
          (copy-tree (photon-model-output-head-projection model)))
    (setq before-score-bonus
          (copy-tree (photon-model-output-head-prototype-score-bonus model)))
    (setq before-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
    (setq report
          (photon-start-long-projection-separation-pocket
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)
           1))
    (should (= (cdr (assq 'accepted report)) 0))
    (should (= (cdr (assq 'rejected report)) 1))
    (should (equal before-projection
                   (photon-model-output-head-projection model)))
    (should (equal before-score-bonus
                   (photon-model-output-head-prototype-score-bonus model)))
    (should (equal before-bias
                   (photon-model-output-head-prototype-bias model)))))

(ert-deftest photon-start-long-rival-negative-distill-penalizes-rival ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before nil)
         (after nil)
         (report nil)
         (negative-prototypes nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set
     model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model 'output-head-start-long-rival-negative-distill-weight 1.0)
    (photon-model-param-set
     model 'output-head-start-long-rival-negative-distill-top-k 3.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq before (photon-model-output-head-prototype-logits model state))
    (should (= (photon-argmax-index before) rival))
    (setq report
          (photon-start-long-rival-negative-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)))
    (setq after (photon-model-output-head-prototype-logits model state))
    (setq negative-prototypes
          (photon-model-output-head-negative-prototypes model))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'negative-updated report)) 1))
    (should (< (nth rival after) (nth rival before)))
    (should (consp (cdr (assq rival negative-prototypes))))))

(ert-deftest photon-start-long-same-shape-margin-distill-separates-rival ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before nil)
         (after nil)
         (before-bias nil)
         (after-bias nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set
     model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-margin-distill-bias-weight
     0.25)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq before (photon-model-output-head-prototype-logits model state))
    (setq before-bias (copy-sequence
                       (photon-model-output-head-prototype-bias model)))
    (should (= (photon-argmax-index before) rival))
    (setq report
          (photon-start-long-same-shape-margin-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)
           0.0))
    (setq after (photon-model-output-head-prototype-logits model state))
    (setq after-bias (photon-model-output-head-prototype-bias model))
    (should (= (cdr (assq 'start-long-total report)) 1))
    (should (= (cdr (assq 'ngram-top-k-hit report)) 1))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'bias-updated report)) 1))
    (should (> (nth target after) (nth target before)))
    (should (< (nth rival after) (nth rival before)))
    (should (> (nth target after-bias) (nth target before-bias)))
    (should (< (nth rival after-bias) (nth rival before-bias)))))

(ert-deftest photon-start-long-same-shape-margin-distill-adapts-bias ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (before nil)
         (after nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
     0.50)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-margin 0.20)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq before (photon-model-output-head-prototype-logits model state))
    (setq report
          (photon-start-long-same-shape-margin-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)
           0.0))
    (setq after (photon-model-output-head-prototype-logits model state))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'bias-updated report)) 1))
    (should (> (photon-target-margin after target rival)
               (photon-target-margin before target rival)))))

(ert-deftest photon-start-long-same-shape-margin-distill-demotes-multiple-rivals ()
  (let* ((config (photon-make-config 11 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival-a 8)
         (rival-b 9)
         (tokens (append context (list target)))
         (state nil)
         (bias-before nil)
         (bias-after nil)
         (report nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "start" "##x"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
     0.50)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-margin 0.20)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-rivals 2.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival-a)
    (photon-update-output-head-prototype model state rival-b)
    (setq bias-before (copy-tree (photon-model-output-head-prototype-bias model)))
    (setq report
          (photon-start-long-same-shape-margin-distill-token-reader
           model
           (photon-make-list-token-reader tokens)
           (length context)
           0.0))
    (setq bias-after (photon-model-output-head-prototype-bias model))
    (should (= (cdr (assq 'candidates report)) 1))
    (should (= (cdr (assq 'rivals report)) 2))
    (should (> (nth target bias-after) (nth target bias-before)))
    (should (< (nth rival-a bias-after) (nth rival-a bias-before)))
    (should (< (nth rival-b bias-after) (nth rival-b bias-before)))))

(ert-deftest photon-output-head-prototype-score-bonus-adjusts-positive-score ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (state nil)
         (before nil)
         (after nil)
         (detail nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state target)
    (setq before (photon-model-output-head-prototype-logits model state))
    (photon-update-output-head-prototype-score-bonus model target 0.50)
    (setq after (photon-model-output-head-prototype-logits model state))
    (setq detail
          (photon-output-head-prototype-token-detail model state target))
    (should (= (cdr (assq 'score-bonus detail)) 0.50))
    (should (> (nth target after) (nth target before)))))

(ert-deftest photon-output-head-context-bonus-prototype-boosts-near-state ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (state nil)
         (before nil)
         (after nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set
     model 'output-head-prototype-context-bonus-weight 2.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (setq before (photon-model-output-head-prototype-logits model state))
    (photon-update-output-head-context-bonus-prototype model state target)
    (setq after (photon-model-output-head-prototype-logits model state))
    (should (> (nth target after) (nth target before)))))

(ert-deftest photon-start-long-same-shape-margin-distill-pocket-reports-best ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (result nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
     0.50)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq result
          (photon-start-long-same-shape-margin-distill-pocket
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)
           0.0
           1))
    (should (assq 'initial-start-long-prototype result))
    (should (assq 'best-start-long-prototype-accuracy-permil result))
    (should (>= (cdr (assq 'best-start-long-prototype-accuracy-permil
                           result))
                (cdr (assq 'accuracy-permil
                           (cdr (assq 'initial-start-long-prototype
                                      result))))))))

(ert-deftest photon-start-long-same-shape-top1-bias-search-accepts-improvement ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (result nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
     0.50)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-margin 0.20)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq result
          (photon-start-long-same-shape-top1-bias-search
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)))
    (should (= (cdr (assq 'candidate-count result)) 1))
    (should (= (cdr (assq 'accepted result)) 1))
    (should (> (cdr (assq 'best-start-long-prototype-accuracy-permil result))
               (cdr (assq 'accuracy-permil
                          (cdr (assq 'initial-start-long-prototype
                                     result))))))))

(ert-deftest photon-start-long-same-shape-top1-search-accepts-context-bonus ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (result nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-top1-search-context-bonus-weight
     2.0)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-margin 0.20)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq result
          (photon-start-long-same-shape-top1-bias-search
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)))
    (should (= (cdr (assq 'candidate-count result)) 1))
    (should (= (cdr (assq 'accepted result)) 1))
    (should (> (cdr (assq 'best-start-long-prototype-accuracy-permil result))
               (cdr (assq 'accuracy-permil
                          (cdr (assq 'initial-start-long-prototype
                                     result))))))))

(ert-deftest photon-start-long-same-shape-top1-search-reports-tolerance ()
  (let* ((config (photon-make-config 10 6 2 1))
         (model (photon-make-model config))
         (context '(4 5 6 4))
         (target 7)
         (rival 8)
         (tokens (append context (list target)))
         (state nil)
         (result nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c" "phot" "word" "##ton"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-projection-enabled 0.0)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-top1-search-context-bonus-weight
     2.0)
    (photon-model-param-set
     model
     'output-head-start-long-same-shape-top1-search-global-tolerance
     25.0)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-margin 0.20)
    (photon-model-param-set
     model 'output-head-start-long-same-shape-margin-distill-top-k 10.0)
    (photon-update-context-anchor-ngram-weighted model context target 4.0)
    (setq state (cdr (assq 'state (photon-loss-report model context target))))
    (photon-update-output-head-prototype model state rival)
    (setq result
          (photon-start-long-same-shape-top1-bias-search
           model
           (lambda () (photon-make-list-token-reader tokens))
           (length context)))
    (should (= (cdr (assq 'global-tolerance-permil result)) 25.0))
    (should (= (cdr (assq 'accuracy-floor-permil result))
               (max 0
                    (- (cdr (assq 'accuracy-permil
                                  (cdr (assq 'initial-output-head-prototype
                                             result))))
                       25.0))))))

(ert-deftest photon-full-readout-distill-scale-can-use-wordpiece-scale ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (long-token 5))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "p" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (should (= (photon-full-readout-distill-update-scale model long-token)
               1.0))
    (photon-model-param-set
     model 'output-head-full-readout-distill-wordpiece-scale-enabled 1.0)
    (should (> (photon-full-readout-distill-update-scale model long-token)
               1.0))
    (photon-model-param-set
     model 'output-head-full-readout-distill-start-long-scale 2.0)
    (should (> (photon-full-readout-distill-update-scale model long-token)
               2.0))))

(ert-deftest photon-full-readout-distill-accepts-teacher-key ()
  (let* ((config (photon-make-config 6 6 2 1))
         (model (photon-make-model config))
         (reader (photon-make-list-token-reader '(1 2 3 4)))
         (result (photon-distill-full-readout-to-output-head-token-reader
                  model reader 2 0.05 'structural-off)))
    (should (eq (cdr (assq 'teacher-key result)) 'structural-off))
    (should (= (cdr (assq 'total result)) 2))))

(ert-deftest photon-context-anchor-ngram-weighted-update-adds-fractional-count ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (logits nil))
    (photon-update-context-anchor-ngram-weighted model '(4 5) 6 0.75)
    (setq logits (photon-context-anchor-ngram-logits model '(4 5)))
    (should (> (nth 6 logits) 0.0))))

(ert-deftest photon-wordpiece-long-rare-readout-rescue-updates-ngram-anchor ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (tokens '(4 5 6))
         (result nil)
         (logits nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (setq result
          (photon-wordpiece-long-rare-readout-rescue-token-reader
           model (photon-make-list-token-reader tokens) 2 0.50))
    (setq logits (photon-context-anchor-ngram-logits model '(4 5)))
    (should (= (cdr (assq 'total result)) 1))
    (should (= (cdr (assq 'updated result)) 1))
    (should (= (cdr (assq 'long-updated result)) 1))
    (should (= (cdr (assq 'rare-updated result)) 1))
    (should (> (nth 6 logits) 0.0))))

(ert-deftest photon-readout-distill-updates-output-head-prototypes ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (tokens '(4 5 6))
         (result nil)
         (prototypes nil)
         (bias nil)
         (before-head nil)
         (after-head nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-bias-weight 1.0)
    (photon-update-context-anchor-ngram-weighted model '(4 5) 6 2.0)
    (setq result
          (photon-distill-readout-to-output-head-prototypes-token-reader
           model (photon-make-list-token-reader tokens) 2 'ngram-anchor))
    (setq prototypes (photon-model-output-head-prototypes model))
    (should (= (cdr (assq 'total result)) 1))
    (should (= (cdr (assq 'teacher-correct result)) 1))
    (should (= (cdr (assq 'updated result)) 1))
    (should (consp (cdr (assq 6 prototypes))))))

(ert-deftest photon-long-piece-state-path-updates-model-path ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (before-table (copy-tree (photon-model-embedding-table model)))
         (before-gain
          (copy-tree (photon-model-gain-get model 'context-current-gain)))
         (before-projection
          (copy-tree (photon-model-projection-get model 'context-proj))))
    (photon-update-long-piece-state-path model '(4 5) 5 6 0.05)
    (should-not (equal before-table (photon-model-embedding-table model)))
    (should-not
     (equal before-gain
            (photon-model-gain-get model 'context-current-gain)))
    (should-not
     (equal before-projection
            (photon-model-projection-get model 'context-proj)))))

(ert-deftest photon-ngram-long-distill-updates-output-head-prototypes ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (tokens '(4 5 6))
         (result nil)
         (prototypes nil)
         (negative-prototypes nil)
         (bias nil)
         (before-head nil)
         (after-head nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "phot"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-bias-weight 1.0)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-contrastive-rate 0.05)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-state-path-rate 0.05)
    (photon-update-context-anchor-ngram-weighted model '(4 5) 6 2.0)
    (setq before-head (copy-tree (photon-model-output-head model)))
    (setq result
          (photon-distill-ngram-long-to-output-head-prototypes-token-reader
           model (photon-make-list-token-reader tokens) 2))
    (setq prototypes (photon-model-output-head-prototypes model))
    (setq negative-prototypes
          (photon-model-output-head-negative-prototypes model))
    (setq bias (photon-model-output-head-prototype-bias model))
    (setq after-head (photon-model-output-head model))
    (should (= (cdr (assq 'total result)) 1))
    (should (= (cdr (assq 'long-total result)) 1))
    (should (= (cdr (assq 'ngram-correct result)) 1))
    (should (= (cdr (assq 'updated result)) 1))
    (should (= (cdr (assq 'negative-updated result)) 1))
    (should (= (cdr (assq 'state-path-updated result)) 1))
    (should (> (nth 6 bias) 0.0))
    (should-not (equal before-head after-head))
    (should (consp negative-prototypes))
    (should (consp (cdr (assq 6 prototypes))))))

(ert-deftest photon-output-head-negative-prototype-penalizes-rival ()
  (let* ((config (photon-make-config 7 2 2 1))
         (model (photon-make-model config))
         (state (photon-vector-from-list '(1.0 0.0)))
         (before nil)
         (after nil))
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-negative-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-projection-enabled 0.0)
    (photon-update-output-head-prototype model state 5)
    (setq before (photon-model-output-head-prototype-logits model state))
    (photon-update-output-head-negative-prototype model state 5)
    (setq after (photon-model-output-head-prototype-logits model state))
    (should (< (nth 5 after) (nth 5 before)))))

(ert-deftest photon-ngram-long-prototype-state-mixes-context-and-token ()
  (let* ((config (photon-make-config 7 3 2 1))
         (model (photon-make-model config))
         (state (photon-vector-from-list '(0.1 0.2 0.3)))
         (mixed nil))
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-token-mix 0.5)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-context-mix 0.5)
    (photon-model-param-set
     model 'output-head-prototype-ngram-long-anchor-mix 0.5)
    (setq mixed
          (photon-ngram-long-output-head-prototype-state
           model state '(4 5) 6))
    (should (= (length mixed) 3))
    (should-not (equal mixed state))))

(ert-deftest photon-output-head-prototype-diverse-trim-keeps-distant-states ()
  (let* ((config (photon-make-config 7 3 2 1))
         (model (photon-make-model config))
         (near-a (photon-vector-from-list '(1.0 0.0 0.0)))
         (near-b (photon-vector-from-list '(0.99 0.01 0.0)))
         (far (photon-vector-from-list '(0.0 1.0 0.0)))
         (states nil))
    (photon-model-param-set model 'output-head-prototype-limit 2.0)
    (photon-model-param-set model 'output-head-prototype-diverse-trim-enabled 1.0)
    (photon-update-output-head-prototype model near-a 6)
    (photon-update-output-head-prototype model near-b 6)
    (photon-update-output-head-prototype model far 6)
    (setq states (cdr (assq 6 (photon-model-output-head-prototypes model))))
    (should (= (length states) 2))
    (should (member far states))
    (should (or (member near-a states) (member near-b states)))))

(ert-deftest photon-output-head-prototype-evaluator-reports-accuracy ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (tokens '(4 5 6))
         (score nil))
    (photon-update-output-head-prototype
     model
     (photon-prediction-state
      (photon-forward-model model '(4 5))
      '(4 5))
     6)
    (setq score
          (photon-evaluate-output-head-prototype-token-reader
           model (photon-make-list-token-reader tokens) 2))
    (should (= (cdr (assq 'total score)) 1))
    (should (assq 'accuracy-permil score))))

(ert-deftest photon-output-head-prototype-supports-covered-targets ()
  (let* ((config (photon-make-config 7 2 2 1))
         (model (photon-make-model config))
         (state (photon-vector-from-list '(1.0 0.0)))
         (logits nil))
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-support-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-support-limit 3.0)
    (photon-model-param-set model 'output-head-prototype-projection-enabled 0.0)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(1.0 0.0)) 4)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(0.9 0.0)) 5)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(0.9 0.0)) 5)
    (setq logits (photon-model-output-head-prototype-logits model state))
    (should (> (nth 5 logits) (nth 4 logits)))))

(ert-deftest photon-output-head-prototype-weights-wordpiece-token-kind ()
  (let* ((config (photon-make-config 7 2 2 1))
         (model (photon-make-model config))
         (state (photon-vector-from-list '(1.0 0.0)))
         (baseline nil)
         (boosted nil))
    (photon-model-metadata-put
     model 'vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "phot" "##on"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-projection-enabled 0.0)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(1.0 0.0)) 5)
    (setq baseline (photon-model-output-head-prototype-logits model state))
    (photon-model-param-set model 'output-head-prototype-long-piece-weight 2.0)
    (setq boosted (photon-model-output-head-prototype-logits model state))
    (should (> (nth 5 boosted) (nth 5 baseline)))))

(ert-deftest photon-output-head-prototype-weights-wordpiece-token-shape ()
  (let* ((config (photon-make-config 8 2 2 1))
         (model (photon-make-model config))
         (state (photon-vector-from-list '(1.0 0.0)))
         (baseline nil)
         (boosted nil))
    (photon-model-metadata-put
     model 'vocab
     '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "phot" "##on" "cat"))
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'output-head-prototype-weight 1.0)
    (photon-model-param-set model 'output-head-prototype-projection-enabled 0.0)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(1.0 0.0)) 5)
    (photon-update-output-head-prototype
     model (photon-vector-from-list '(1.0 0.0)) 6)
    (setq baseline (photon-model-output-head-prototype-logits model state))
    (photon-model-param-set
     model 'output-head-prototype-start-long-piece-weight 2.0)
    (setq boosted (photon-model-output-head-prototype-logits model state))
    (should (> (nth 5 boosted) (nth 5 baseline)))
    (should (= (nth 6 boosted) (nth 6 baseline)))))

(ert-deftest photon-full-readout-distill-pocket-reports-prototype-score ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (tokens '(4 5 6 4))
         (result nil))
    (setq result
          (photon-distill-full-readout-to-output-head-token-reader-pocket
           model
           (lambda () (photon-make-list-token-reader tokens))
           2 0.05 1 'full))
    (should (assq 'initial-output-head-prototype result))
    (should (assq 'best-prototype-accuracy-permil result))
    (should (assq 'initial-start-long-linear result))
    (should (assq 'best-start-long-linear-accuracy-permil result))
    (should (assq 'best-selection-score result))))

(ert-deftest photon-wordpiece-generation-adjusts-repeat-and-continuation ()
  (let* ((config (photon-make-config 6 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "phot" "##on"))
         (logits '(0.0 0.0 0.0 0.0 10.0 10.0)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (let ((adjusted (photon-generation-adjust-logits model '(4 4 5 5) logits)))
      (should (< (nth 1 adjusted) -999999.0))
      (should (< (nth 5 adjusted) (nth 5 logits)))
      (should (<= (nth 4 adjusted) (nth 4 logits))))))

(ert-deftest photon-wordpiece-generation-penalizes-repeated-bigram ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 10.0 10.0 10.0)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-model-param-set model 'wordpiece-generation-repeat-ngram-bias-weight 0.75)
    (let ((adjusted (photon-generation-adjust-logits
                     model '(4 5 6 4) logits)))
      (should (< (nth 5 adjusted) (nth 5 logits)))
      (should (= (nth 6 adjusted) (nth 6 logits))))))

(ert-deftest photon-wordpiece-generation-adapts-repeated-bigram-penalty ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 10.0 10.0 10.0)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (let ((single (photon-generation-adjust-logits
                   model '(4 5 6 4) logits
                   '((repeat-ngram-bias . 0.75)
                     (repeat-ngram-bias-min-count . 2.0))))
          (repeated (photon-generation-adjust-logits
                     model '(4 5 4 5 6 4) logits
                     '((repeat-ngram-bias . 0.75)
                       (repeat-ngram-bias-min-count . 2.0)))))
      (should (= (nth 5 single) (nth 5 logits)))
      (should (< (nth 5 repeated) (nth 5 logits))))))

(ert-deftest photon-wordpiece-generation-boosts-ngram-candidates ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 1.0 1.0 1.0)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-update-context-anchor-ngram model '(4 5) 6)
    (let ((adjusted (photon-generation-adjust-logits
                     model '(4 5) logits
                     '((ngram-bias . 1.5)))))
      (should (> (nth 6 adjusted) (nth 6 logits)))
      (should (= (nth 4 adjusted) (nth 4 logits))))))

(ert-deftest photon-wordpiece-generation-gates-distant-ngram-candidates ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 10.0 1.0 1.0)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-update-context-anchor-ngram model '(4 5) 6)
    (let ((ungated (photon-generation-adjust-logits
                    model '(4 5) logits '((ngram-bias . 2.0))))
          (gated (photon-generation-adjust-logits
                  model '(4 5) logits
                  '((ngram-bias . 2.0)
                    (ngram-bias-gate-margin . 2.0)))))
      (should (> (nth 6 ungated) (nth 6 logits)))
      (should (= (nth 6 gated) (nth 6 logits))))))

(ert-deftest photon-wordpiece-generation-gates-close-ngram-candidates ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 10.0 1.0 9.5)))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-update-context-anchor-ngram model '(4 5) 6)
    (let ((adjusted (photon-generation-adjust-logits
                     model '(4 5) logits
                     '((ngram-bias . 2.0)
                       (ngram-bias-gate-margin . 1.0)))))
      (should (> (nth 6 adjusted) (nth 6 logits)))
      (should (< (- (nth 6 adjusted) (nth 6 logits)) 2.0)))))

(ert-deftest photon-wordpiece-generation-repeat-scales-ngram-candidates ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "b" "c"))
         (logits '(0.0 0.0 0.0 0.0 1.0 1.0 1.0))
         (scaled-options '((ngram-bias . 2.0)
                           (ngram-bias-repeat-scale . 0.0)
                           (generation-history . (4 5 6 4 5)))))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-update-context-anchor-ngram model '(4 5) 6)
    (let ((baseline (photon-generation-adjust-logits
                     model '(4 5) logits
                     '((ngram-bias . 2.0))))
          (scaled (photon-generation-adjust-logits
                   model '(4 5) logits scaled-options)))
      (should (> (nth 6 baseline) (nth 6 logits)))
      (should (= (nth 6 scaled) (nth 6 logits))))))

(ert-deftest photon-wordpiece-generate-text-passes-ngram-bias-option ()
  (let* ((config (photon-make-config 7 6 2 1))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "a" "##b" "c")))
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-metadata-put model 'context-size 2)
    (photon-model-metadata-put model 'wordpiece-max-piece-size 2)
    (photon-model-param-set model 'wordpiece-generation-recent-token-bias-weight 0.0)
    (photon-update-context-anchor-ngram model '(4 5) 6)
    (should (string-suffix-p
             "c"
             (photon-model-generate-text
              model "ab" 1 '((ngram-bias . 100.0)))))))

(ert-deftest photon-wordpiece-readout-bias-boosts-long-rare-ngram-signal ()
  (let* ((config (photon-make-config 7 6 3 2))
         (model (photon-make-model config))
         (vocab '("[PAD]" "[UNK]" "[BOS]" "[EOS]" "p" "phot" "##long"))
         (source '(0.0 0.0 0.0 0.0 0.2 0.5 0.5))
         bias)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-param-set model 'wordpiece-readout-bias-enabled 1.0)
    (setq bias (photon-wordpiece-readout-bias-logits model source))
    (should (> (nth 6 bias) (nth 4 bias)))
    (should (> (nth 6 bias) 0.0))))

(ert-deftest photon-generation-sampling-supports-top-k-and-penalty ()
  (let* ((config (photon-make-config 6 6 3 2))
         (model (photon-make-model config))
         (generated
          (photon-model-generate
           model '(1 2 3) 2
           '((mode . sample) (temperature . 0.8) (top-k . 3)
             (repetition-penalty . 1.2)))))
    (should (= (length generated) 5))
    (should (integerp (car (last generated))))))

(ert-deftest photon-generation-sampling-seed-is-deterministic ()
  (let* ((config (photon-make-config 6 6 3 2))
         (model (photon-make-model config))
         (options '((mode . sample) (temperature . 0.8)
                    (top-k . 4) (seed . 17)))
         (first (photon-model-generate model '(1 2 3) 6 options))
         (second (photon-model-generate model '(1 2 3) 6 options)))
    (should (equal first second))))

(ert-deftest photon-sampling-does-not-always-pick-masked-first-index ()
  (let ((token (photon-sample-index
                '(-1000000000.0 10.0 0.0)
                '((mode . sample) (temperature . 1.0))
                nil)))
    (should (= token 1))))

(ert-deftest photon-wordpiece-training-generates-text ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (text "photon stream\nphoton checked\n")
         (result (photon-train-wordpiece-text text 2 6 2 1 0.20 4))
         (model (cdr (assq 'model result))))
    (should (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece))
    (should (= (photon-model-param-get model 'context-anchor-ngram-weight) 16.0))
    (should (= (photon-model-param-get model 'readout-logit-temperature) 8.0))
    (should (= (photon-model-param-get model 'output-head-prototype-projection-enabled) 0.0))
    (should (= (photon-model-param-get model 'output-head-linear-readout-weight) 1.0))
    (should (> (length (cdr (assq 'tokens result))) 0))
    (should (stringp (photon-model-generate-text model "photon" 2)))))

(ert-deftest photon-readout-temperature-preserves-argmax ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (model (photon-make-model (photon-make-config 4 4 2 1)))
         (state (photon-vector-from-list '(1.0 0.0 0.0 0.0)))
         (raw nil)
         (scaled nil))
    (setcar (nthcdr 0 (photon-model-output-head model))
            (photon-vector-from-list '(0.0 0.0 0.0 0.0)))
    (setcar (nthcdr 1 (photon-model-output-head model))
            (photon-vector-from-list '(8.0 0.0 0.0 0.0)))
    (setcar (nthcdr 2 (photon-model-output-head model))
            (photon-vector-from-list '(2.0 0.0 0.0 0.0)))
    (setcar (nthcdr 3 (photon-model-output-head model))
            (photon-vector-from-list '(-1.0 0.0 0.0 0.0)))
    (setcar (nthcdr 0 (photon-model-target-prototypes model))
            (photon-vector-from-list '(0.0 0.0 0.0 0.0)))
    (setcar (nthcdr 1 (photon-model-target-prototypes model))
            (photon-vector-from-list '(0.0 0.0 0.0 0.0)))
    (setcar (nthcdr 2 (photon-model-target-prototypes model))
            (photon-vector-from-list '(0.0 0.0 0.0 0.0)))
    (setcar (nthcdr 3 (photon-model-target-prototypes model))
            (photon-vector-from-list '(0.0 0.0 0.0 0.0)))
    (setq raw (photon-model-readout-logits model state nil))
    (photon-model-param-set model 'readout-logit-temperature 4.0)
    (setq scaled (photon-model-readout-logits model state nil))
    (should (= (photon-argmax-index raw) (photon-argmax-index scaled)))
    (should (< (photon-target-margin scaled 1 2)
               (photon-target-margin raw 1 2)))))

(ert-deftest photon-wordpiece-split-runs-final-ngram-rescue ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (train-path (make-temp-file "photon-wordpiece-train" nil ".txt"))
         (eval-path (make-temp-file "photon-wordpiece-eval" nil ".txt")))
    (unwind-protect
        (progn
          (with-temp-file train-path
            (insert "photon stream abcabc photon stream\n"))
          (with-temp-file eval-path
            (insert "photon stream abcabc photon checked\n"))
          (let* ((result (photon-train-wordpiece-text-file-split
                          train-path eval-path 3 6 3 1 0.20 4))
                 (rescue (cdr (assq 'wordpiece-final-ngram-rescue result))))
            (should (assq 'fallback-after result))
            (should (consp (cdr (assq 'history rescue))))
            (should (assq 'accuracy-after-permil
                          (cdr (assq 'last rescue))))
            (should (assq 'wordpiece-long-rare-readout-rescue result))
            (should (assq 'full-readout-distill result))))
      (ignore-errors (delete-file train-path))
      (ignore-errors (delete-file eval-path)))))

(ert-deftest photon-wordpiece-diagnostics-report-fallback-misses ()
  (let* ((photon-vector-backend (photon-make-elisp-vector-backend))
         (text "photon stream\nnext photon\n")
         (result (photon-train-wordpiece-text text 2 6 2 1 0.20 4))
         (model (cdr (assq 'model result)))
         (tokens (cdr (assq 'tokens result)))
         (diagnostics (photon-wordpiece-diagnostics model tokens 2 4)))
    (should (> (cdr (assq 'total diagnostics)) 0))
    (should (assq 'summary diagnostics))
    (should (consp (cdr (assq 'token-kinds diagnostics))))
    (should (consp (cdr (assq 'token-shapes diagnostics))))
    (should (consp (cdr (assq 'readout-ablations diagnostics))))
    (should (consp (cdr (assq 'readout-component-shape-gaps diagnostics))))
    (should (consp (cdr (assq 'prototype-competition-shapes diagnostics))))
    (let ((competition
           (car (cdr (assq 'prototype-competition-shapes diagnostics)))))
      (should (assq 'average-target-rank competition))
      (should (assq 'average-target-score competition))
      (should (assq 'average-rival-score competition))
      (should (assq 'rival-shape-counts competition))
      (should (assq 'rival-token-counts competition))
      (should (assq 'top-rival-tokens competition))
      (when (cdr (assq 'top-rival-tokens competition))
        (let ((rival (car (cdr (assq 'top-rival-tokens competition)))))
          (should (assq 'token rival))
          (should (assq 'piece rival))
          (should (assq 'count rival)))))
    (should (assq 'unknown-targets (cdr (assq 'summary diagnostics))))
    (should (assq 'rare-targets (cdr (assq 'summary diagnostics))))
    (should (listp (cdr (assq 'fallback-miss-summary diagnostics))))
    (when (cdr (assq 'fallback-misses diagnostics))
      (let ((record (car (cdr (assq 'fallback-misses diagnostics)))))
        (should (assq 'index record))
        (should (assq 'target-piece record))
        (should (assq 'target-kind record))
        (should (assq 'target-count record))
        (should (assq 'target-rare record))
        (should (assq 'memory-hit record))
        (should (assq 'top-5 record))
        (should (assq 'readout-contributions record))))))

(ert-deftest photon-text-training-attaches-vocab-metadata ()
  (let* ((result (photon-train-text "abababababab" 4 8 4 6 0.20))
         (model (cdr (assq 'model result)))
         (vocab (photon-model-metadata-get model 'vocab))
         (generated (photon-model-generate-text model "ab" 4)))
    (should (equal vocab '("a" "b")))
    (should (stringp generated))
    (should (= (length generated) 6))))

(ert-deftest photon-model-generation-trace-reports-top-k ()
  (let* ((result (photon-train-text "abababababab" 4 8 4 2 0.20))
         (model (cdr (assq 'model result)))
         (trace-result
          (photon-model-generate-text-with-trace model "ab" 3))
         (trace (cdr (assq 'trace trace-result))))
    (should (= (length trace) 3))
    (should (stringp (cdr (assq 'generated trace-result))))
    (should (assq 'top-5 (car trace)))
    (should (assq 'readout-contributions (car trace)))
    (should (stringp (cdr (assq 'prediction-char (car trace)))))))

(ert-deftest photon-model-generation-trace-accepts-sampling-options ()
  (let* ((result (photon-train-text "abababababab" 4 8 4 2 0.20))
         (model (cdr (assq 'model result)))
         (options '((mode . sample) (temperature . 0.9) (top-k . 2)
                    (repetition-penalty . 1.1)))
         (trace-result
          (photon-model-generate-text-with-trace model "ab" 3 options))
         (trace (cdr (assq 'trace trace-result))))
    (should (equal (cdr (assq 'options trace-result)) options))
    (should (equal (cdr (assq 'options (car trace))) options))
    (should (eq (cdr (assq 'mode (car trace))) 'sample))
    (should (= (length trace) 3))))

(ert-deftest photon-read-text-file-loads-corpus ()
  (let ((path (make-temp-file "photon-corpus-" nil ".txt" "abcabc")))
    (unwind-protect
        (should (equal (photon-read-text-file path) "abcabc"))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-stream-text-file-trains-with-small-read-chunks ()
  (let ((path (make-temp-file "photon-stream-corpus-" nil ".txt"
                              "abcabc abcabc abcabc")))
    (unwind-protect
        (let* ((result (photon-train-text-file-stream path 3 6 3 1 0.20 4))
               (model (cdr (assq 'model result)))
               (before (cdr (assq 'before result)))
               (after (cdr (assq 'after result))))
          (should (equal (photon-model-metadata-get model 'tokenizer)
                         'char-stream))
          (should (= (photon-model-metadata-get model 'stream-chunk-bytes) 4))
          (should (> (cdr (assq 'total before)) 0))
          (should (> (cdr (assq 'total after)) 0)))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-stream-text-file-split-evaluates-heldout ()
  (let ((train-path (make-temp-file "photon-stream-train-" nil ".txt"
                                    "abcabc abcabc abcabc"))
        (eval-path (make-temp-file "photon-stream-eval-" nil ".txt"
                                   "bcabca bcabca bcabca")))
    (unwind-protect
        (let* ((result (photon-train-text-file-stream-split
                        train-path eval-path 3 6 3 1 0.20 4))
               (model (cdr (assq 'model result)))
               (after (cdr (assq 'after result)))
               (fallback-after (cdr (assq 'fallback-after result))))
          (should (equal (photon-model-metadata-get model 'tokenizer)
                         'char-stream))
          (should (photon-model-metadata-get model 'eval-path))
          (should (> (cdr (assq 'total after)) 0))
          (should (> (cdr (assq 'total fallback-after)) 0)))
      (when (file-exists-p train-path)
        (delete-file train-path))
      (when (file-exists-p eval-path)
        (delete-file eval-path)))))

(ert-deftest photon-stream-file-diagnostics-reports-ranking-and-classes ()
  (let ((train-path (make-temp-file "photon-stream-diag-train-" nil ".txt"
                                    "abc abc abc abc"))
        (eval-path (make-temp-file "photon-stream-diag-eval-" nil ".txt"
                                   "bca bca bca")))
    (unwind-protect
        (let* ((result (photon-train-text-file-stream-split
                        train-path eval-path 3 6 3 1 0.20 4))
               (model (cdr (assq 'model result)))
               (vocab (cdr (assq 'vocab result)))
               (diagnostics
                (photon-stream-file-diagnostics
                 model eval-path vocab 3 4 3)))
          (should (> (cdr (assq 'total diagnostics)) 0))
          (should (assq 'fallback-accuracy-permil diagnostics))
          (should (assq 'top-3-accuracy-permil diagnostics))
          (should (assq 'top-5-accuracy-permil diagnostics))
          (should (assq 'average-target-margin diagnostics))
          (should (assq 'memory-hits diagnostics))
          (should (assq 'newline-fallback-accuracy-permil diagnostics))
          (should (assq 'newline-top-3-accuracy-permil diagnostics))
          (should (assq 'newline-average-target-margin diagnostics))
          (should (assq 'summary diagnostics))
          (should (assq 'readout-ablations diagnostics))
          (should (assq 'readout-component-gaps diagnostics))
          (should (assq 'ngram-head-gap-summary diagnostics))
          (should (assq 'ngram-head-gaps diagnostics))
          (should (assq 'fallback-miss-summary diagnostics))
          (should (listp (cdr (assq 'newline-targets diagnostics))))
          (should (consp (cdr (assq 'char-classes diagnostics))))
          (let ((ablations (cdr (assq 'readout-ablations diagnostics)))
                (has-full nil)
                (has-linear-output-head nil)
                (has-output-head-prototype nil)
                (has-raw-output-head nil)
                (has-output-head nil)
                (has-raw-prototype nil)
                (has-prototype nil)
                (has-state-cluster nil)
                (has-embedding nil)
                (has-structural-off nil))
            (while ablations
              (when (eq (cdr (assq 'mode (car ablations))) 'full)
                (setq has-full t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'linear-output-head-only)
                (setq has-linear-output-head t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'output-head-prototype-only)
                (setq has-output-head-prototype t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'raw-output-head-only)
                (setq has-raw-output-head t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'output-head-only)
                (setq has-output-head t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'raw-prototype-only)
                (setq has-raw-prototype t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'prototype-only)
                (setq has-prototype t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'state-cluster-only)
                (setq has-state-cluster t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'embedding-anchor-only)
                (setq has-embedding t))
              (when (eq (cdr (assq 'mode (car ablations)))
                        'structural-off)
                (setq has-structural-off t))
              (setq ablations (cdr ablations)))
            (should has-full)
            (should has-linear-output-head)
            (should has-output-head-prototype)
            (should has-raw-output-head)
            (should has-output-head)
            (should has-raw-prototype)
            (should has-prototype)
            (should has-state-cluster)
            (should has-embedding)
            (should has-structural-off))
          (let ((gaps (cdr (assq 'readout-component-gaps diagnostics))))
            (should (consp gaps))
            (should (assq 'ngram-head-gap (car gaps)))
            (should (assq 'average-head-margin (car gaps))))
          (when (cdr (assq 'fallback-miss-summary diagnostics))
            (let* ((record (car (cdr (assq 'fallback-miss-summary
                                           diagnostics))))
                   (contributions
                    (cdr (assq 'readout-contributions record)))
                   (has-prototype-contribution nil))
              (should (assq 'readout-contributions record))
              (should (assq 'component (car contributions)))
              (while contributions
                (when (eq (cdr (assq 'component (car contributions)))
                          'prototype)
                  (setq has-prototype-contribution t))
                (setq contributions (cdr contributions)))
              (should has-prototype-contribution)))
          (should (listp (cdr (assq 'fallback-misses diagnostics)))))
      (when (file-exists-p train-path)
        (delete-file train-path))
      (when (file-exists-p eval-path)
        (delete-file eval-path)))))

(ert-deftest photon-model-generation-returns-extended-prompt ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (generated (photon-model-generate model '(1 2 3 4) 3)))
    (should (= (length generated) 7))))

(ert-deftest photon-save-load-round-trips-model ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (path (make-temp-file "photon-model-" nil ".el")))
    (unwind-protect
        (progn
          (photon-save-model model path)
          (should (equal model (photon-load-model path)))
          (with-temp-buffer
            (insert-file-contents path)
            (goto-char (point-min))
            (let ((data (read (current-buffer))))
              (should (eq (cdr (assq 'format data)) 'photon-model-v1))
              (should (assq 'version data)))))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-save-load-restores-vector-backend ()
  (let ((path (make-temp-file "photon-vector-model-" nil ".el")))
    (unwind-protect
        (progn
          (photon-use-elisp-vector-backend)
          (let ((model (photon-make-model (photon-make-config 8 8 4 2))))
            (photon-save-model model path))
          (photon-use-list-vector-backend)
          (let ((loaded (photon-load-model path)))
            (should (eq (photon-vector-backend-name) 'elisp-vector))
            (should (= (length (photon-model-generate loaded '(1 2 3 4) 2))
                       6))))
      (photon-use-list-vector-backend)
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-batch-options-checkpoint-and-resume ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (tokens '(1 2 3 4 1 2 3 4 1 2 3 4))
         (path (make-temp-file "photon-checkpoint-" nil ".el")))
    (unwind-protect
        (let ((result
               (photon-train-batch-options
                model tokens 4 1 0.20 2
                (list (cons 'shuffle t)
                      (cons 'checkpoint-path path)
                      (cons 'train-split-numerator 3)
                      (cons 'train-split-denominator 4)))))
          (should (file-exists-p path))
          (should (> (cdr (assq 'train-pairs result)) 0))
          (should (> (cdr (assq 'eval-pairs result)) 0))
          (let ((resumed
                 (photon-train-batch-options
                  model tokens 4 1 0.10 2
                  (list (cons 'resume t)
                        (cons 'checkpoint-path path)))))
            (should (assq 'model resumed))
            (should (consp (cdr (assq 'history resumed))))))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-training-history-round-trips ()
  (let* ((history (list (photon-training-history-entry
	                         0
	                         '((accuracy-permil . 250))
	                         '((accuracy-permil . 500)
                                   (fallback-accuracy-permil . 400)
                                   (loss . ((total . 1.0))))
	                         "checkpoint.el")))
         (path (make-temp-file "photon-history-" nil ".el")))
    (unwind-protect
        (progn
          (photon-save-training-history history path
                                        '((checkpoint-path . "checkpoint.el")))
          (should (equal history (photon-load-training-history path)))
          (should (= (photon-training-history-last-epoch history) 0)))
      (when (file-exists-p path)
        (delete-file path)))))

(ert-deftest photon-eval-better-prefers-fallback-on-loss-tie ()
  (let ((current '((accuracy-permil . 900)
                   (fallback-accuracy-permil . 500)
                   (loss . ((total-loss . 1000)))))
        (candidate '((accuracy-permil . 850)
                     (fallback-accuracy-permil . 625)
                     (loss . ((total-loss . 1000))))))
    (should (photon-eval-better-p candidate current))))

(ert-deftest photon-batch-dataset-loader-resumes-history-epochs ()
  (let ((data-path (make-temp-file "photon-batch-data-" nil ".txt"
                                   "abcd abcd abcd abcd"))
        (checkpoint-path (make-temp-file "photon-batch-checkpoint-" nil ".el"))
        (history-path (make-temp-file "photon-batch-history-" nil ".el")))
    (unwind-protect
        (let* ((dataset (photon-load-batch-dataset-text-file data-path 4))
               (config (photon-make-config
                        (length (cdr (assq 'vocab dataset))) 8 4 2))
               (model (photon-make-model config))
               (result
                (photon-train-batch-dataset
                 model dataset 2 0.20 2
                 (list (cons 'checkpoint-path checkpoint-path)
                       (cons 'history-path history-path))))
               (resumed
                (photon-train-batch-dataset
                 model dataset 1 0.10 2
                 (list (cons 'resume t)
                       (cons 'checkpoint-path checkpoint-path)
                       (cons 'history-path history-path))))
               (history (cdr (assq 'history resumed))))
          (should (file-exists-p checkpoint-path))
          (should (file-exists-p history-path))
          (should (> (cdr (assq 'token-count result)) 0))
          (should (= (car (nth 0 history)) 0))
	          (should (= (car (nth 1 history)) 1))
	          (should (= (car (nth 2 history)) 2))
	          (should (assq 'snapshot (cdr (nth 2 history))))
                  (should (assq 'eval-fallback-accuracy-permil
                                (cdr (assq 'snapshot
                                           (cdr (nth 2 history))))))
                  (should (assq 'epoch-elapsed-seconds
                                (cdr (assq 'snapshot
                                           (cdr (nth 2 history)))))))
      (when (file-exists-p data-path)
        (delete-file data-path))
      (when (file-exists-p checkpoint-path)
        (delete-file checkpoint-path))
      (when (file-exists-p history-path)
        (delete-file history-path)))))

(ert-deftest photon-batch-training-saves-best-checkpoint-and-csv-history ()
  (let ((data-path (make-temp-file "photon-best-data-" nil ".txt"
                                   "abcd abcd abcd abcd"))
        (checkpoint-path (make-temp-file "photon-best-checkpoint-" nil ".el"))
        (best-path (make-temp-file "photon-best-model-" nil ".el"))
        (history-path (make-temp-file "photon-best-history-" nil ".el"))
        (csv-path (make-temp-file "photon-best-history-" nil ".csv")))
    (unwind-protect
        (let* ((dataset (photon-load-batch-dataset-text-file data-path 4))
               (config (photon-make-config
                        (length (cdr (assq 'vocab dataset))) 8 4 2))
               (model (photon-make-model config))
               (result
                (photon-train-batch-dataset
                 model dataset 2 0.20 2
                 (list (cons 'checkpoint-path checkpoint-path)
                       (cons 'best-checkpoint-path best-path)
                       (cons 'history-path history-path)
                       (cons 'history-csv-path csv-path)))))
          (should (file-exists-p checkpoint-path))
          (should (file-exists-p best-path))
          (should (file-exists-p csv-path))
          (should (integerp (cdr (assq 'best-epoch result))))
          (with-temp-buffer
            (insert-file-contents csv-path)
	            (should (string-match-p "epoch,train_accuracy_permil"
	                                    (buffer-string)))
            (should (string-match-p "eval_fallback_accuracy_permil"
                                    (buffer-string)))
            (should (string-match-p "epoch_elapsed_seconds"
                                    (buffer-string)))
            (should (string-match-p "total_elapsed_seconds"
                                    (buffer-string)))
	            (should (string-match-p "\n0," (buffer-string)))))
      (when (file-exists-p data-path)
        (delete-file data-path))
      (when (file-exists-p checkpoint-path)
        (delete-file checkpoint-path))
      (when (file-exists-p best-path)
        (delete-file best-path))
      (when (file-exists-p history-path)
        (delete-file history-path))
      (when (file-exists-p csv-path)
        (delete-file csv-path)))))

(ert-deftest photon-wordpiece-best-checkpoint-restores-vocab-for-generation ()
  (let ((data-path (make-temp-file "photon-wp-best-data-" nil ".txt"
                                   "photon stream photon stream photon stream"))
        (checkpoint-path (make-temp-file "photon-wp-checkpoint-" nil ".el"))
        (best-path (make-temp-file "photon-wp-best-model-" nil ".el"))
        (history-path (make-temp-file "photon-wp-history-" nil ".el"))
        (csv-path (make-temp-file "photon-wp-history-" nil ".csv"))
        (vocab-path (make-temp-file "photon-wp-vocab-" nil ".el")))
    (delete-file vocab-path)
    (unwind-protect
        (let* ((dataset
                (photon-load-batch-dataset-text-file
                 data-path 4
                 (list (cons 'tokenizer 'wordpiece)
                       (cons 'vocab-path vocab-path)
                       (cons 'wordpiece-max-piece-size 4))))
               (config (photon-make-config
                        (length (cdr (assq 'vocab dataset))) 8 4 2))
               (model (photon-make-model config))
               (result
                (photon-train-batch-dataset
                 model dataset 2 0.20 2
                 (list (cons 'checkpoint-path checkpoint-path)
                       (cons 'best-checkpoint-path best-path)
                       (cons 'history-path history-path)
                       (cons 'history-csv-path csv-path))))
               (returned-model (cdr (assq 'model result)))
               (best-model (photon-load-model best-path))
               (saved-vocab (photon-load-vocab vocab-path)))
          (should (cdr (assq 'restored-best result)))
          (should (file-exists-p best-path))
          (should (file-exists-p vocab-path))
          (should (equal saved-vocab (cdr (assq 'vocab dataset))))
          (should (equal (photon-model-metadata-get best-model 'vocab)
                         saved-vocab))
          (should (assq 'unknown-token-count
                        (photon-model-metadata-get
                         best-model 'token-diagnostics)))
          (should (assq 'rare-token-count
                        (photon-model-metadata-get
                         best-model 'token-diagnostics)))
          (should (equal (photon-model-metadata-get returned-model 'vocab)
                         saved-vocab))
          (should (equal returned-model best-model))
          (should (stringp
                   (photon-model-generate-text
                    best-model "photon" 2 '((mode . greedy))))))
      (when (file-exists-p data-path)
        (delete-file data-path))
      (when (file-exists-p checkpoint-path)
        (delete-file checkpoint-path))
      (when (file-exists-p best-path)
        (delete-file best-path))
      (when (file-exists-p history-path)
        (delete-file history-path))
      (when (file-exists-p csv-path)
        (delete-file csv-path))
      (when (file-exists-p vocab-path)
        (delete-file vocab-path)))))

(ert-deftest photon-evaluate-pairs-does-not-update-memory ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (pairs (photon-training-pairs '(1 2 3 4 1 2 3 4) 4))
         (before (length (photon-model-next-token-memory model))))
    (photon-evaluate-pairs model pairs)
    (should (= (length (photon-model-next-token-memory model)) before))))

(ert-deftest photon-evaluate-pairs-reports-fallback-accuracy ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (pairs (photon-training-pairs '(1 2 3 4 1 2 3 4) 4))
         (result (photon-evaluate-pairs model pairs)))
    (should (assq 'fallback-accuracy-permil result))
    (should (assq 'fallback-top-3-accuracy-permil result))
    (should (assq 'fallback-top-5-accuracy-permil result))
    (should (assq 'memory-hit-permil result))))

(ert-deftest photon-evaluate-pairs-fallback-matches-standalone ()
  (let* ((config (photon-make-config 8 8 4 2))
         (model (photon-make-model config))
         (pairs (photon-training-pairs '(1 2 3 4 1 2 3 4) 4))
         (combined (photon-evaluate-pairs model pairs))
         (fallback (photon-evaluate-pairs-fallback model pairs))
         (fields '(fallback-correct
                   fallback-total
                   fallback-accuracy-permil
                   fallback-top-3-accuracy-permil
                   fallback-top-5-accuracy-permil
                   memory-hit-permil)))
    (while fields
      (should (equal (cdr (assq (car fields) combined))
                     (cdr (assq (car fields) fallback))))
      (setq fields (cdr fields)))))

(ert-deftest photon-multi-query-returns-token-id ()
  (let* ((config (photon-make-config 16 8 4 2))
         (token (photon-multi-query-next-token config '(1 3 5 7) 9)))
    (should (integerp token))
    (should (<= 0 token))
    (should (< token 16))))

(provide 'photon-test)

;;; photon-test.el ends here
