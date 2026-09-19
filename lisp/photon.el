;;; photon.el --- Small PHOTON-style model sketch -*- lexical-binding: t; -*-

;; This file intentionally stays close to basic Emacs Lisp so it can be loaded
;; by the current NeLisp runtime.

;;; NeLisp runtime math / hash compatibility shims.
;;
;; The standalone NeLisp binary does not expose the C math builtins
;; `exp' / `log' / `random' / `sxhash'.  Its stdlib `exp'/`log' wrappers
;; route through a JIT float bridge (`nl-jit-call-float-unary') and
;; `sxhash' through a JIT trampoline (`nl-jit-call-out-1'); neither is
;; wired in every standalone build, so calling them aborts evaluation.
;; Photon's text generation needs `exp' (softmax), `log' (log-sum-exp
;; perplexity) and `sxhash' (seeded sampler), so provide pure-elisp
;; fallbacks guarded by `fboundp'.  On real Emacs (and a fully wired
;; NeLisp) the native builtins win and these defuns never run.
(unless (fboundp 'exp)
  (defun exp (x)
    "Pure-elisp natural exponential (range reduction + Taylor series)."
    (let* ((x (float x))
           (ln2 0.6931471805599453)
           (k (floor (+ (/ x ln2) 0.5)))
           (r (- x (* k ln2)))
           (term 1.0) (sum 1.0) (n 1))
      (while (<= n 17)
        (setq term (/ (* term r) (float n)))
        (setq sum (+ sum term))
        (setq n (1+ n)))
      (let ((p 1.0) (m (abs k)) (i 0))
        (while (< i m) (setq p (* p 2.0)) (setq i (1+ i)))
        (if (< k 0) (/ sum p) (* sum p))))))

(unless (fboundp 'log)
  (defun log (x &optional base)
    "Pure-elisp natural logarithm (2^e scaling + atanh series).
With BASE, return the logarithm of X to that base."
    (let ((x (float x)))
      (if (<= x 0.0)
          -1.0e30
        (let ((e 0))
          (while (>= x 2.0) (setq x (/ x 2.0)) (setq e (1+ e)))
          (while (< x 1.0)  (setq x (* x 2.0)) (setq e (1- e)))
          (let* ((y (/ (- x 1.0) (+ x 1.0)))
                 (y2 (* y y)) (term y) (sum y) (n 3))
            (while (< n 31)
              (setq term (* term y2))
              (setq sum (+ sum (/ term (float n))))
              (setq n (+ n 2)))
            (let ((ln (+ (* 2.0 sum) (* (float e) 0.6931471805599453))))
              (if base (/ ln (log base)) ln))))))))

(defvar photon--compat-random-state 1
  "Linear-congruential state for the NeLisp `random' fallback.")
(unless (fboundp 'random)
  (defun random (&optional limit)
    "Pure-elisp LCG `random' fallback for the standalone NeLisp runtime."
    (setq photon--compat-random-state
          (mod (+ (* photon--compat-random-state 1103515245) 12345)
               2147483648))
    (if (and (integerp limit) (> limit 0))
        (mod photon--compat-random-state limit)
      photon--compat-random-state)))

(unless (fboundp 'sxhash)
  (defun sxhash (object)
    "Deterministic pure-elisp `sxhash' fallback for the NeLisp runtime.
Exact Emacs hash values are not reproduced; only determinism and a
reasonable distribution are required by Photon's seeded sampler."
    (cond
     ((null object) 0)
     ((integerp object) (mod (abs object) 1000000007))
     ((floatp object) (mod (abs (truncate (* object 1000.0))) 1000000007))
     ((symbolp object) (sxhash (symbol-name object)))
     ((stringp object)
      (let ((h 0) (i 0) (n (length object)))
        (while (< i n)
          (setq h (mod (+ (* h 131) (aref object i)) 1000000007))
          (setq i (1+ i)))
        h))
     ((consp object)
      (mod (+ (* (sxhash (car object)) 1000003)
              (* (sxhash (cdr object)) 131) 7)
           1000000007))
     ((vectorp object)
      (let ((h 0) (i 0) (n (length object)))
        (while (< i n)
          (setq h (mod (+ (* h 131) (sxhash (aref object i))) 1000000007))
          (setq i (1+ i)))
        h))
     (t 0))))

(defvar photon--io-read-result nil
  "Scratch holder used by `photon--read-sexp-string' on the NeLisp runtime.")

(defun photon--read-sexp-string (content)
  "Parse and return the first sexp in CONTENT.
Real Emacs uses `read-from-string' directly.  The standalone NeLisp
runtime's `read-from-string' is quadratic on multi-KB input, so route
CONTENT through its O(n) file loader by wrapping it as a `setq' form in
a temp file and `load'ing that."
  (if (fboundp 'current-buffer)
      (car (read-from-string content))
    (let ((wrapper (make-temp-file "photon-io")))
      (with-temp-file wrapper
        (insert "(setq photon--io-read-result '" content ")"))
      (setq photon--io-read-result nil)
      (load wrapper)
      (when (fboundp 'delete-file)
        (delete-file wrapper))
      photon--io-read-result)))

(load (expand-file-name "lisp/photon-vector.el"))

(defconst photon-version "0.99.0")

(defun photon-make-config (vocab-size hidden-size chunk-size levels)
  (list (cons 'vocab-size vocab-size)
        (cons 'hidden-size hidden-size)
        (cons 'chunk-size chunk-size)
        (cons 'levels levels)))

(defun photon-config-get (config key)
  (cdr (assq key config)))

(defun photon-range (count)
  (let ((result nil)
        (i 0))
    (while (< i count)
      (setq result (cons i result))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-token-embedding (config token)
  (let ((hidden-size (photon-config-get config 'hidden-size))
        (vocab-size (max 1 (photon-config-get config 'vocab-size)))
        (result nil)
        (i 0))
    (while (< i hidden-size)
      (let* ((raw (mod (+ (* (1+ token) (+ i 3)) (* 17 i)) vocab-size))
             (centered (- (/ (* 2.0 raw) vocab-size) 1.0)))
        (setq result (cons centered result)))
      (setq i (1+ i)))
    (photon-vector-from-list (nreverse result))))

(defun photon-logit-probe (config token)
  (photon-token-embedding config token))

(defun photon-make-embedding-table (config)
  (let ((vocab-size (photon-config-get config 'vocab-size))
        (rows nil)
        (token 0))
    (while (< token vocab-size)
      (setq rows (cons (photon-token-embedding config token) rows))
      (setq token (1+ token)))
    (nreverse rows)))

(defun photon-make-gain-vector (config value)
  (photon-vector-from-list
   (photon-repeat value (photon-config-get config 'hidden-size))))

(defun photon-make-projection-matrix (config diagonal off-diagonal)
  (let ((hidden-size (photon-config-get config 'hidden-size))
        (rows nil)
        (row 0))
    (while (< row hidden-size)
      (let ((cols nil)
            (col 0))
        (while (< col hidden-size)
          (setq cols (cons (if (= row col) diagonal off-diagonal) cols))
          (setq col (1+ col)))
        (setq rows (cons (photon-vector-from-list (nreverse cols)) rows)))
      (setq row (1+ row)))
    (nreverse rows)))

(defun photon-make-lowrank-projection (config)
  (let ((hidden-size (photon-config-get config 'hidden-size)))
    (list (cons 'kind 'low-rank)
          (cons 'rows hidden-size)
          (cons 'cols hidden-size)
          (cons 'diag (photon-make-gain-vector config 1.0))
          (cons 'left (photon-vector-zeros hidden-size))
          (cons 'right (photon-vector-zeros hidden-size)))))

(defun photon-projection-shape (projection)
  (if (and (consp projection) (eq (cdr (assq 'kind projection)) 'low-rank))
      (list (cdr (assq 'rows projection)) (cdr (assq 'cols projection)))
    (photon-matrix-shape projection)))

(defun photon-projection-apply-raw (projection vec)
  (if (and (consp projection) (eq (cdr (assq 'kind projection)) 'low-rank))
      (let* ((diag (cdr (assq 'diag projection)))
             (left (cdr (assq 'left projection)))
             (right (cdr (assq 'right projection)))
             (rank-activation (photon-vector-dot right vec)))
        (photon-vector-add
         (photon-vector-mul diag vec)
         (photon-vector-scale rank-activation left)))
    (photon-matrix-vector-dot projection vec)))

(defun photon-make-layer-projections (config)
  (list (cons 'chunk-proj (photon-make-lowrank-projection config))
        (cons 'context-proj (photon-make-lowrank-projection config))
        (cons 'converter-proj (photon-make-lowrank-projection config))
        (cons 'decoder-proj (photon-make-lowrank-projection config))))

(defun photon-make-layer-gains (config)
  (list (cons 'chunk-gain (photon-make-gain-vector config 1.0))
        (cons 'context-current-gain (photon-make-gain-vector config 1.0))
        (cons 'context-previous-gain (photon-make-gain-vector config 1.0))
        (cons 'converter-gain (photon-make-gain-vector config 1.0))
        (cons 'decoder-prev-gain (photon-make-gain-vector config 1.0))))

(defun photon-chunk-tokens (tokens chunk-size)
  (let ((chunks nil)
        (current nil)
        (n 0))
    (while tokens
      (setq current (cons (car tokens) current))
      (setq n (1+ n))
      (setq tokens (cdr tokens))
      (when (= n chunk-size)
        (setq chunks (cons (nreverse current) chunks))
        (setq current nil)
        (setq n 0)))
    (when current
      (setq chunks (cons (nreverse current) chunks)))
    (nreverse chunks)))

(defun photon-average-vectors (vectors hidden-size)
  (let ((sum (photon-vector-zeros hidden-size))
        (count 0))
    (while vectors
      (setq sum (photon-vector-add sum (car vectors)))
      (setq count (1+ count))
      (setq vectors (cdr vectors)))
    (if (= count 0)
        sum
      (photon-vector-scale (/ 1.0 count) sum))))

(defun photon-chunker (config chunk)
  (let ((embeddings nil)
        (hidden-size (photon-config-get config 'hidden-size)))
    (while chunk
      (setq embeddings
            (cons (photon-token-embedding config (car chunk)) embeddings))
      (setq chunk (cdr chunk)))
    (photon-vector-squash (photon-average-vectors embeddings hidden-size))))

(defun photon-model-token-embedding (model token)
  (let* ((cache-key (photon-token-cache-key token))
         (cached (photon-model-cache-get model cache-key)))
    (if cached
        cached
      (photon-model-cache-put
       model cache-key
       (let ((table (cdr (assq 'embedding-table model))))
         (if table
             (nth token table)
           (photon-token-embedding (photon-model-config model) token)))))))

(defun photon-model-chunker (model chunk)
  (let* ((cache-key (photon-chunk-cache-key chunk))
         (cached (photon-model-cache-get model cache-key)))
    (if cached
        cached
      (photon-model-cache-put
       model cache-key
       (let ((embeddings nil)
             (hidden-size (photon-config-get (photon-model-config model) 'hidden-size))
             (chunk-scale (photon-model-param-get model 'chunk-scale))
             (walk chunk))
         (while walk
           (setq embeddings
                 (cons (photon-model-token-embedding model (car walk)) embeddings))
           (setq walk (cdr walk)))
         (photon-vector-squash
          (photon-vector-mul
           (photon-model-gain-get model 'chunk-gain)
           (photon-model-project
            model 'chunk-proj
            (photon-vector-scale chunk-scale
                                 (photon-average-vectors embeddings hidden-size))))))))))

(defun photon-context-step (current previous)
  (photon-vector-squash
   (photon-vector-add (photon-vector-scale 0.72 current)
                      (photon-vector-scale 0.48 previous))))

(defun photon-context-encoder (config chunk-states)
  (let ((hidden-size (photon-config-get config 'hidden-size))
        (previous nil)
        (encoded nil))
    (setq previous (photon-vector-zeros hidden-size))
    (while chunk-states
      (setq previous (photon-context-step (car chunk-states) previous))
      (setq encoded (cons previous encoded))
      (setq chunk-states (cdr chunk-states)))
    (nreverse encoded)))

(defun photon-model-context-step (model current previous)
  (photon-vector-squash
   (photon-model-project
    model 'context-proj
    (photon-vector-add
     (photon-vector-mul
      (photon-model-gain-get model 'context-current-gain)
      (photon-vector-scale (photon-model-param-get model 'context-current) current))
     (photon-vector-mul
      (photon-model-gain-get model 'context-previous-gain)
      (photon-vector-scale (photon-model-param-get model 'context-previous) previous))))))

(defun photon-model-context-encoder (model chunk-states)
  (let ((hidden-size (photon-config-get (photon-model-config model) 'hidden-size))
        (previous nil)
        (encoded nil))
    (setq previous (photon-vector-zeros hidden-size))
    (while chunk-states
      (setq previous (photon-model-context-step model (car chunk-states) previous))
      (setq encoded (cons previous encoded))
      (setq chunk-states (cdr chunk-states)))
    (nreverse encoded)))

(defun photon-model-context-encoder-from (model chunk-states previous)
  (let ((encoded nil))
    (while chunk-states
      (setq previous (photon-model-context-step model (car chunk-states) previous))
      (setq encoded (cons previous encoded))
      (setq chunk-states (cdr chunk-states)))
    (nreverse encoded)))

(defun photon-context-converter (config context)
  (let ((chunk-size (photon-config-get config 'chunk-size))
        (result nil)
        (i 0))
    (while (< i chunk-size)
      (let ((bias (/ (float (1+ i)) (* 3.0 chunk-size))))
        (setq result
              (cons (photon-vector-squash
                     (photon-vector-add context
                                        (photon-vector-scale bias context)))
                    result)))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-model-context-converter (model context)
  (let* ((config (photon-model-config model))
         (chunk-size (photon-config-get config 'chunk-size))
         (converter-scale (photon-model-param-get model 'converter-scale))
         (result nil)
         (i 0))
    (while (< i chunk-size)
      (let ((bias (* converter-scale
                     (/ (float (1+ i)) (* 3.0 chunk-size)))))
        (setq result
              (cons (photon-vector-squash
                     (photon-vector-mul
                      (photon-model-gain-get model 'converter-gain)
                      (photon-model-project
                       model 'converter-proj
                       (photon-vector-add context
                                          (photon-vector-scale bias context)))))
                    result)))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-local-decode-chunk (config chunk context)
  (let ((conditioners (photon-context-converter config context))
        (previous (photon-vector-zeros (photon-config-get config 'hidden-size)))
        (states nil))
    (while conditioners
      (let* ((token (if chunk (car chunk) 0))
             (embedding (photon-token-embedding config token))
             (mixed (photon-vector-add
                     (photon-vector-add (car conditioners) embedding)
                     (photon-vector-scale 0.35 previous))))
        (setq previous (photon-vector-squash mixed))
        (setq states (cons previous states)))
      (setq conditioners (cdr conditioners))
      (when chunk
        (setq chunk (cdr chunk))))
    (nreverse states)))

(defun photon-model-local-decode-chunk (model chunk context)
  (let* ((config (photon-model-config model))
         (conditioners (photon-model-context-converter model context))
         (previous (photon-vector-zeros (photon-config-get config 'hidden-size)))
         (decoder-token-mix (photon-model-param-get model 'decoder-token-mix))
         (decoder-prev (photon-model-param-get model 'decoder-prev))
         (states nil))
    (while conditioners
      (let* ((token (if chunk (car chunk) 0))
             (embedding (photon-model-token-embedding model token))
             (mixed (photon-vector-add
                     (photon-vector-add
                      (car conditioners)
                      (photon-vector-scale decoder-token-mix embedding))
                     (photon-vector-mul
                      (photon-model-gain-get model 'decoder-prev-gain)
                      (photon-model-project
                       model 'decoder-proj
                       (photon-vector-scale decoder-prev previous))))))
        (setq previous (photon-vector-squash mixed))
        (setq states (cons previous states)))
      (setq conditioners (cdr conditioners))
      (when chunk
        (setq chunk (cdr chunk))))
    (nreverse states)))

(defun photon-token-logits (config state)
  (let ((vocab-size (photon-config-get config 'vocab-size))
        (token 0)
        (logits nil))
    (while (< token vocab-size)
      (setq logits
            (cons (photon-vector-dot state (photon-logit-probe config token))
                  logits))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-seed-weight (row col)
  (let* ((raw (mod (+ (* 1103515245 (+ row 1))
                      (* 12345 (+ col 7)))
                   997))
         (centered (- (/ (* 2.0 raw) 997.0) 1.0)))
    (* 0.05 centered)))

(defun photon-make-output-head (config)
  (let ((vocab-size (photon-config-get config 'vocab-size))
        (hidden-size (photon-config-get config 'hidden-size))
        (rows nil)
        (token 0))
    (while (< token vocab-size)
      (let ((cols nil)
            (col 0))
        (while (< col hidden-size)
          (setq cols (cons (photon-seed-weight token col) cols))
          (setq col (1+ col)))
        (setq rows (cons (photon-vector-from-list (nreverse cols)) rows)))
      (setq token (1+ token)))
    (nreverse rows)))

(defun photon-make-output-head-bias (config)
  (photon-repeat 0.0 (photon-config-get config 'vocab-size)))

(defun photon-make-output-head-prototype-bias (config)
  (photon-repeat 0.0 (photon-config-get config 'vocab-size)))

(defun photon-make-output-head-prototype-score-bonus (config)
  (photon-repeat 0.0 (photon-config-get config 'vocab-size)))

(defun photon-make-output-head-prototypes (_config)
  nil)

(defun photon-make-output-head-negative-prototypes (_config)
  nil)

(defun photon-make-output-head-context-bonus-prototypes (_config)
  nil)

(defun photon-make-output-head-projection (config)
  (photon-make-lowrank-projection config))

(defun photon-make-reconstruction-head (config)
  (let ((vocab-size (photon-config-get config 'vocab-size))
        (hidden-size (photon-config-get config 'hidden-size))
        (rows nil)
        (token 0))
    (while (< token vocab-size)
      (let ((cols nil)
            (embedding (photon-vector-to-list (photon-token-embedding config token)))
            (col 0))
        (while (< col hidden-size)
          (setq cols
                (cons (+ (* 0.90 (car embedding))
                         (photon-seed-weight token col))
                      cols))
          (setq embedding (cdr embedding))
          (setq col (1+ col)))
        (setq rows (cons (photon-vector-from-list (nreverse cols)) rows)))
      (setq token (1+ token)))
    (nreverse rows)))

(defun photon-make-target-prototypes (config)
  (let ((vocab-size (photon-config-get config 'vocab-size))
        (hidden-size (photon-config-get config 'hidden-size))
        (rows nil)
        (token 0))
    (while (< token vocab-size)
      (setq rows (cons (photon-vector-zeros hidden-size) rows))
      (setq token (1+ token)))
    (nreverse rows)))

(defun photon-make-target-counts (config)
  (photon-repeat 0 (photon-config-get config 'vocab-size)))

(defun photon-make-architecture-params ()
  (list (cons 'chunk-scale 1.0)
        (cons 'context-current 0.72)
        (cons 'context-previous 0.48)
        (cons 'converter-scale 1.0)
        (cons 'decoder-prev 0.35)
        (cons 'decoder-token-mix 1.0)
        (cons 'context-anchor-enabled 1.0)
        (cons 'context-anchor-recency 0.70)
        (cons 'context-anchor-prefix-weight 1.0)
        (cons 'context-anchor-ngram-enabled 1.0)
        (cons 'context-anchor-ngram-size 3.0)
        (cons 'context-anchor-ngram-weight 2.0)
        (cons 'wordpiece-context-anchor-ngram-weight 6.0)
        (cons 'wordpiece-final-context-anchor-ngram-weight 16.0)
        (cons 'readout-logit-temperature 1.0)
        (cons 'wordpiece-final-readout-temperature 8.0)
        (cons 'wordpiece-long-token-update-weight 0.35)
        (cons 'wordpiece-rare-token-update-weight 0.50)
        (cons 'wordpiece-update-scale-max 2.50)
        (cons 'wordpiece-readout-bias-enabled 0.0)
        (cons 'wordpiece-long-readout-bias-weight 0.75)
        (cons 'wordpiece-rare-readout-bias-weight 0.50)
        (cons 'wordpiece-token-readout-bias-weight 0.35)
        (cons 'wordpiece-readout-bias-max 2.50)
        (cons 'wordpiece-ngram-rescue-weight 1.00)
        (cons 'wordpiece-ngram-rescue-epochs 4.0)
        (cons 'wordpiece-final-ngram-rescue-weight 1.00)
        (cons 'wordpiece-final-ngram-rescue-epochs 3.0)
        (cons 'wordpiece-long-rare-readout-rescue-weight 0.75)
        (cons 'wordpiece-long-rare-readout-rescue-epochs 2.0)
        (cons 'wordpiece-long-teacher-state-path-weight 0.0)
        (cons 'wordpiece-continuation-teacher-state-path-weight 0.0)
        (cons 'wordpiece-start-long-teacher-state-path-weight 0.0)
        (cons 'wordpiece-start-long-teacher-token-mix 0.0)
        (cons 'wordpiece-start-long-teacher-context-mix 0.0)
        (cons 'wordpiece-start-long-teacher-anchor-mix 0.0)
        (cons 'wordpiece-start-long-boundary-readout-enabled 0.0)
        (cons 'wordpiece-start-long-boundary-readout-weight 0.0)
        (cons 'wordpiece-start-long-boundary-readout-size 4.0)
        (cons 'wordpiece-start-long-boundary-ngram-mix 0.0)
        (cons 'output-head-full-readout-distill-start-long-boundary-teacher-weight
              0.0)
        (cons 'wordpiece-start-long-classifier-enabled 0.0)
        (cons 'wordpiece-start-long-classifier-learning-rate-scale 1.0)
        (cons 'wordpiece-start-long-classifier-candidate-search-enabled
              0.0)
        (cons 'wordpiece-start-long-classifier-candidate-learning-rate 0.20)
        (cons 'wordpiece-start-long-classifier-candidate-top-k 10.0)
        (cons 'wordpiece-start-long-classifier-candidate-margin 0.10)
        (cons 'wordpiece-start-long-classifier-candidate-limit 64.0)
        (cons 'output-head-full-readout-distill-start-long-classifier-teacher-weight
              0.0)
        (cons 'output-head-full-readout-distill-start-long-ngram-teacher-weight
              0.0)
        (cons 'wordpiece-start-long-classifier-readout-weight 0.0)
        (cons 'context-anchor-ngram-contrastive-enabled 1.0)
        (cons 'context-anchor-ngram-contrastive-weight 0.25)
        (cons 'context-anchor-weight 0.0)
        (cons 'context-anchor-weight-max 1.0)
        (cons 'char-class-bias-enabled 1.0)
        (cons 'char-class-bias-weight 0.75)
        (cons 'char-boundary-bias-enabled 1.0)
        (cons 'char-boundary-bias-weight 1.0)
        (cons 'char-boundary-bias-size 6.0)
        (cons 'structural-boundary-bias-enabled 1.0)
        (cons 'structural-boundary-bias-weight 0.10)
        (cons 'structural-boundary-bias-size 6.0)
        (cons 'structural-bias-letter-gate-scale 0.0)
        (cons 'structural-bias-space-gate-scale 0.75)
        (cons 'structural-tie-breaker-enabled 1.0)
        (cons 'structural-tie-breaker-weight 2.0)
        (cons 'newline-boundary-bonus 4.20)
        (cons 'generation-repeat-bias-weight 1.25)
        (cons 'generation-newline-run-bias-weight 3.0)
        (cons 'generation-space-run-bias-weight 1.0)
        (cons 'wordpiece-generation-repeat-bias-weight 2.50)
        (cons 'wordpiece-generation-continuation-run-bias-weight 1.50)
        (cons 'wordpiece-generation-recent-token-bias-weight 0.40)
        (cons 'wordpiece-generation-recent-token-window 8.0)
        (cons 'wordpiece-generation-ngram-bias-weight 0.0)
        (cons 'wordpiece-generation-ngram-bias-gate-margin -1.0)
        (cons 'wordpiece-generation-ngram-bias-repeat-scale 1.0)
        (cons 'wordpiece-generation-ngram-bias-repeat-window 24.0)
        (cons 'wordpiece-generation-repeat-ngram-bias-weight 0.0)
        (cons 'wordpiece-generation-repeat-ngram-bias-min-count 1.0)
        (cons 'wordpiece-generation-repeat-ngram-window 24.0)
        (cons 'sampling-repetition-penalty 1.10)
        (cons 'next-token-memory-enabled 1.0)
        (cons 'next-token-margin 0.25)
        (cons 'output-head-aux-update-weight 0.50)
        (cons 'output-head-anchor-distill-weight 2.50)
        (cons 'output-head-cluster-distill-weight 1.00)
        (cons 'output-head-cluster-distill-epochs 5.0)
        (cons 'output-head-prototype-weight 8.0)
        (cons 'output-head-prototype-limit 32.0)
        (cons 'output-head-prototype-support-weight 0.0)
        (cons 'output-head-prototype-support-limit 3.0)
        (cons 'output-head-prototype-negative-weight 0.0)
        (cons 'output-head-prototype-negative-limit 32.0)
        (cons 'output-head-prototype-long-piece-weight 1.0)
        (cons 'output-head-prototype-short-piece-weight 1.0)
        (cons 'output-head-prototype-special-piece-weight 1.0)
        (cons 'output-head-prototype-start-long-piece-weight 1.0)
        (cons 'output-head-prototype-continuation-long-piece-weight 1.0)
        (cons 'output-head-prototype-start-short-piece-weight 1.0)
        (cons 'output-head-prototype-continuation-short-piece-weight 1.0)
        (cons 'output-head-prototype-diverse-trim-enabled 0.0)
        (cons 'output-head-prototype-projection-enabled 1.0)
        (cons 'wordpiece-final-output-head-prototype-projection-enabled 0.0)
        (cons 'output-head-linear-readout-weight 0.0)
        (cons 'wordpiece-final-output-head-linear-readout-weight 1.0)
        (cons 'output-head-compress-weight 0.40)
        (cons 'output-head-compress-epochs 12.0)
        (cons 'output-head-token-readout-distill-weight 0.50)
        (cons 'output-head-token-readout-distill-epochs 3.0)
        (cons 'output-head-full-readout-distill-weight 2.00)
        (cons 'output-head-full-readout-distill-epochs 8.0)
        (cons 'output-head-full-readout-distill-wordpiece-scale-enabled 0.0)
        (cons 'output-head-full-readout-distill-start-long-scale 1.0)
        (cons 'output-head-full-readout-distill-continuation-long-scale 1.0)
        (cons 'output-head-full-readout-distill-start-short-scale 1.0)
        (cons 'output-head-full-readout-distill-continuation-short-scale 1.0)
        (cons 'output-head-full-readout-distill-selection-start-long-weight 0.0)
        (cons 'output-head-prototype-readout-distill-epochs 1.0)
        (cons 'output-head-prototype-ngram-long-distill-epochs 1.0)
        (cons 'output-head-prototype-ngram-long-bias-weight 0.0)
        (cons 'output-head-prototype-ngram-long-token-mix 0.0)
        (cons 'output-head-prototype-ngram-long-context-mix 0.0)
        (cons 'output-head-prototype-ngram-long-anchor-mix 0.0)
        (cons 'output-head-prototype-ngram-long-contrastive-rate 0.02)
        (cons 'output-head-prototype-ngram-long-state-path-rate 0.0)
        (cons 'output-head-prototype-context-bonus-weight 0.0)
        (cons 'output-head-prototype-context-bonus-limit 16.0)
        (cons 'output-head-start-long-positive-distill-weight 0.0)
        (cons 'output-head-start-long-positive-distill-epochs 1.0)
        (cons 'output-head-start-long-positive-distill-top-k 3.0)
        (cons 'output-head-start-long-projected-prototype-distill-weight 0.0)
        (cons 'output-head-start-long-projected-prototype-distill-epochs 1.0)
        (cons 'output-head-start-long-projected-prototype-distill-top-k 1.0)
        (cons 'output-head-start-long-prototype-merge-rate 0.0)
        (cons 'output-head-start-long-prototype-merge-epochs 1.0)
        (cons 'output-head-start-long-prototype-merge-top-k 3.0)
        (cons 'output-head-start-long-prototype-merge-margin 0.10)
        (cons 'output-head-start-long-prototype-merge-target-bias-rate 0.0)
        (cons 'output-head-start-long-prototype-merge-rival-bias-rate 0.0)
        (cons 'output-head-start-long-prototype-merge-score-bonus-rate 0.0)
        (cons 'output-head-start-long-prototype-merge-dominant-rival-bias-rate
              0.0)
        (cons 'output-head-start-long-prototype-merge-dominant-rival-min-count
              2.0)
        (cons 'output-head-start-long-prototype-merge-state-separation-rate
              0.0)
        (cons 'output-head-start-long-prototype-merge-dense-head-rate 0.0)
        (cons 'output-head-start-long-prototype-merge-rival-line-search-step
              0.0)
        (cons 'output-head-start-long-prototype-merge-rival-line-search-max
              0.0)
        (cons 'output-head-start-long-prototype-merge-rival-negative-weight
              0.0)
        (cons 'output-head-start-long-prototype-merge-rivals 1.0)
        (cons 'output-head-start-long-prototype-merge-pocket-enabled 0.0)
        (cons 'output-head-start-long-prototype-merge-candidate-search-enabled
              0.0)
        (cons 'output-head-start-long-prototype-merge-candidate-limit 64.0)
        (cons 'output-head-start-long-prototype-merge-global-tolerance 0.0)
        (cons 'output-head-start-long-projection-separation-rate 0.0)
        (cons 'output-head-start-long-projection-separation-epochs 1.0)
        (cons 'output-head-start-long-projection-separation-top-k 3.0)
        (cons 'output-head-start-long-projection-separation-margin 0.10)
        (cons 'output-head-start-long-projection-separation-score-bonus-rate
              0.0)
        (cons 'output-head-start-long-projection-separation-rival-bias-rate
              0.0)
        (cons 'output-head-start-long-projection-separation-pocket-enabled 0.0)
        (cons 'output-head-start-long-projection-separation-global-tolerance
              0.0)
        (cons 'output-head-start-long-rival-negative-distill-weight 0.0)
        (cons 'output-head-start-long-rival-negative-distill-epochs 1.0)
        (cons 'output-head-start-long-rival-negative-distill-top-k 3.0)
        (cons 'output-head-start-long-rival-negative-distill-rivals 1.0)
        (cons 'output-head-start-long-same-shape-margin-distill-weight 0.0)
        (cons 'output-head-start-long-same-shape-margin-distill-bias-weight
              0.0)
        (cons 'output-head-start-long-same-shape-margin-distill-epochs 1.0)
        (cons 'output-head-start-long-same-shape-margin-distill-top-k 3.0)
        (cons 'output-head-start-long-same-shape-margin-distill-rivals 1.0)
        (cons 'output-head-start-long-same-shape-margin-distill-margin 0.10)
        (cons 'output-head-start-long-same-shape-margin-distill-adaptive-bias-max
              0.0)
        (cons 'output-head-start-long-same-shape-top1-search-enabled 0.0)
        (cons 'output-head-start-long-same-shape-top1-search-score-bonus-max
              0.0)
        (cons 'output-head-start-long-same-shape-top1-search-context-bonus-weight
              0.0)
        (cons 'output-head-start-long-same-shape-top1-search-global-tolerance
              0.0)
        (cons 'output-head-ngram-distill-weight 0.50)
        (cons 'prototype-ngram-distill-weight 1.00)
        (cons 'prototype-weight 0.75)
        (cons 'output-head-state-cluster-readout-weight 1.0)
        (cons 'state-cluster-readout-weight 1.0)
        (cons 'state-cluster-limit 64.0)
        (cons 'next-token-weight 1.0)
        (cons 'reconstruction-weight 0.05)
        (cons 'next-context-weight 0.02)))

(defun photon-make-model (config)
  (list (cons 'config config)
        (cons 'architecture-params (photon-make-architecture-params))
        (cons 'layer-gains (photon-make-layer-gains config))
        (cons 'layer-projections (photon-make-layer-projections config))
        (cons 'embedding-table (photon-make-embedding-table config))
        (cons 'output-head (photon-make-output-head config))
        (cons 'output-head-bias (photon-make-output-head-bias config))
        (cons 'start-long-classifier-head (photon-make-output-head config))
        (cons 'start-long-classifier-bias
              (photon-make-output-head-bias config))
        (cons 'output-head-prototype-bias
              (photon-make-output-head-prototype-bias config))
        (cons 'output-head-prototype-score-bonus
              (photon-make-output-head-prototype-score-bonus config))
        (cons 'output-head-prototypes
              (photon-make-output-head-prototypes config))
        (cons 'output-head-negative-prototypes
              (photon-make-output-head-negative-prototypes config))
        (cons 'output-head-context-bonus-prototypes
              (photon-make-output-head-context-bonus-prototypes config))
        (cons 'output-head-projection
              (photon-make-output-head-projection config))
        (cons 'reconstruction-head (photon-make-reconstruction-head config))
        (cons 'target-prototypes (photon-make-target-prototypes config))
        (cons 'target-counts (photon-make-target-counts config))
        (cons 'target-state-clusters nil)
        (cons 'context-anchor-ngrams nil)
        (cons 'start-long-boundary-readouts nil)
        (cons 'char-class-biases nil)
        (cons 'char-boundary-biases nil)
        (cons 'structural-boundary-biases nil)
        (cons 'next-token-memory nil)
        (cons 'next-token-memory-limit 512)
        (cons 'cache nil)
        (cons 'cache-stats (photon-make-cache-stats))
        (cons 'cache-limit 512)
        (cons 'metadata nil)))

(defun photon-model-config (model)
  (cdr (assq 'config model)))

(defun photon-model-output-head (model)
  (cdr (assq 'output-head model)))

(defun photon-model-output-head-bias (model)
  (let ((cell (assq 'output-head-bias model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-bias (photon-model-config model)))))

(defun photon-model-start-long-classifier-head (model)
  (let ((cell (assq 'start-long-classifier-head model)))
    (if cell
        (cdr cell)
      (photon-make-output-head (photon-model-config model)))))

(defun photon-model-start-long-classifier-bias (model)
  (let ((cell (assq 'start-long-classifier-bias model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-bias (photon-model-config model)))))

(defun photon-model-output-head-prototypes (model)
  (let ((cell (assq 'output-head-prototypes model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-prototypes (photon-model-config model)))))

(defun photon-model-output-head-negative-prototypes (model)
  (let ((cell (assq 'output-head-negative-prototypes model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-negative-prototypes
       (photon-model-config model)))))

(defun photon-model-output-head-context-bonus-prototypes (model)
  (let ((cell (assq 'output-head-context-bonus-prototypes model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-context-bonus-prototypes
       (photon-model-config model)))))

(defun photon-model-output-head-prototype-bias (model)
  (let ((cell (assq 'output-head-prototype-bias model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-prototype-bias (photon-model-config model)))))

(defun photon-model-output-head-prototype-score-bonus (model)
  (let ((cell (assq 'output-head-prototype-score-bonus model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-prototype-score-bonus
       (photon-model-config model)))))

(defun photon-model-output-head-projection (model)
  (let ((cell (assq 'output-head-projection model)))
    (if cell
        (cdr cell)
      (photon-make-output-head-projection (photon-model-config model)))))

(defun photon-model-reconstruction-head (model)
  (let ((cell (assq 'reconstruction-head model)))
    (if cell
        (cdr cell)
      (photon-make-reconstruction-head (photon-model-config model)))))

(defun photon-model-target-prototypes (model)
  (let ((cell (assq 'target-prototypes model)))
    (if cell
        (cdr cell)
      (photon-make-target-prototypes (photon-model-config model)))))

(defun photon-model-target-counts (model)
  (let ((cell (assq 'target-counts model)))
    (if cell
        (cdr cell)
      (photon-make-target-counts (photon-model-config model)))))

(defun photon-model-target-state-clusters (model)
  (let ((cell (assq 'target-state-clusters model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-context-anchor-ngrams (model)
  (let ((cell (assq 'context-anchor-ngrams model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-start-long-boundary-readouts (model)
  (let ((cell (assq 'start-long-boundary-readouts model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-char-class-biases (model)
  (let ((cell (assq 'char-class-biases model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-char-boundary-biases (model)
  (let ((cell (assq 'char-boundary-biases model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-structural-boundary-biases (model)
  (let ((cell (assq 'structural-boundary-biases model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-next-token-memory (model)
  (let ((cell (assq 'next-token-memory model)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-model-next-token-memory-limit (model)
  (let ((cell (assq 'next-token-memory-limit model)))
    (if cell
        (cdr cell)
      512)))

(defun photon-model-next-token-memory-enabled-p (model)
  (> (photon-model-param-get model 'next-token-memory-enabled) 0.0))

(defun photon-model-cache (model)
  (cdr (assq 'cache model)))

(defun photon-model-set-cache (model cache)
  (let ((cell (assq 'cache model)))
    (if cell
        (setcdr cell cache)
      (setq model (append model (list (cons 'cache cache))))))
  model)

(defun photon-model-cache-limit (model)
  (let ((cell (assq 'cache-limit model)))
    (if cell
        (cdr cell)
      512)))

(defun photon-cache-trim (cache limit)
  (if (or (null limit) (<= limit 0))
      cache
    (let ((result nil)
          (walk cache)
          (i 0))
      (while (and walk (< i limit))
        (setq result (cons (car walk) result))
        (setq walk (cdr walk))
        (setq i (1+ i)))
      (nreverse result))))

(defun photon-cache-remove-key (cache key)
  (let ((result nil))
    (while cache
      (unless (equal (car (car cache)) key)
        (setq result (cons (car cache) result)))
      (setq cache (cdr cache)))
    (nreverse result)))

(defun photon-model-set-cache-limit (model limit)
  (let ((cell (assq 'cache-limit model)))
    (if cell
        (setcdr cell limit)
      (setq model (append model (list (cons 'cache-limit limit))))))
  (photon-model-set-cache model
                          (photon-cache-trim (photon-model-cache model) limit))
  model)

(defun photon-model-clear-cache (model)
  (photon-model-cache-stats-increment model 'clears nil)
  (photon-model-set-cache model nil))

(defun photon-cache-key (kind name vec)
  (list kind name (photon-vector-fingerprint vec)))

(defun photon-token-cache-key (token)
  (cons 'token-embedding token))

(defun photon-chunk-cache-key (chunk)
  (cons 'chunk-state (copy-sequence chunk)))

(defun photon-window-cache-key (tokens)
  (cons 'forward-window (copy-sequence tokens)))

(defun photon-prediction-cache-key (tokens)
  (cons 'prediction-window (copy-sequence tokens)))

(defun photon-next-token-memory-key (tokens)
  (cons 'next-token-memory (copy-sequence tokens)))

(defun photon-incremental-forward-cache-key ()
  (cons 'incremental-forward 'last))

(defun photon-cache-key-kind (key)
  (if (consp key)
      (car key)
    'unknown))

(defun photon-make-cache-stats ()
  (list (cons 'hits 0)
        (cons 'misses 0)
        (cons 'puts 0)
        (cons 'clears 0)
        (cons 'by-kind nil)))

(defun photon-model-cache-stats (model)
  (let ((cell (assq 'cache-stats model)))
    (if cell
        (cdr cell)
      (photon-make-cache-stats))))

(defun photon-model-set-cache-stats (model stats)
  (let ((cell (assq 'cache-stats model)))
    (if cell
        (setcdr cell stats)
      (setq model (append model (list (cons 'cache-stats stats))))))
  model)

(defun photon-alist-increment (alist key)
  (let ((cell (assq key alist)))
    (if cell
        (setcdr cell (1+ (cdr cell)))
      (setq alist (cons (cons key 1) alist)))
    alist))

(defun photon-model-cache-stats-increment (model field kind)
  (let* ((stats (photon-model-cache-stats model))
         (by-kind-cell (assq 'by-kind stats))
         (by-kind (cdr by-kind-cell)))
    (setq stats (photon-alist-increment stats field))
    (when kind
      (let* ((kind-cell (assq kind by-kind))
             (kind-stats (if kind-cell
                             (cdr kind-cell)
                           nil)))
        (setq kind-stats (photon-alist-increment kind-stats field))
        (if kind-cell
            (setcdr kind-cell kind-stats)
          (setq by-kind (cons (cons kind kind-stats) by-kind)))
        (if by-kind-cell
            (setcdr by-kind-cell by-kind)
          (setq stats (cons (cons 'by-kind by-kind) stats)))))
    (photon-model-set-cache-stats model stats)))

(defun photon-cache-stats-get (stats field)
  (let ((cell (assq field stats)))
    (if cell
        (cdr cell)
      0)))

(defun photon-cache-stats-kind-get (stats kind field)
  (let* ((by-kind (cdr (assq 'by-kind stats)))
         (kind-stats (cdr (assq kind by-kind))))
    (photon-cache-stats-get kind-stats field)))

(defun photon-model-reset-cache-stats (model)
  (photon-model-set-cache-stats model (photon-make-cache-stats)))

(defun photon-model-cache-get (model key)
  (let ((cell (assoc key (photon-model-cache model)))
        (kind (photon-cache-key-kind key)))
    (if cell
        (progn
          (photon-model-cache-stats-increment model 'hits kind)
          (cdr cell))
      (photon-model-cache-stats-increment model 'misses kind)
      nil)))

(defun photon-model-cache-put (model key value)
  (photon-model-cache-stats-increment model 'puts (photon-cache-key-kind key))
  (photon-model-set-cache
   model
   (photon-cache-trim
    (cons (cons key value)
          (photon-cache-remove-key (photon-model-cache model) key))
    (photon-model-cache-limit model)))
  value)

(defun photon-model-architecture-params (model)
  (cdr (assq 'architecture-params model)))

(defun photon-model-layer-gains (model)
  (cdr (assq 'layer-gains model)))

(defun photon-model-layer-projections (model)
  (cdr (assq 'layer-projections model)))

(defun photon-model-set-layer-projections (model projections)
  (let ((cell (assq 'layer-projections model)))
    (if cell
        (setcdr cell projections)
      (setq model (append model (list (cons 'layer-projections projections))))))
  model)

(defun photon-model-projection-get (model key)
  (let ((cell (assq key (photon-model-layer-projections model))))
    (if cell
        (cdr cell)
      (cdr (assq key (photon-make-layer-projections (photon-model-config model)))))))

(defun photon-model-projection-set (model key value)
  (let* ((projections (photon-model-layer-projections model))
         (cell (assq key projections)))
    (if cell
        (setcdr cell value)
      (setq projections (cons (cons key value) projections)))
    (photon-model-set-layer-projections model projections)
    (photon-model-clear-cache model)))

(defun photon-model-project (model key vec)
  (let* ((cache-key (photon-cache-key 'projection key vec))
         (cached (photon-model-cache-get model cache-key)))
    (if cached
        cached
      (photon-model-cache-put
       model cache-key
       (photon-projection-apply-raw
        (photon-model-projection-get model key)
        vec)))))

(defun photon-model-set-layer-gains (model gains)
  (let ((cell (assq 'layer-gains model)))
    (if cell
        (setcdr cell gains)
      (setq model (append model (list (cons 'layer-gains gains))))))
  model)

(defun photon-model-gain-get (model key)
  (let ((cell (assq key (photon-model-layer-gains model))))
    (if cell
        (cdr cell)
      (cdr (assq key (photon-make-layer-gains (photon-model-config model)))))))

(defun photon-model-gain-set (model key value)
  (let* ((gains (photon-model-layer-gains model))
         (cell (assq key gains)))
    (if cell
        (setcdr cell value)
      (setq gains (cons (cons key value) gains)))
    (photon-model-set-layer-gains model gains)
    (photon-model-clear-cache model)))

(defun photon-model-set-architecture-params (model params)
  (let ((cell (assq 'architecture-params model)))
    (if cell
        (setcdr cell params)
      (setq model (append model (list (cons 'architecture-params params))))))
  model)

(defun photon-model-param-get (model key)
  (let ((cell (assq key (photon-model-architecture-params model))))
    (if cell
        (cdr cell)
      (cdr (assq key (photon-make-architecture-params))))))

(defun photon-clamp (value low high)
  (cond ((< value low) low)
        ((> value high) high)
        (t value)))

(defun photon-model-param-set (model key value)
  (let* ((params (photon-model-architecture-params model))
         (cell (assq key params)))
    (if cell
        (setcdr cell value)
      (setq params (cons (cons key value) params)))
    (photon-model-set-architecture-params model params)
    (photon-model-clear-cache model)))

(defun photon-model-param-add (model key delta low high)
  (photon-model-param-set
   model key
   (photon-clamp (+ (photon-model-param-get model key) delta) low high)))

(defun photon-model-embedding-table (model)
  (cdr (assq 'embedding-table model)))

(defun photon-model-set-embedding-table (model embedding-table)
  (let ((cell (assq 'embedding-table model)))
    (if cell
        (setcdr cell embedding-table)
      (setq model (append model (list (cons 'embedding-table embedding-table))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-metadata (model)
  (cdr (assq 'metadata model)))

(defun photon-model-set-metadata (model metadata)
  (let ((cell (assq 'metadata model)))
    (if cell
        (setcdr cell metadata)
      (setq model (append model (list (cons 'metadata metadata))))))
  model)

(defun photon-model-metadata-get (model key)
  (cdr (assq key (photon-model-metadata model))))

(defun photon-model-metadata-put (model key value)
  (let* ((metadata (photon-model-metadata model))
         (cell (assq key metadata)))
    (if cell
        (setcdr cell value)
      (setq metadata (cons (cons key value) metadata)))
    (photon-model-set-metadata model metadata)))

(defun photon-model-set-output-head (model output-head)
  (let ((cell (assq 'output-head model)))
    (setcdr cell output-head))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-bias (model output-head-bias)
  (let ((cell (assq 'output-head-bias model)))
    (if cell
        (setcdr cell output-head-bias)
      (setq model
            (append model (list (cons 'output-head-bias
                                      output-head-bias))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-start-long-classifier-head (model head)
  (let ((cell (assq 'start-long-classifier-head model)))
    (if cell
        (setcdr cell head)
      (setq model
            (append model
                    (list (cons 'start-long-classifier-head head))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-start-long-classifier-bias (model bias)
  (let ((cell (assq 'start-long-classifier-bias model)))
    (if cell
        (setcdr cell bias)
      (setq model
            (append model
                    (list (cons 'start-long-classifier-bias bias))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-prototypes (model prototypes)
  (let ((cell (assq 'output-head-prototypes model)))
    (if cell
        (setcdr cell prototypes)
      (setq model
            (append model (list (cons 'output-head-prototypes
                                      prototypes))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-negative-prototypes (model prototypes)
  (let ((cell (assq 'output-head-negative-prototypes model)))
    (if cell
        (setcdr cell prototypes)
      (setq model
            (append model (list (cons 'output-head-negative-prototypes
                                      prototypes))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-context-bonus-prototypes
    (model prototypes)
  (let ((cell (assq 'output-head-context-bonus-prototypes model)))
    (if cell
        (setcdr cell prototypes)
      (setq model
            (append model
                    (list (cons 'output-head-context-bonus-prototypes
                                prototypes))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-prototype-bias (model bias)
  (let ((cell (assq 'output-head-prototype-bias model)))
    (if cell
        (setcdr cell bias)
      (setq model
            (append model (list (cons 'output-head-prototype-bias bias))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-prototype-score-bonus (model bonus)
  (let ((cell (assq 'output-head-prototype-score-bonus model)))
    (if cell
        (setcdr cell bonus)
      (setq model
            (append model
                    (list (cons 'output-head-prototype-score-bonus
                                bonus))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-output-head-projection (model projection)
  (let ((cell (assq 'output-head-projection model)))
    (if cell
        (setcdr cell projection)
      (setq model
            (append model (list (cons 'output-head-projection
                                      projection))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-reconstruction-head (model reconstruction-head)
  (let ((cell (assq 'reconstruction-head model)))
    (if cell
        (setcdr cell reconstruction-head)
      (setq model
            (append model (list (cons 'reconstruction-head
                                      reconstruction-head))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-target-prototypes (model target-prototypes)
  (let ((cell (assq 'target-prototypes model)))
    (if cell
        (setcdr cell target-prototypes)
      (setq model
            (append model (list (cons 'target-prototypes
                                      target-prototypes))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-target-counts (model target-counts)
  (let ((cell (assq 'target-counts model)))
    (if cell
        (setcdr cell target-counts)
      (setq model
            (append model (list (cons 'target-counts target-counts))))))
  model)

(defun photon-model-set-target-state-clusters (model clusters)
  (let ((cell (assq 'target-state-clusters model)))
    (if cell
        (setcdr cell clusters)
      (setq model
            (append model (list (cons 'target-state-clusters clusters))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-context-anchor-ngrams (model ngrams)
  (let ((cell (assq 'context-anchor-ngrams model)))
    (if cell
        (setcdr cell ngrams)
      (setq model
            (append model (list (cons 'context-anchor-ngrams ngrams))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-start-long-boundary-readouts (model readouts)
  (let ((cell (assq 'start-long-boundary-readouts model)))
    (if cell
        (setcdr cell readouts)
      (setq model
            (append model
                    (list (cons 'start-long-boundary-readouts
                                readouts))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-char-class-biases (model biases)
  (let ((cell (assq 'char-class-biases model)))
    (if cell
        (setcdr cell biases)
      (setq model
            (append model (list (cons 'char-class-biases biases))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-char-boundary-biases (model biases)
  (let ((cell (assq 'char-boundary-biases model)))
    (if cell
        (setcdr cell biases)
      (setq model
            (append model (list (cons 'char-boundary-biases biases))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-structural-boundary-biases (model biases)
  (let ((cell (assq 'structural-boundary-biases model)))
    (if cell
        (setcdr cell biases)
      (setq model
            (append model (list (cons 'structural-boundary-biases
                                      biases))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-set-next-token-memory (model memory)
  (let ((cell (assq 'next-token-memory model)))
    (if cell
        (setcdr cell memory)
      (setq model (append model (list (cons 'next-token-memory memory))))))
  (photon-model-clear-cache model)
  model)

(defun photon-model-next-token-memory-raw-get (model context)
  (cdr (assoc (photon-next-token-memory-key context)
              (photon-model-next-token-memory model))))

(defun photon-model-next-token-memory-raw-hit-p (model context)
  (if (assoc (photon-next-token-memory-key context)
             (photon-model-next-token-memory model))
      t
    nil))

(defun photon-model-next-token-memory-get (model context)
  (if (photon-model-next-token-memory-enabled-p model)
      (photon-model-next-token-memory-raw-get model context)
    nil))

(defun photon-model-next-token-memory-hit-p (model context)
  (if (photon-model-next-token-memory-enabled-p model)
      (photon-model-next-token-memory-raw-hit-p model context)
    nil))

(defun photon-model-remember-next-token (model context target)
  (when (photon-model-next-token-memory-enabled-p model)
    (let* ((key (photon-next-token-memory-key context))
           (memory (photon-model-next-token-memory model))
           (cell (assoc key memory)))
      (if cell
          (setcdr cell target)
        (setq memory (cons (cons key target) memory)))
      (photon-model-set-next-token-memory
       model
       (photon-cache-trim memory
                          (photon-model-next-token-memory-limit model))))))

(defun photon-nth-set (items index value)
  (let ((result nil)
        (i 0))
    (while items
      (setq result (cons (if (= i index) value (car items)) result))
      (setq items (cdr items))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-model-output-head-logits (model state)
  (let ((head (photon-model-output-head model))
        (bias (photon-model-output-head-bias model))
        (logits nil))
    (while head
      (setq logits
            (cons (+ (photon-vector-dot state (car head))
                     (or (car bias) 0.0))
                  logits))
      (setq head (cdr head)))
      (setq bias (cdr bias))
    (nreverse logits)))

(defun photon-model-target-prototype-logits (model state)
  (let ((prototypes (photon-model-target-prototypes model))
        (prototype-weight (photon-model-param-get model 'prototype-weight))
        (logits nil))
    (while prototypes
      (setq logits
            (cons (* prototype-weight
                     (photon-vector-dot state (car prototypes)))
                  logits))
      (setq prototypes (cdr prototypes)))
    (nreverse logits)))

(defun photon-model-target-state-cluster-limit (model)
  (max 1 (truncate (photon-model-param-get model 'state-cluster-limit))))

(defun photon-target-state-cluster-trim (states limit)
  (if (or (null limit) (<= limit 0))
      states
    (let ((result nil)
          (walk states)
          (i 0))
      (while (and walk (< i limit))
        (setq result (cons (car walk) result))
        (setq walk (cdr walk))
        (setq i (1+ i)))
      (nreverse result))))

(defun photon-state-min-distance (state selected)
  (let ((best nil)
        (walk selected))
    (while walk
      (let ((distance (photon-context-state-distance state (car walk))))
        (when (or (null best) (< distance best))
          (setq best distance)))
      (setq walk (cdr walk)))
    (or best 0.0)))

(defun photon-diverse-state-trim (states limit)
  (if (or (null limit) (<= limit 0) (<= (length states) limit))
      states
    (let ((selected (list (car states)))
          (remaining (cdr states)))
      (while (and remaining (< (length selected) limit))
        (let ((best nil)
              (best-score nil)
              (walk remaining))
          (while walk
            (let ((score (photon-state-min-distance (car walk) selected)))
              (when (or (null best-score) (> score best-score))
                (setq best-score score)
                (setq best (car walk))))
            (setq walk (cdr walk)))
          (setq selected (append selected (list best)))
          (setq remaining (delq best remaining))))
      selected)))

(defun photon-model-target-state-cluster-logits (model state)
  (let ((clusters (photon-model-target-state-clusters model))
        (vocab-size (photon-config-get (photon-model-config model)
                                       'vocab-size))
        (scale (photon-model-param-get model 'state-cluster-readout-weight))
        (token 0)
        (logits nil))
    (while (< token vocab-size)
      (let* ((cell (assq token clusters))
             (states (cdr cell))
             (best nil))
        (while states
          (let ((score (- (photon-context-state-distance state
                                                         (car states)))))
            (when (or (null best) (> score best))
              (setq best score)))
          (setq states (cdr states)))
        (setq logits (cons (if best (* scale best) -1000.0) logits)))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-model-output-head-prototype-limit (model)
  (max 1 (truncate
          (photon-model-param-get model 'output-head-prototype-limit))))

(defun photon-model-output-head-prototype-support-limit (model)
  (max 1 (truncate
          (photon-model-param-get model
                                  'output-head-prototype-support-limit))))

(defun photon-model-output-head-prototype-negative-limit (model)
  (max 1 (truncate
          (photon-model-param-get model
                                  'output-head-prototype-negative-limit))))

(defun photon-model-output-head-prototype-context-bonus-limit (model)
  (max 1 (truncate
          (photon-model-param-get
           model 'output-head-prototype-context-bonus-limit))))

(defun photon-prototype-insert-top-score (scores score limit)
  (let ((result nil)
        (inserted nil)
        (remaining scores))
    (while remaining
      (when (and (not inserted) (> score (car remaining)))
        (setq result (cons score result))
        (setq inserted t))
      (setq result (cons (car remaining) result))
      (setq remaining (cdr remaining)))
    (unless inserted
      (setq result (cons score result)))
    (photon-take (nreverse result) limit)))

(defun photon-output-head-prototype-supported-score (scores support-weight)
  (let ((best (car scores))
        (support nil)
        (count 0)
        (tail (cdr scores)))
    (while tail
      (setq support (+ (or support 0.0) (car tail)))
      (setq count (1+ count))
      (setq tail (cdr tail)))
    (if (and support (> count 0))
        (+ best (* support-weight (/ support count)))
      best)))

(defun photon-output-head-prototype-token-kind-weight (model token)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (if (and vocab (eq (photon-model-metadata-get model 'tokenizer)
                       'wordpiece))
        (let ((shape (photon-wordpiece-token-shape vocab token))
              (kind (photon-wordpiece-token-kind vocab token)))
          (cond
           ((eq shape 'start-long-piece)
            (* (photon-model-param-get
                model 'output-head-prototype-long-piece-weight)
               (photon-model-param-get
                model 'output-head-prototype-start-long-piece-weight)))
           ((eq shape 'continuation-long-piece)
            (* (photon-model-param-get
                model 'output-head-prototype-long-piece-weight)
               (photon-model-param-get
                model
                'output-head-prototype-continuation-long-piece-weight)))
           ((eq shape 'start-short-piece)
            (* (photon-model-param-get
                model 'output-head-prototype-short-piece-weight)
               (photon-model-param-get
                model 'output-head-prototype-start-short-piece-weight)))
           ((eq shape 'continuation-short-piece)
            (* (photon-model-param-get
                model 'output-head-prototype-short-piece-weight)
               (photon-model-param-get
                model
                'output-head-prototype-continuation-short-piece-weight)))
           ((eq kind 'long-piece)
            (photon-model-param-get
             model 'output-head-prototype-long-piece-weight))
           ((eq kind 'short-piece)
            (photon-model-param-get
             model 'output-head-prototype-short-piece-weight))
           ((eq kind 'special)
            (photon-model-param-get
             model 'output-head-prototype-special-piece-weight))
           (t 1.0)))
      1.0)))

(defun photon-output-head-prototype-negative-scale (model token)
  (let ((scale (photon-model-param-get
                model 'output-head-prototype-negative-weight))
        (vocab (photon-model-metadata-get model 'vocab)))
    (if (and vocab
             (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
             (eq (photon-wordpiece-token-shape vocab token)
                 'start-long-piece))
	        (+ scale
	           (photon-model-param-get
	            model 'output-head-start-long-rival-negative-distill-weight)
	           (photon-model-param-get
	            model
	            'output-head-start-long-prototype-merge-rival-negative-weight))
      scale)))

(defun photon-output-head-prototype-best-supported-score
    (model state states support-limit support-weight)
  (let ((top-scores nil))
    (while states
      (let* ((prototype-state
              (photon-model-output-head-projected-state model
                                                        (car states)))
             (score (/ 1.0
                       (+ 1.0
                          (photon-context-state-distance
                           state prototype-state)))))
        (setq top-scores
              (photon-prototype-insert-top-score
               top-scores score support-limit)))
      (setq states (cdr states)))
    (when top-scores
      (photon-output-head-prototype-supported-score
       top-scores support-weight))))

(defun photon-output-head-prototype-best-state (model state token)
  (let* ((prototypes (photon-model-output-head-prototypes model))
         (states (cdr (assq token prototypes)))
         (best-state nil)
         (best-score nil))
    (while states
      (let* ((raw-state (car states))
             (prototype-state
              (photon-model-output-head-projected-state model raw-state))
             (score (/ 1.0
                       (+ 1.0
                          (photon-context-state-distance
                           state prototype-state)))))
        (when (or (null best-score) (> score best-score))
          (setq best-score score)
          (setq best-state raw-state)))
      (setq states (cdr states)))
    best-state))

(defun photon-output-head-prototype-token-detail
    (model projected-state token)
  (let* ((prototypes (photon-model-output-head-prototypes model))
         (cell (assq token prototypes))
         (states (cdr cell))
         (support-limit
          (photon-model-output-head-prototype-support-limit model))
         (support-weight
          (photon-model-param-get model
                                  'output-head-prototype-support-weight))
         (score
          (photon-output-head-prototype-best-supported-score
           model projected-state states support-limit support-weight))
         (bias (or (nth token (photon-model-output-head-prototype-bias model))
                   0.0))
         (score-bonus
          (or (nth token
                   (photon-model-output-head-prototype-score-bonus model))
              0.0))
         (kind-weight
          (photon-output-head-prototype-token-kind-weight model token))
         (scale (max 0.0
                     (+ (photon-model-param-get
                         model 'output-head-prototype-weight)
                        score-bonus))))
    (list (cons 'prototype-count (length states))
          (cons 'positive-score (or score 0.0))
          (cons 'positive-distance
                (if score (- (/ 1.0 score) 1.0) nil))
          (cons 'bias bias)
          (cons 'score-bonus score-bonus)
          (cons 'kind-weight kind-weight)
          (cons 'weighted-positive
                (if score (* scale kind-weight score) 0.0)))))

(defun photon-update-output-head-prototype-bias (model target amount)
  (let* ((bias (photon-model-output-head-prototype-bias model))
         (current (or (nth target bias) 0.0)))
    (setq bias (photon-nth-set bias target (+ current amount)))
    (photon-model-set-output-head-prototype-bias model bias)))

(defun photon-update-output-head-prototype-score-bonus (model target amount)
  (let* ((bonus (photon-model-output-head-prototype-score-bonus model))
         (current (or (nth target bonus) 0.0)))
    (setq bonus (photon-nth-set bonus target (+ current amount)))
    (photon-model-set-output-head-prototype-score-bonus model bonus)))

(defun photon-update-output-head-context-bonus-prototype
    (model state target)
  (let* ((prototypes
          (photon-model-output-head-context-bonus-prototypes model))
         (cell (assq target prototypes))
         (limit (photon-model-output-head-prototype-context-bonus-limit
                 model))
         (states (if cell (cdr cell) nil)))
    (setq states
          (photon-target-state-cluster-trim (cons state states) limit))
    (if cell
        (setcdr cell states)
      (setq prototypes (cons (cons target states) prototypes)))
    (photon-model-set-output-head-context-bonus-prototypes
     model prototypes)))

(defun photon-model-context-average-embedding (model context)
  (let ((sum nil)
        (count 0)
        (walk context))
    (while walk
      (let ((embedding (photon-model-token-embedding model (car walk))))
        (setq sum (if sum
                      (photon-vector-add sum embedding)
                    embedding))
        (setq count (1+ count)))
      (setq walk (cdr walk)))
    (if (and sum (> count 0))
        (photon-vector-scale (/ 1.0 count) sum)
      (photon-vector-zeros
       (photon-config-get (photon-model-config model) 'hidden-size)))))

(defun photon-ngram-long-output-head-prototype-state
    (model state context target)
  (let ((token-mix
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-token-mix))
        (context-mix
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-context-mix))
        (anchor-mix
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-anchor-mix))
        (result state))
    (when (> token-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              token-mix
              (photon-model-token-embedding model target)))))
    (when (> context-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              context-mix
              (photon-model-context-average-embedding model context)))))
    (when (> anchor-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              anchor-mix
              (photon-context-anchor-state model context)))))
    (photon-vector-squash result)))

(defun photon-model-output-head-projected-state (model state)
  (if (> (photon-model-param-get
          model 'output-head-prototype-projection-enabled)
         0.0)
      (photon-vector-squash
       (photon-projection-apply-raw
        (photon-model-output-head-projection model)
        state))
    state))

(defun photon-model-output-head-prototype-logits (model state)
  (let ((projected-state
         (photon-model-output-head-projected-state model state))
        (prototypes (photon-model-output-head-prototypes model))
        (negative-prototypes
         (photon-model-output-head-negative-prototypes model))
        (context-bonus-prototypes
         (photon-model-output-head-context-bonus-prototypes model))
        (vocab-size (photon-config-get (photon-model-config model)
                                       'vocab-size))
        (scale (photon-model-param-get model 'output-head-prototype-weight))
        (bias (photon-model-output-head-prototype-bias model))
        (score-bonus (photon-model-output-head-prototype-score-bonus model))
        (support-weight
         (photon-model-param-get model
                                 'output-head-prototype-support-weight))
        (support-limit
         (photon-model-output-head-prototype-support-limit model))
        (negative-limit
         (photon-model-output-head-prototype-negative-limit model))
        (context-bonus-weight
         (photon-model-param-get
          model 'output-head-prototype-context-bonus-weight))
        (context-bonus-limit
         (photon-model-output-head-prototype-context-bonus-limit model))
        (token 0)
        (logits nil))
    (while (< token vocab-size)
      (let* ((cell (assq token prototypes))
             (states (cdr cell))
             (negative-cell (assq token negative-prototypes))
             (negative-states (cdr negative-cell))
             (context-bonus-cell
              (assq token context-bonus-prototypes))
             (context-bonus-states (cdr context-bonus-cell))
             (positive-score
              (photon-output-head-prototype-best-supported-score
               model projected-state states support-limit support-weight))
             (negative-score
              (photon-output-head-prototype-best-supported-score
               model projected-state negative-states negative-limit 0.0))
             (context-bonus-score
              (and (> context-bonus-weight 0.0)
                   (photon-output-head-prototype-best-supported-score
                    model projected-state context-bonus-states
                    context-bonus-limit 0.0)))
             (token-scale
              (max 0.0 (+ scale (or (nth token score-bonus) 0.0))))
             (negative-scale
              (photon-output-head-prototype-negative-scale model token)))
        (setq logits
              (cons
               (-
                (+ (or (nth token bias) 0.0)
                   (if positive-score
                       (* token-scale
                          (photon-output-head-prototype-token-kind-weight
                           model token)
                          positive-score)
                     0.0)
                   (if context-bonus-score
                       (* context-bonus-weight context-bonus-score)
                     0.0))
                (if negative-score
                    (* negative-scale negative-score)
                  0.0))
               logits)))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-update-output-head-projection (model state learning-rate)
  (let ((projection (copy-tree (photon-model-output-head-projection model)))
        (delta (photon-vector-scale learning-rate state)))
    (if (and (consp projection)
             (eq (cdr (assq 'kind projection)) 'low-rank))
        (progn
          (setcdr (assq 'left projection)
                  (photon-vector-squash
                   (photon-vector-add (cdr (assq 'left projection))
                                      delta)))
          (setcdr (assq 'right projection)
                  (photon-vector-squash
                   (photon-vector-add (cdr (assq 'right projection))
                                      delta)))
          (setcdr (assq 'diag projection)
                  (photon-vector-squash
                   (photon-vector-add
                    (cdr (assq 'diag projection))
                    (photon-vector-scale (* 0.10 learning-rate) state)))))
      (setq projection
            (photon-make-output-head-projection
             (photon-model-config model))))
    (photon-model-set-output-head-projection model projection)))

(defun photon-update-output-head-projection-direction
    (model direction learning-rate)
  (let ((projection (copy-tree (photon-model-output-head-projection model)))
        (delta (photon-vector-scale learning-rate direction)))
    (if (and (consp projection)
             (eq (cdr (assq 'kind projection)) 'low-rank))
        (progn
          (setcdr (assq 'left projection)
                  (photon-vector-squash
                   (photon-vector-add (cdr (assq 'left projection))
                                      delta)))
          (setcdr (assq 'right projection)
                  (photon-vector-squash
                   (photon-vector-add (cdr (assq 'right projection))
                                      delta)))
          (setcdr (assq 'diag projection)
                  (photon-vector-squash
                   (photon-vector-add
                    (cdr (assq 'diag projection))
                    (photon-vector-scale (* 0.10 learning-rate)
                                         direction)))))
      (setq projection
            (photon-make-output-head-projection
             (photon-model-config model))))
    (photon-model-set-output-head-projection model projection)))

(defun photon-update-output-head-prototype (model state target)
  (let* ((prototypes (photon-model-output-head-prototypes model))
         (cell (assq target prototypes))
         (limit (photon-model-output-head-prototype-limit model))
         (states (if cell (cdr cell) nil)))
    (setq states
          (if (> (photon-model-param-get
                  model 'output-head-prototype-diverse-trim-enabled)
                 0.0)
              (photon-diverse-state-trim (cons state states) limit)
            (photon-target-state-cluster-trim
             (cons state states)
             limit)))
    (if cell
        (setcdr cell states)
      (setq prototypes (cons (cons target states) prototypes)))
    (photon-model-set-output-head-prototypes model prototypes)))

(defun photon-update-output-head-projected-prototype (model state target)
  (photon-update-output-head-prototype
   model
   (photon-model-output-head-projected-state model state)
   target))

(defun photon-update-output-head-negative-prototype (model state target)
  (let* ((prototypes (photon-model-output-head-negative-prototypes model))
         (cell (assq target prototypes))
         (limit (photon-model-output-head-prototype-negative-limit model))
         (states (if cell (cdr cell) nil)))
    (setq states
          (if (> (photon-model-param-get
                  model 'output-head-prototype-diverse-trim-enabled)
                 0.0)
              (photon-diverse-state-trim (cons state states) limit)
            (photon-target-state-cluster-trim
             (cons state states)
             limit)))
    (if cell
        (setcdr cell states)
      (setq prototypes (cons (cons target states) prototypes)))
    (photon-model-set-output-head-negative-prototypes model prototypes)))

(defun photon-model-dense-prototype-logits (model state)
  (photon-model-target-state-cluster-logits model state))

(defun photon-model-output-head-state-cluster-logits (model state)
  (let ((scale (photon-model-param-get
                model 'output-head-state-cluster-readout-weight)))
    (photon-logits-add-scaled
     (photon-logits-zeros model)
     (photon-model-target-state-cluster-logits model state)
     scale)))

(defun photon-model-dense-output-head-logits (model state)
  (photon-model-output-head-state-cluster-logits model state))

(defun photon-model-raw-output-head-logits (model state)
  (photon-logits-add-scaled
   (photon-model-output-head-prototype-logits model state)
   (photon-model-output-head-logits model state)
   (photon-model-param-get model 'output-head-linear-readout-weight)))

(defun photon-model-token-logits (model state)
  (photon-logits-add-scaled
   (photon-model-output-head-logits model state)
   (photon-model-target-prototype-logits model state)
   1.0))

(defun photon-logits-add-scaled (left right scale)
  (let ((result nil))
    (while (and left right)
      (setq result (cons (+ (car left) (* scale (car right))) result))
      (setq left (cdr left))
      (setq right (cdr right)))
    (nreverse result)))

(defun photon-logits-zeros (model)
  (photon-repeat 0.0 (photon-config-get (photon-model-config model)
                                        'vocab-size)))

(defun photon-sum-numbers (items)
  (let ((sum 0.0))
    (while items
      (setq sum (+ sum (car items)))
      (setq items (cdr items)))
    sum))

(defun photon-context-suffix (context size)
  (let* ((length (length context))
         (drop (- length size)))
    (when (< drop 0)
      (setq drop 0))
    (copy-sequence (nthcdr drop context))))

(defun photon-context-anchor-ngram-key (suffix)
  (cons 'context-anchor-ngram (copy-sequence suffix)))

(defun photon-context-anchor-ngram-enabled-p (model)
  (> (photon-model-param-get model 'context-anchor-ngram-enabled) 0.0))

(defun photon-start-long-boundary-readout-enabled-p (model)
  (> (photon-model-param-get
      model 'wordpiece-start-long-boundary-readout-enabled)
     0.0))

(defun photon-start-long-classifier-enabled-p (model)
  (> (photon-model-param-get model 'wordpiece-start-long-classifier-enabled)
     0.0))

(defun photon-context-anchor-ngram-contrastive-enabled-p (model)
  (> (photon-model-param-get
      model 'context-anchor-ngram-contrastive-enabled)
     0.0))

(defun photon-context-anchor-ngram-max-size (model)
  (max 1 (truncate (photon-model-param-get model 'context-anchor-ngram-size))))

(defun photon-start-long-boundary-readout-max-size (model)
  (max 1
       (truncate
        (photon-model-param-get
         model 'wordpiece-start-long-boundary-readout-size))))

(defun photon-context-anchor-ngram-logits (model context)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (max-size (photon-context-anchor-ngram-max-size model))
         (size 1)
         (logits (photon-repeat 0.0 vocab-size))
         (ngrams (photon-model-context-anchor-ngrams model)))
    (while (<= size max-size)
      (when (>= (length context) size)
        (let* ((suffix (photon-context-suffix context size))
               (counts (cdr (assoc (photon-context-anchor-ngram-key suffix)
                                   ngrams)))
               (total (if counts (photon-sum-numbers counts) 0.0))
               (scale (/ (float size) (float max-size)))
               (index 0)
               (walk counts))
          (when (> total 0.0)
            (while walk
              (setq logits
                    (photon-nth-set
                     logits index
                     (+ (nth index logits)
                        (* scale (/ (car walk) total)))))
              (setq index (1+ index))
              (setq walk (cdr walk))))))
      (setq size (1+ size)))
    logits))

(defun photon-start-long-token-p (model token)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (and vocab
         (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
         (eq (photon-wordpiece-token-shape vocab token)
             'start-long-piece))))

(defun photon-model-start-long-classifier-logits (model state)
  (let ((head (photon-model-start-long-classifier-head model))
        (bias (photon-model-start-long-classifier-bias model))
        (token 0)
        (logits nil))
    (while head
      (setq logits
            (cons
             (if (photon-start-long-token-p model token)
                 (+ (photon-vector-dot state (car head))
                    (or (car bias) 0.0))
               -1000.0)
             logits))
      (setq head (cdr head))
      (setq bias (cdr bias))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-update-start-long-classifier
    (model state prediction target learning-rate)
  (when (and (photon-start-long-classifier-enabled-p model)
             (photon-start-long-token-p model target)
             (/= prediction target))
    (let* ((rate (* learning-rate
                    (photon-model-param-get
                     model 'wordpiece-start-long-classifier-learning-rate-scale)
                    (photon-wordpiece-update-scale model target)))
           (head (photon-model-start-long-classifier-head model))
           (bias (photon-model-start-long-classifier-bias model))
           (target-row (nth target head))
           (prediction-row (nth prediction head))
           (delta (photon-vector-scale rate state)))
      (setq head
            (photon-nth-set head target
                            (photon-vector-add target-row delta)))
      (setq head
            (photon-nth-set
             head prediction
             (photon-vector-add prediction-row
                                (photon-vector-scale -1.0 delta))))
      (setq bias (photon-nth-set bias target (+ (nth target bias) rate)))
      (setq bias
            (photon-nth-set bias prediction
                            (- (nth prediction bias) rate)))
      (photon-model-set-start-long-classifier-head model head)
      (photon-model-set-start-long-classifier-bias model bias)
      t)))

(defun photon-evaluate-start-long-classifier-token-reader
    (model next-token context-size)
  (let ((correct 0)
        (top-3-correct 0)
        (top-5-correct 0)
        (total 0)
        (rank-sum 0.0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (when (photon-start-long-token-p model token)
          (let* ((states (photon-forward-model model window))
                 (state (photon-prediction-state states window))
                 (logits
                  (photon-model-start-long-classifier-logits model state))
                 (prediction (photon-argmax-index logits))
                 (rank (photon-logit-rank logits token)))
            (when (= prediction token)
              (setq correct (1+ correct)))
            (when (<= rank 3)
              (setq top-3-correct (1+ top-3-correct)))
            (when (<= rank 5)
              (setq top-5-correct (1+ top-5-correct)))
            (setq rank-sum (+ rank-sum rank))
            (setq total (1+ total))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (list (cons 'shape 'start-long-piece)
          (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'top-3-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
          (cons 'top-5-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
          (cons 'average-target-rank
                (if (= total 0) 0.0 (/ rank-sum total))))))

(defun photon-start-long-classifier-candidates-token-reader
    (model next-token context-size)
  (let ((candidates nil)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (window nil)
        (filled 0)
        (token nil)
        (top-k
         (truncate
          (photon-model-param-get
           model 'wordpiece-start-long-classifier-candidate-top-k)))
        (target-margin
         (photon-model-param-get
          model 'wordpiece-start-long-classifier-candidate-margin))
        (candidate-limit
         (truncate
          (photon-model-param-get
           model 'wordpiece-start-long-classifier-candidate-limit))))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((start-long-p (photon-start-long-token-p model token))
               (states (and start-long-p
                            (photon-forward-model model window)))
               (state (and states
                           (photon-prediction-state states window)))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (classifier-logits
                (and start-long-p
                     (photon-model-start-long-classifier-logits
                      model state)))
               (prediction
                (and classifier-logits
                     (photon-argmax-index classifier-logits)))
               (rival
                (and classifier-logits
                     (photon-best-non-target-index
                      classifier-logits token)))
               (rank-before
                (and classifier-logits
                     (photon-logit-rank classifier-logits token)))
               (margin-before
                (if rival
                    (photon-target-margin classifier-logits token rival)
                  0.0)))
          (when start-long-p
            (setq start-long-total (1+ start-long-total)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit prediction
                     (/= prediction token)
                     (< (length candidates) candidate-limit)
                     (< margin-before target-margin))
            (setq candidates
                  (cons (list (cons 'target token)
                              (cons 'prediction prediction)
                              (cons 'state state)
                              (cons 'rank-before rank-before)
                              (cons 'margin-before margin-before))
                        candidates))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (setq candidates (nreverse candidates))
    (list (cons 'candidates candidates)
          (cons 'candidate-count (length candidates))
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'target-margin target-margin)
          (cons 'candidate-limit candidate-limit)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-classifier-candidate-search
    (model token-reader-factory context-size)
  (let* ((candidate-report
          (photon-start-long-classifier-candidates-token-reader
           model (funcall token-reader-factory) context-size))
         (candidates (cdr (assq 'candidates candidate-report)))
         (initial-score
          (photon-evaluate-start-long-classifier-token-reader
           model (funcall token-reader-factory) context-size))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (best-top-3 (cdr (assq 'top-3-accuracy-permil initial-score)))
         (best-top-5 (cdr (assq 'top-5-accuracy-permil initial-score)))
         (best-rank (cdr (assq 'average-target-rank initial-score)))
         (best-head
          (copy-tree (photon-model-start-long-classifier-head model)))
         (best-bias
          (copy-tree (photon-model-start-long-classifier-bias model)))
         (learning-rate
          (photon-model-param-get
           model 'wordpiece-start-long-classifier-candidate-learning-rate))
         (tested 0)
         (accepted 0)
         (rejected 0)
         (updated 0))
    (while candidates
      (setq tested (1+ tested))
      (photon-model-set-start-long-classifier-head
       model (copy-tree best-head))
      (photon-model-set-start-long-classifier-bias
       model (copy-tree best-bias))
      (when (photon-update-start-long-classifier
             model
             (cdr (assq 'state (car candidates)))
             (cdr (assq 'prediction (car candidates)))
             (cdr (assq 'target (car candidates)))
             learning-rate)
        (let* ((score
                (photon-evaluate-start-long-classifier-token-reader
                 model (funcall token-reader-factory) context-size))
               (accuracy (cdr (assq 'accuracy-permil score)))
               (top-3 (cdr (assq 'top-3-accuracy-permil score)))
               (top-5 (cdr (assq 'top-5-accuracy-permil score)))
               (rank (cdr (assq 'average-target-rank score))))
          (if (or (> accuracy best-accuracy)
                  (and (= accuracy best-accuracy)
                       (> top-3 best-top-3))
                  (and (= accuracy best-accuracy)
                       (= top-3 best-top-3)
                       (> top-5 best-top-5))
                  (and (= accuracy best-accuracy)
                       (= top-3 best-top-3)
                       (= top-5 best-top-5)
                       (< rank best-rank)))
              (progn
                (setq accepted (1+ accepted))
                (setq updated (1+ updated))
                (setq best-accuracy accuracy)
                (setq best-top-3 top-3)
                (setq best-top-5 top-5)
                (setq best-rank rank)
                (setq best-head
                      (copy-tree
                       (photon-model-start-long-classifier-head model)))
                (setq best-bias
                      (copy-tree
                       (photon-model-start-long-classifier-bias model))))
            (setq rejected (1+ rejected)))))
      (setq candidates (cdr candidates)))
    (photon-model-set-start-long-classifier-head model best-head)
    (photon-model-set-start-long-classifier-bias model best-bias)
    (list (cons 'candidate-count
                (cdr (assq 'candidate-count candidate-report)))
          (cons 'tested tested)
          (cons 'accepted accepted)
          (cons 'rejected rejected)
          (cons 'updated updated)
          (cons 'learning-rate learning-rate)
          (cons 'initial-start-long-classifier initial-score)
          (cons 'best-start-long-classifier-accuracy-permil best-accuracy)
          (cons 'best-start-long-top-3-accuracy-permil best-top-3)
          (cons 'best-start-long-top-5-accuracy-permil best-top-5)
          (cons 'best-start-long-average-target-rank best-rank)
          (cons 'candidate-report candidate-report))))

(defun photon-start-long-boundary-readout-key (suffix)
  (cons 'start-long-boundary-readout (copy-sequence suffix)))

(defun photon-update-start-long-boundary-readout (model context target)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (when (and context
               vocab
               (photon-start-long-boundary-readout-enabled-p model)
               (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
               (eq (photon-wordpiece-token-shape vocab target)
                   'start-long-piece))
      (let* ((config (photon-model-config model))
             (vocab-size (photon-config-get config 'vocab-size))
             (max-size (photon-start-long-boundary-readout-max-size model))
             (size 1)
             (readouts
              (photon-model-start-long-boundary-readouts model)))
        (while (<= size max-size)
          (when (>= (length context) size)
            (let* ((suffix (photon-context-suffix context size))
                   (key (photon-start-long-boundary-readout-key suffix))
                   (cell (assoc key readouts))
                   (counts (if cell
                               (cdr cell)
                             (photon-repeat 0.0 vocab-size))))
              (setq counts
                    (photon-nth-set counts target
                                    (+ (nth target counts) 1.0)))
              (if cell
                  (setcdr cell counts)
                (setq readouts (cons (cons key counts) readouts)))))
          (setq size (1+ size)))
        (photon-model-set-start-long-boundary-readouts model readouts)))))

(defun photon-start-long-boundary-readout-logits (model context)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (vocab (photon-model-metadata-get model 'vocab))
         (max-size (photon-start-long-boundary-readout-max-size model))
         (ngram-mix
          (photon-model-param-get
           model 'wordpiece-start-long-boundary-ngram-mix))
         (size 1)
         (logits (photon-repeat 0.0 vocab-size))
         (readouts (photon-model-start-long-boundary-readouts model)))
    (when (and context
               vocab
               (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece))
      (when (> ngram-mix 0.0)
        (let ((ngram-logits (photon-context-anchor-ngram-logits model context))
              (token 0))
          (while (< token vocab-size)
            (when (eq (photon-wordpiece-token-shape vocab token)
                      'start-long-piece)
              (setq logits
                    (photon-nth-set
                     logits token
                     (+ (nth token logits)
                        (* ngram-mix (nth token ngram-logits))))))
            (setq token (1+ token)))))
      (while (<= size max-size)
        (when (>= (length context) size)
          (let* ((suffix (photon-context-suffix context size))
                 (counts
                  (cdr (assoc
                        (photon-start-long-boundary-readout-key suffix)
                        readouts)))
                 (total (if counts (photon-sum-numbers counts) 0.0))
                 (scale (/ (float size) (float max-size)))
                 (token 0))
            (when (> total 0.0)
              (while (< token vocab-size)
                (when (eq (photon-wordpiece-token-shape vocab token)
                          'start-long-piece)
                  (setq logits
                        (photon-nth-set
                         logits token
                         (+ (nth token logits)
                            (* scale (/ (nth token counts) total))))))
                (setq token (1+ token))))))
        (setq size (1+ size))))
    logits))

(defun photon-update-context-anchor-ngram-weighted
    (model context target amount)
  (when (and context (photon-context-anchor-ngram-enabled-p model))
    (let* ((config (photon-model-config model))
           (vocab-size (photon-config-get config 'vocab-size))
           (max-size (photon-context-anchor-ngram-max-size model))
           (size 1)
           (ngrams (photon-model-context-anchor-ngrams model)))
      (while (<= size max-size)
        (when (>= (length context) size)
          (let* ((suffix (photon-context-suffix context size))
                 (key (photon-context-anchor-ngram-key suffix))
                 (cell (assoc key ngrams))
                 (counts (if cell
                             (cdr cell)
                           (photon-repeat 0.0 vocab-size))))
            (setq counts
                  (photon-nth-set counts target
                                  (+ (nth target counts) amount)))
            (if cell
                (setcdr cell counts)
              (setq ngrams (cons (cons key counts) ngrams)))))
        (setq size (1+ size)))
      (photon-model-set-context-anchor-ngrams model ngrams))))

(defun photon-update-context-anchor-ngram (model context target)
  (photon-update-context-anchor-ngram-weighted model context target 1.0))

(defun photon-update-context-anchor-ngram-contrastive (model context target
                                                            rival margin
                                                            learning-rate)
  (when (and context
             rival
             (/= target rival)
             (photon-context-anchor-ngram-enabled-p model)
             (photon-context-anchor-ngram-contrastive-enabled-p model)
             (< margin (photon-model-param-get model 'next-token-margin)))
    (let* ((config (photon-model-config model))
           (vocab-size (photon-config-get config 'vocab-size))
           (max-size (photon-context-anchor-ngram-max-size model))
           (amount (* learning-rate
                      (photon-model-param-get
                       model 'context-anchor-ngram-contrastive-weight)))
           (size 1)
           (ngrams (photon-model-context-anchor-ngrams model)))
      (while (<= size max-size)
        (when (>= (length context) size)
          (let* ((suffix (photon-context-suffix context size))
                 (key (photon-context-anchor-ngram-key suffix))
                 (cell (assoc key ngrams))
                 (counts (if cell
                             (cdr cell)
                           (photon-repeat 0.0 vocab-size))))
            (setq counts
                  (photon-nth-set
                   counts target
                   (+ (nth target counts) amount)))
            (setq counts
                  (photon-nth-set
                   counts rival
                   (max 0.0 (- (nth rival counts) amount))))
            (if cell
                (setcdr cell counts)
              (setq ngrams (cons (cons key counts) ngrams)))))
        (setq size (1+ size)))
      (photon-model-set-context-anchor-ngrams model ngrams))))

(defun photon-char-class-bias-enabled-p (model)
  (> (photon-model-param-get model 'char-class-bias-enabled) 0.0))

(defun photon-char-boundary-bias-enabled-p (model)
  (> (photon-model-param-get model 'char-boundary-bias-enabled) 0.0))

(defun photon-structural-boundary-bias-enabled-p (model)
  (> (photon-model-param-get model 'structural-boundary-bias-enabled) 0.0))

(defun photon-char-boundary-bias-max-size (model)
  (max 1 (truncate (photon-model-param-get model 'char-boundary-bias-size))))

(defun photon-structural-boundary-bias-max-size (model)
  (max 1 (truncate (photon-model-param-get
                    model 'structural-boundary-bias-size))))

(defun photon-structural-char-class-p (class)
  (or (eq class 'newline)
      (eq class 'space)))

(defun photon-context-line-boundary-candidate-p (model context)
  (let ((candidate t)
        (walk context)
        (class nil))
    (while walk
      (setq class (photon-model-token-char-class model (car walk)))
      (unless (eq class 'letter)
        (setq candidate nil))
      (setq walk (cdr walk)))
    candidate))

(defun photon-model-token-char-class (model token)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (if vocab
        (photon-char-class (photon-token-to-char vocab token))
      'unknown)))

(defun photon-class-count-increment (counts class)
  (let ((cell (assq class counts)))
    (if cell
        (setcdr cell (1+ (cdr cell)))
      (setq counts (cons (cons class 1) counts))))
  counts)

(defun photon-class-count-total (counts)
  (let ((total 0))
    (while counts
      (setq total (+ total (cdr (car counts))))
      (setq counts (cdr counts)))
    total))

(defun photon-class-count-score (counts class)
  (let ((total (photon-class-count-total counts))
        (cell (assq class counts)))
    (if (and cell (> total 0))
        (/ (float (cdr cell)) total)
      0.0)))

(defun photon-context-class-suffix (model context size)
  (let ((tokens (photon-context-suffix context size))
        (classes nil))
    (while tokens
      (setq classes
            (cons (photon-model-token-char-class model (car tokens))
                  classes))
      (setq tokens (cdr tokens)))
    (nreverse classes)))

(defun photon-char-boundary-bias-key (classes)
  (cons 'char-boundary-bias (copy-sequence classes)))

(defun photon-structural-boundary-bias-key (suffix)
  (cons 'structural-boundary-bias (copy-sequence suffix)))

(defun photon-update-char-class-bias (model context target)
  (when (photon-model-metadata-get model 'vocab)
    (let* ((target-class (photon-model-token-char-class model target))
           (class-biases
            (photon-class-count-increment
             (photon-model-char-class-biases model)
             target-class))
           (max-size (photon-char-boundary-bias-max-size model))
           (structural-max-size
            (photon-structural-boundary-bias-max-size model))
           (size 1)
           (boundary-biases (photon-model-char-boundary-biases model))
           (structural-biases
            (photon-model-structural-boundary-biases model)))
      (photon-model-set-char-class-biases model class-biases)
      (while (<= size max-size)
        (when (>= (length context) size)
          (let* ((classes (photon-context-class-suffix model context size))
                 (key (photon-char-boundary-bias-key classes))
                 (cell (assoc key boundary-biases))
                 (counts (if cell (cdr cell) nil)))
            (setq counts (photon-class-count-increment counts target-class))
            (if cell
                (setcdr cell counts)
              (setq boundary-biases
                    (cons (cons key counts) boundary-biases)))))
        (setq size (1+ size)))
      (photon-model-set-char-boundary-biases model boundary-biases)
      (when (photon-structural-char-class-p target-class)
        (setq size 1)
        (while (<= size structural-max-size)
          (when (>= (length context) size)
            (let* ((suffix (photon-context-suffix context size))
                   (key (photon-structural-boundary-bias-key suffix))
                   (cell (assoc key structural-biases))
                   (counts (if cell (cdr cell) nil)))
              (setq counts
                    (photon-class-count-increment counts target-class))
              (if cell
                  (setcdr cell counts)
                (setq structural-biases
                      (cons (cons key counts) structural-biases)))))
          (setq size (1+ size)))
        (photon-model-set-structural-boundary-biases
         model structural-biases)))))

(defun photon-char-class-bias-logits (model context)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (class-biases (photon-model-char-class-biases model))
         (boundary-biases (photon-model-char-boundary-biases model))
         (structural-biases
          (photon-model-structural-boundary-biases model))
         (max-size (photon-char-boundary-bias-max-size model))
         (structural-max-size
          (photon-structural-boundary-bias-max-size model))
         (token 0)
         (logits nil))
    (while (< token vocab-size)
      (let* ((class (photon-model-token-char-class model token))
             (score 0.0)
             (size 1))
        (when (and (photon-structural-char-class-p class)
                   (photon-char-class-bias-enabled-p model))
          (setq score
                (+ score
                   (* (photon-model-param-get model 'char-class-bias-weight)
                      (photon-class-count-score class-biases class)))))
        (when (and (photon-structural-char-class-p class)
                   context
                   (photon-char-boundary-bias-enabled-p model))
          (while (<= size max-size)
            (when (>= (length context) size)
              (let* ((classes (photon-context-class-suffix model context size))
                     (counts (cdr (assoc (photon-char-boundary-bias-key
                                          classes)
                                         boundary-biases)))
                     (scale (/ (float size) (float max-size))))
                (setq score
                      (+ score
                         (* (photon-model-param-get
                             model 'char-boundary-bias-weight)
                            scale
                            (photon-class-count-score counts class))))))
            (setq size (1+ size))))
        (when (and (photon-structural-char-class-p class)
                   context
                   (photon-structural-boundary-bias-enabled-p model))
          (setq size 1)
          (while (<= size structural-max-size)
            (when (>= (length context) size)
              (let* ((suffix (photon-context-suffix context size))
                     (counts (cdr (assoc
                                   (photon-structural-boundary-bias-key
                                    suffix)
                                   structural-biases)))
                     (scale (/ (float size) (float structural-max-size))))
                (setq score
                      (+ score
                         (* (photon-model-param-get
                             model 'structural-boundary-bias-weight)
                            scale
                            (photon-class-count-score counts class))))))
            (setq size (1+ size))))
        (when (and (eq class 'newline)
                   context
                   (photon-context-line-boundary-candidate-p model context))
          (setq score
                (+ score
                   (photon-model-param-get model 'newline-boundary-bonus))))
        (setq logits (cons score logits)))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-context-anchor-state (model context)
  (let* ((config (photon-model-config model))
         (hidden-size (photon-config-get config 'hidden-size))
         (recency (photon-model-param-get model 'context-anchor-recency))
         (tokens (reverse context))
         (weight 1.0)
         (total-weight 0.0)
         (state (photon-vector-zeros hidden-size)))
    (while tokens
      (setq state
            (photon-vector-add
             state
             (photon-vector-scale
              weight
              (photon-token-embedding config (car tokens)))))
      (setq total-weight (+ total-weight weight))
      (setq weight (* weight recency))
      (setq tokens (cdr tokens)))
    (if (> total-weight 0.0)
        (photon-vector-scale (/ 1.0 total-weight) state)
      state)))

(defun photon-model-token-embedding-logits (model token)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (anchor (photon-token-embedding config token))
         (candidate 0)
         (logits nil))
    (while (< candidate vocab-size)
      (setq logits
            (cons (photon-vector-dot
                   anchor
                   (photon-token-embedding config candidate))
                  logits))
      (setq candidate (1+ candidate)))
    (nreverse logits)))

(defun photon-model-context-anchor-logits (model context)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (anchor (photon-context-anchor-state model context))
         (prefix-weight (photon-model-param-get
                         model 'context-anchor-prefix-weight))
         (token 0)
         (logits nil))
    (while (< token vocab-size)
      (setq logits
            (cons (photon-vector-dot
                   anchor
                   (photon-token-embedding config token))
                  logits))
      (setq token (1+ token)))
    (setq logits (nreverse logits))
    (if (and context
             (> prefix-weight 0.0)
             (<= (length context)
                 (photon-context-anchor-ngram-max-size model)))
        (photon-logits-add-scaled
         logits
         (photon-model-token-embedding-logits model (car context))
         prefix-weight)
      logits)))

(defun photon-model-context-anchor-enabled-p (model)
  (> (photon-model-param-get model 'context-anchor-enabled) 0.0))

(defun photon-structural-full-suffix-counts (model context)
  (when context
    (cdr (assoc (photon-structural-boundary-bias-key
                 (photon-context-suffix context (length context)))
                (photon-model-structural-boundary-biases model)))))

(defun photon-structural-count-score-direct (counts class)
  (let ((cell (assq class counts)))
    (if cell (cdr cell) 0.0)))

(defun photon-structural-tie-breaker-enabled-p (model)
  (> (photon-model-param-get model 'structural-tie-breaker-enabled) 0.0))

(defun photon-structural-tie-breaker-logits (model context)
  (let* ((config (photon-model-config model))
         (vocab-size (photon-config-get config 'vocab-size))
         (counts (photon-structural-full-suffix-counts model context))
         (token 0)
         (logits nil))
    (while (< token vocab-size)
      (let ((class (photon-model-token-char-class model token)))
        (setq logits
              (cons (if (photon-structural-char-class-p class)
                        (photon-class-count-score counts class)
                      0.0)
                    logits)))
      (setq token (1+ token)))
    (nreverse logits)))

(defun photon-structural-bias-gate-scale (model context base-logits)
  (let* ((prediction (photon-argmax-index base-logits))
         (class (photon-model-token-char-class model prediction))
         (all-letter (and context
                          (photon-context-line-boundary-candidate-p
                           model context)))
         (structural-counts
          (photon-structural-full-suffix-counts model context)))
    (cond ((and all-letter (null structural-counts))
           1.0)
          ((eq class 'letter)
           (photon-model-param-get model
                                   'structural-bias-letter-gate-scale))
          ((eq class 'space)
          (photon-model-param-get model
                                   'structural-bias-space-gate-scale))
          (t 1.0))))

(defun photon-start-long-only-logits (model logits)
  (let ((token 0)
        (result nil)
        (remaining logits))
    (while remaining
      (setq result
            (cons (if (photon-start-long-token-p model token)
                      (car remaining)
                    0.0)
                  result))
      (setq token (1+ token))
      (setq remaining (cdr remaining)))
    (nreverse result)))

(defun photon-model-readout-components (model state context)
  (let* ((linear-output-head (photon-model-output-head-logits model state))
         (output-head-prototype
          (photon-model-output-head-prototype-logits model state))
         (output-head-cluster
          (photon-model-output-head-state-cluster-logits model state))
         (raw-output-head (photon-model-raw-output-head-logits model state))
         (output-head (photon-model-dense-output-head-logits model state))
         (prototype (photon-model-target-prototype-logits model state))
         (state-cluster
          (photon-model-target-state-cluster-logits model state))
         (dense-prototype-readout state-cluster)
         (token-readout
          (photon-logits-add-scaled linear-output-head prototype 1.0))
         (char-structural-bias (photon-logits-zeros model))
         (structural-tie-breaker (photon-logits-zeros model))
         (embedding-anchor
          (if context
              (photon-model-token-embedding-logits model
                                                   (car (last context)))
            (photon-logits-zeros model)))
         (ngram-anchor (photon-logits-zeros model))
         (start-long-boundary-readout (photon-logits-zeros model))
         (start-long-boundary-teacher (photon-logits-zeros model))
         (start-long-classifier (photon-logits-zeros model))
         (start-long-classifier-readout (photon-logits-zeros model))
         (start-long-classifier-teacher (photon-logits-zeros model))
         (start-long-ngram-teacher (photon-logits-zeros model))
         (wordpiece-readout-bias (photon-logits-zeros model))
         (context-anchor (photon-logits-zeros model))
         (non-structural-readout token-readout)
         (base-readout token-readout)
         (full nil)
         (anchor-weight (photon-model-param-get model 'context-anchor-weight))
         (start-long-boundary-weight
          (photon-model-param-get
           model 'wordpiece-start-long-boundary-readout-weight))
         (start-long-boundary-teacher-weight
          (photon-model-param-get
           model
           'output-head-full-readout-distill-start-long-boundary-teacher-weight))
         (start-long-classifier-teacher-weight
          (photon-model-param-get
           model
           'output-head-full-readout-distill-start-long-classifier-teacher-weight))
         (start-long-classifier-readout-weight
          (photon-model-param-get
           model 'wordpiece-start-long-classifier-readout-weight))
         (start-long-ngram-teacher-weight
          (photon-model-param-get
           model
           'output-head-full-readout-distill-start-long-ngram-teacher-weight)))
    (when (photon-start-long-classifier-enabled-p model)
      (setq start-long-classifier
            (photon-model-start-long-classifier-logits model state)))
    (when (and context
               (photon-context-anchor-ngram-enabled-p model)
               (> (photon-model-param-get model
                                          'context-anchor-ngram-weight)
                  0.0))
      (setq ngram-anchor
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-context-anchor-ngram-logits model context)
             (photon-model-param-get model
                                     'context-anchor-ngram-weight)))
      (setq non-structural-readout
            (photon-logits-add-scaled
             non-structural-readout ngram-anchor 1.0))
      (setq wordpiece-readout-bias
            (photon-wordpiece-readout-bias-logits model ngram-anchor))
      (setq wordpiece-readout-bias
            (photon-logits-add-scaled
             wordpiece-readout-bias
             (photon-wordpiece-readout-bias-logits model token-readout)
             (photon-model-param-get
              model 'wordpiece-token-readout-bias-weight)))
      (setq non-structural-readout
            (photon-logits-add-scaled
             non-structural-readout wordpiece-readout-bias 1.0)))
    (when (and context
               (photon-start-long-boundary-readout-enabled-p model)
               (> start-long-boundary-weight 0.0))
      (setq start-long-boundary-readout
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-start-long-boundary-readout-logits model context)
             start-long-boundary-weight))
      (setq non-structural-readout
            (photon-logits-add-scaled
             non-structural-readout start-long-boundary-readout 1.0)))
    (when (and context
               (photon-model-context-anchor-enabled-p model)
               (> anchor-weight 0.0))
      (setq context-anchor
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-model-context-anchor-logits model context)
             anchor-weight))
      (setq non-structural-readout
            (photon-logits-add-scaled
             non-structural-readout context-anchor 1.0)))
    (setq base-readout non-structural-readout)
    (when (and context
               (photon-structural-tie-breaker-enabled-p model)
               (> (photon-model-param-get model
                                          'structural-tie-breaker-weight)
                  0.0))
      (setq structural-tie-breaker
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-structural-tie-breaker-logits model context)
             (photon-model-param-get
              model 'structural-tie-breaker-weight)))
      (setq base-readout
            (photon-logits-add-scaled
             base-readout structural-tie-breaker 1.0)))
    (setq full base-readout)
    (when (and (photon-start-long-classifier-enabled-p model)
               (> start-long-classifier-readout-weight 0.0))
      (setq start-long-classifier-readout
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-start-long-only-logits model start-long-classifier)
             start-long-classifier-readout-weight))
      (setq full
            (photon-logits-add-scaled
             full start-long-classifier-readout 1.0)))
    (when (or (photon-char-class-bias-enabled-p model)
              (photon-char-boundary-bias-enabled-p model)
              (photon-structural-boundary-bias-enabled-p model))
      (setq char-structural-bias
            (photon-logits-add-scaled
             (photon-logits-zeros model)
             (photon-char-class-bias-logits model context)
             (photon-structural-bias-gate-scale
             model context base-readout)))
      (setq full
            (photon-logits-add-scaled full char-structural-bias 1.0)))
    (when (and context
               (photon-start-long-boundary-readout-enabled-p model)
               (> start-long-boundary-teacher-weight 0.0))
      (setq start-long-boundary-teacher
            (photon-logits-add-scaled
             full
             (photon-start-long-boundary-readout-logits model context)
             start-long-boundary-teacher-weight)))
    (when (and (photon-start-long-classifier-enabled-p model)
               (> start-long-classifier-teacher-weight 0.0))
      (setq start-long-classifier-teacher
            (photon-logits-add-scaled
             full
             start-long-classifier
             start-long-classifier-teacher-weight)))
    (when (and context (> start-long-ngram-teacher-weight 0.0))
      (setq start-long-ngram-teacher
            (photon-logits-add-scaled
             full
             (photon-start-long-only-logits model ngram-anchor)
             start-long-ngram-teacher-weight)))
    (when (> (photon-model-param-get model 'readout-logit-temperature) 1.0)
      (setq full
            (photon-logits-scale-temperature
             full
             (photon-model-param-get model 'readout-logit-temperature))))
    (list (cons 'linear-output-head linear-output-head)
          (cons 'output-head-prototype output-head-prototype)
          (cons 'raw-output-head raw-output-head)
          (cons 'output-head-cluster output-head-cluster)
          (cons 'output-head output-head)
          (cons 'prototype prototype)
          (cons 'state-cluster state-cluster)
          (cons 'dense-prototype-readout dense-prototype-readout)
          (cons 'token-readout token-readout)
          (cons 'embedding-anchor embedding-anchor)
          (cons 'base-readout base-readout)
          (cons 'char-structural-bias char-structural-bias)
          (cons 'structural-tie-breaker structural-tie-breaker)
          (cons 'ngram-anchor ngram-anchor)
          (cons 'start-long-boundary-readout start-long-boundary-readout)
          (cons 'start-long-boundary-teacher start-long-boundary-teacher)
          (cons 'start-long-classifier start-long-classifier)
          (cons 'start-long-classifier-readout start-long-classifier-readout)
          (cons 'start-long-classifier-teacher start-long-classifier-teacher)
          (cons 'start-long-ngram-teacher start-long-ngram-teacher)
          (cons 'wordpiece-readout-bias wordpiece-readout-bias)
          (cons 'context-anchor context-anchor)
          (cons 'structural-off non-structural-readout)
          (cons 'char-structural-bias-off base-readout)
          (cons 'full full))))

(defun photon-model-readout-logits (model state context)
  (cdr (assq 'full
             (photon-model-readout-components model state context))))

(defun photon-prediction-state (states tokens)
  (let ((index (1- (length tokens))))
    (when (< index 0)
      (setq index 0))
    (while (>= index (length states))
      (setq index (1- index)))
    (nth index states)))

(defun photon-model-reconstruction-logits (model state)
  (let ((head (photon-model-reconstruction-head model))
        (logits nil))
    (while head
      (setq logits (cons (photon-vector-dot state (car head)) logits))
      (setq head (cdr head)))
    (nreverse logits)))

(defun photon-argmax-index (values)
  (let ((best-index 0)
        (best-value (car values))
        (index 0))
    (while values
      (when (> (car values) best-value)
        (setq best-value (car values))
        (setq best-index index))
      (setq values (cdr values))
      (setq index (1+ index)))
    best-index))

(defun photon-best-non-target-index (values target)
  (let ((best-index target)
        (best-value nil)
        (index 0))
    (while values
      (unless (= index target)
        (when (or (null best-value) (> (car values) best-value))
          (setq best-value (car values))
          (setq best-index index)))
      (setq values (cdr values))
      (setq index (1+ index)))
    best-index))

(defun photon-target-margin (logits target rival)
  (- (nth target logits) (nth rival logits)))

(defun photon-top-k-logits (logits k)
  (let ((indexed nil)
        (index 0))
    (while logits
      (setq indexed (cons (cons index (car logits)) indexed))
      (setq logits (cdr logits))
      (setq index (1+ index)))
    (photon-take
     (sort indexed (lambda (left right) (> (cdr left) (cdr right))))
     k)))

(defun photon-top-k-hit-p (top-k target)
  (let ((hit nil))
    (while top-k
      (when (= (car (car top-k)) target)
        (setq hit t))
      (setq top-k (cdr top-k)))
    hit))

(defun photon-logit-rank (logits target)
  (let ((target-logit (nth target logits))
        (rank 1)
        (index 0))
    (while logits
      (when (or (> (car logits) target-logit)
                (and (< index target)
                     (= (car logits) target-logit)))
        (setq rank (1+ rank)))
      (setq logits (cdr logits))
      (setq index (1+ index)))
    rank))

(defun photon-logits-scale-temperature (logits temperature)
  (let ((temp (max 0.0001 (or temperature 1.0)))
        (result nil))
    (while logits
      (setq result (cons (/ (car logits) temp) result))
      (setq logits (cdr logits)))
    (nreverse result)))

(defun photon-logits-keep-indices (logits indices)
  (let ((result nil)
        (index 0))
    (while logits
      (setq result
            (cons (if (photon-member-equal-p index indices)
                      (car logits)
                    -1000000000.0)
                  result))
      (setq logits (cdr logits))
      (setq index (1+ index)))
    (nreverse result)))

(defun photon-logits-apply-top-k (logits k)
  (if (and k (> k 0))
      (photon-logits-keep-indices
       logits
       (mapcar 'car (photon-top-k-logits logits k)))
    logits))

(defun photon-softmax-probs (logits)
  (let ((max-logit (apply 'max logits))
        (sum 0.0)
        (exps nil)
        (walk logits))
    (while walk
      (let ((value (exp (- (car walk) max-logit))))
        (setq exps (cons value exps))
        (setq sum (+ sum value)))
      (setq walk (cdr walk)))
    (setq exps (nreverse exps))
    (let ((result nil))
      (while exps
        (setq result (cons (/ (car exps) sum) result))
        (setq exps (cdr exps)))
      (nreverse result))))

(defun photon-logits-apply-top-p (logits top-p)
  (if (and top-p (> top-p 0.0) (< top-p 1.0))
      (let* ((probs (photon-softmax-probs logits))
             (indexed nil)
             (index 0)
             (keep nil)
             (sum 0.0))
        (while probs
          (setq indexed (cons (cons index (car probs)) indexed))
          (setq index (1+ index))
          (setq probs (cdr probs)))
        (setq indexed
              (sort indexed (lambda (left right) (> (cdr left) (cdr right)))))
        (while (and indexed (< sum top-p))
          (setq keep (cons (car (car indexed)) keep))
          (setq sum (+ sum (cdr (car indexed))))
          (setq indexed (cdr indexed)))
        (photon-logits-keep-indices logits keep))
    logits))

(defun photon-logits-apply-repetition-penalty (logits context penalty)
  (if (and context penalty (> penalty 1.0))
      (let ((result nil)
            (index 0))
        (while logits
          (setq result
                (cons (if (photon-member-equal-p index context)
                          (/ (car logits) penalty)
                        (car logits))
                      result))
          (setq logits (cdr logits))
          (setq index (1+ index)))
        (nreverse result))
    logits))

(defun photon-sampling-threshold (options context)
  (let ((seed (cdr (assq 'seed options))))
    (if seed
        (let* ((seed-value (if (numberp seed) seed (sxhash seed)))
               (context-value (abs (sxhash context)))
               (mixed (+ (* 1103515245 seed-value)
                         (* 12345 (length context))
                         context-value)))
          (/ (float (mod (abs mixed) 1000000)) 1000000.0))
      (/ (float (random 1000000)) 1000000.0))))

(defun photon-sample-index-from-probs (probs &optional options context)
  (let ((threshold (photon-sampling-threshold options context))
        (sum 0.0)
        (index 0)
        (chosen 0)
        (done nil))
    (while (and probs (not done))
      (setq sum (+ sum (car probs)))
      (when (>= sum threshold)
        (setq chosen index)
        (setq done t))
      (setq probs (cdr probs))
      (setq index (1+ index)))
    chosen))

(defun photon-sample-index (logits &optional options context)
  (let* ((mode (or (cdr (assq 'mode options)) 'greedy))
         (temperature (or (cdr (assq 'temperature options)) 1.0))
         (top-k (cdr (assq 'top-k options)))
         (top-p (cdr (assq 'top-p options)))
         (penalty (cdr (assq 'repetition-penalty options)))
         (adjusted (photon-logits-scale-temperature logits temperature)))
    (setq adjusted
          (photon-logits-apply-repetition-penalty adjusted context penalty))
    (setq adjusted (photon-logits-apply-top-k adjusted top-k))
    (setq adjusted (photon-logits-apply-top-p adjusted top-p))
    (if (eq mode 'sample)
        (photon-sample-index-from-probs
         (photon-softmax-probs adjusted) options context)
      (photon-argmax-index adjusted))))

(defun photon-forward (config tokens)
  (let* ((chunks (photon-chunk-tokens tokens (photon-config-get config 'chunk-size)))
         (chunk-states nil)
         (contexts nil)
         (decoded nil))
    (let ((walk chunks))
      (while walk
        (setq chunk-states
              (cons (photon-chunker config (car walk)) chunk-states))
        (setq walk (cdr walk))))
    (setq contexts (photon-context-encoder config (nreverse chunk-states)))
    (while (and chunks contexts)
      (setq decoded
            (append decoded
                    (photon-local-decode-chunk config (car chunks) (car contexts))))
      (setq chunks (cdr chunks))
      (setq contexts (cdr contexts)))
    decoded))

(defun photon-forward-model-decode (model chunks contexts)
  (let ((decoded nil))
    (while (and chunks contexts)
      (setq decoded
            (append decoded
                    (photon-model-local-decode-chunk
                     model (car chunks) (car contexts))))
      (setq chunks (cdr chunks))
      (setq contexts (cdr contexts)))
    decoded))

(defun photon-forward-model-build-info (model tokens)
  (let* ((config (photon-model-config model))
         (chunks (photon-chunk-tokens tokens (photon-config-get config 'chunk-size)))
         (chunk-states nil)
         (contexts nil)
         (decoded nil)
         (walk chunks))
    (while walk
      (setq chunk-states
            (cons (photon-model-chunker model (car walk)) chunk-states))
      (setq walk (cdr walk)))
    (setq chunk-states (nreverse chunk-states))
    (setq contexts (photon-model-context-encoder model chunk-states))
    (setq decoded (photon-forward-model-decode model chunks contexts))
    (list (cons 'tokens (copy-sequence tokens))
          (cons 'chunks chunks)
          (cons 'chunk-states chunk-states)
          (cons 'contexts contexts)
          (cons 'decoded decoded)
          (cons 'incremental nil))))

(defun photon-forward-model-incremental-info (model tokens previous-info)
  (let* ((previous-tokens (cdr (assq 'tokens previous-info)))
         (config (photon-model-config model))
         (chunk-size (photon-config-get config 'chunk-size))
         (stable-chunks (/ (length previous-tokens) chunk-size))
         (chunks (photon-chunk-tokens tokens chunk-size))
         (previous-chunks (cdr (assq 'chunks previous-info)))
         (previous-chunk-states (cdr (assq 'chunk-states previous-info)))
         (previous-contexts (cdr (assq 'contexts previous-info)))
         (previous-decoded (cdr (assq 'decoded previous-info))))
    (if (and previous-tokens
             (> (length tokens) (length previous-tokens))
             (photon-prefix-p previous-tokens tokens)
             (> stable-chunks 0)
             (photon-list-equal-prefix-p previous-chunks chunks stable-chunks))
        (let* ((prefix-chunk-states (photon-take previous-chunk-states stable-chunks))
               (prefix-contexts (photon-take previous-contexts stable-chunks))
               (prefix-decoded (photon-take previous-decoded (* stable-chunks chunk-size)))
               (suffix-chunks (nthcdr stable-chunks chunks))
               (suffix-chunk-states nil)
               (suffix-contexts nil)
               (suffix-decoded nil)
               (previous-context (car (last prefix-contexts)))
               (walk suffix-chunks))
          (while walk
            (setq suffix-chunk-states
                  (cons (photon-model-chunker model (car walk))
                        suffix-chunk-states))
            (setq walk (cdr walk)))
          (setq suffix-chunk-states (nreverse suffix-chunk-states))
          (setq suffix-contexts
                (photon-model-context-encoder-from
                 model suffix-chunk-states previous-context))
          (setq suffix-decoded
                (photon-forward-model-decode model suffix-chunks suffix-contexts))
          (list (cons 'tokens (copy-sequence tokens))
                (cons 'chunks chunks)
                (cons 'chunk-states
                      (append prefix-chunk-states suffix-chunk-states))
                (cons 'contexts (append prefix-contexts suffix-contexts))
                (cons 'decoded (append prefix-decoded suffix-decoded))
                (cons 'incremental t)
                (cons 'reused-chunks stable-chunks)))
      nil)))

(defun photon-forward-model (model tokens)
  (let* ((cache-key (photon-window-cache-key tokens))
         (cached (photon-model-cache-get model cache-key)))
    (if cached
        cached
      (photon-model-cache-put
       model cache-key
       (let* ((last-key (photon-incremental-forward-cache-key))
              (previous-info (photon-model-cache-get model last-key))
              (info (or (and previous-info
                             (photon-forward-model-incremental-info
                              model tokens previous-info))
                        (photon-forward-model-build-info model tokens)))
              (decoded (cdr (assq 'decoded info))))
         (photon-model-cache-put model last-key info)
         decoded)))))

(defun photon-next-token (config tokens)
  (let* ((states (photon-forward config tokens))
         (last-state (photon-prediction-state states tokens))
         (logits (photon-token-logits config last-state)))
    (photon-argmax-index logits)))

(defun photon-model-next-token (model tokens)
  (let ((remembered (photon-model-next-token-memory-get model tokens)))
    (if remembered
        remembered
      (let* ((states (photon-forward-model model tokens))
             (last-state (photon-prediction-state states tokens))
             (logits (photon-model-readout-logits model last-state tokens)))
        (photon-argmax-index logits)))))

(defun photon-token-run-length (tokens token)
  (let ((walk (reverse tokens))
        (run 0)
        (done nil))
    (while (and walk (not done))
      (if (= (car walk) token)
          (setq run (1+ run))
        (setq done t))
      (setq walk (cdr walk)))
    run))

(defun photon-wordpiece-continuation-run-length (vocab tokens)
  (let ((walk (reverse tokens))
        (run 0)
        (done nil))
    (while (and walk (not done))
      (if (photon-wordpiece-token-continuation-p vocab (car walk))
          (setq run (1+ run))
        (setq done t))
      (setq walk (cdr walk)))
    run))

(defun photon-logits-add-token-list-bias (logits tokens bias)
  (let ((result nil)
        (index 0))
    (while logits
      (setq result
            (cons (+ (car logits)
                     (if (photon-member-equal-p index tokens) bias 0.0))
                  result))
      (setq logits (cdr logits))
      (setq index (1+ index)))
    (nreverse result)))

(defun photon-context-repeated-bigram-candidates (context window)
  (let* ((recent (photon-context-suffix context (max 2 window)))
         (last-token (car (last recent)))
         (previous recent)
         (next (cdr recent))
         (candidates nil))
    (while next
      (when (= (car previous) last-token)
        (setq candidates (cons (car next) candidates)))
      (setq previous (cdr previous))
      (setq next (cdr next)))
    candidates))

(defun photon-logits-add-token-bias (logits token bias)
  (let ((result nil)
        (index 0))
    (while logits
      (setq result
            (cons (+ (car logits) (if (= index token) bias 0.0))
                  result))
      (setq logits (cdr logits))
      (setq index (1+ index)))
    (nreverse result)))

(defun photon-generation-gated-ngram-bias-weight
    (base-logits ngram-logits weight gate-margin)
  (if (or (null gate-margin) (< gate-margin 0.0))
      weight
    (let* ((candidate (photon-argmax-index ngram-logits))
           (candidate-ngram-score (nth candidate ngram-logits))
           (best (photon-argmax-index base-logits))
           (candidate-logit (nth candidate base-logits))
           (best-logit (nth best base-logits))
           (gap (- candidate-logit best-logit))
           (factor 0.0))
      (when (> candidate-ngram-score 0.0)
        (setq factor
              (if (<= gate-margin 0.0)
                  (if (>= gap 0.0) 1.0 0.0)
                (min 1.0 (max 0.0 (/ (+ gap gate-margin)
                                      gate-margin))))))
      (* weight factor))))

(defun photon-generation-repeat-scaled-ngram-bias-weight
    (ngram-logits weight history repeat-scale repeat-window)
  (let ((candidate (photon-argmax-index ngram-logits)))
    (if (or (>= repeat-scale 1.0)
            (not (photon-member-equal-p
                  candidate
                  (photon-context-repeated-bigram-candidates
                   history repeat-window))))
        weight
      (* weight (max 0.0 repeat-scale)))))

(defun photon-generation-repeat-ngram-bias-candidates
    (history window min-count)
  (let ((candidates (photon-context-repeated-bigram-candidates
                     history window)))
    (if (>= (length candidates) min-count)
        candidates
      nil)))

(defun photon-generation-options-with-history (options history)
  (cons (cons 'generation-history history)
        (assq-delete-all 'generation-history (copy-sequence options))))

(defun photon-generation-adjust-logits (model context logits &optional options)
  (let ((adjusted logits))
    (when context
      (let* ((last-token (car (last context)))
             (run (photon-token-run-length context last-token))
             (repeat-weight
              (photon-model-param-get model 'generation-repeat-bias-weight)))
        (when (and (>= run 2) (> repeat-weight 0.0))
          (setq adjusted
                (photon-logits-add-token-bias
                 adjusted last-token (* -1.0 repeat-weight run)))))
      (let* ((vocab (photon-model-metadata-get model 'vocab))
             (last-token (car (last context)))
             (last-char (and vocab (photon-token-to-char vocab last-token))))
        (when (and last-char (string= last-char "\n")
                   (>= (photon-token-run-length context last-token) 1))
          (setq adjusted
                (photon-logits-add-token-bias
                 adjusted last-token
                 (* -1.0
                    (photon-model-param-get
                     model 'generation-newline-run-bias-weight)))))
        (when (and last-char (string= last-char " ")
                   (>= (photon-token-run-length context last-token) 2))
          (setq adjusted
                (photon-logits-add-token-bias
                 adjusted last-token
                 (* -1.0
                    (photon-model-param-get
                     model 'generation-space-run-bias-weight)))))
        (when (and vocab
                   (eq (photon-model-metadata-get model 'tokenizer)
                       'wordpiece))
          (let* ((repeat-run (photon-token-run-length context last-token))
                 (repeat-weight
                  (photon-model-param-get
                   model 'wordpiece-generation-repeat-bias-weight))
                 (continuation-run
                  (photon-wordpiece-continuation-run-length vocab context))
                 (continuation-weight
                  (photon-model-param-get
                   model 'wordpiece-generation-continuation-run-bias-weight))
                 (recent-weight
                  (photon-model-param-get
                   model 'wordpiece-generation-recent-token-bias-weight))
                 (recent-window
                  (max 1
                       (truncate
                        (photon-model-param-get
                         model 'wordpiece-generation-recent-token-window))))
                 (ngram-bias-weight
                  (or (cdr (assq 'ngram-bias options))
                      (photon-model-param-get
                       model 'wordpiece-generation-ngram-bias-weight)))
                 (ngram-bias-gate-margin
                  (or (cdr (assq 'ngram-bias-gate-margin options))
                      (photon-model-param-get
                       model 'wordpiece-generation-ngram-bias-gate-margin)))
                 (ngram-bias-repeat-scale
                  (or (cdr (assq 'ngram-bias-repeat-scale options))
                      (photon-model-param-get
                       model 'wordpiece-generation-ngram-bias-repeat-scale)))
                 (ngram-bias-repeat-window
                  (max 2
                       (truncate
                        (or (cdr (assq 'ngram-bias-repeat-window options))
                            (photon-model-param-get
                             model
                             'wordpiece-generation-ngram-bias-repeat-window)))))
                 (history (or (cdr (assq 'generation-history options))
                              context))
                 (repeat-ngram-weight
                  (or (cdr (assq 'repeat-ngram-bias options))
                      (photon-model-param-get
                       model 'wordpiece-generation-repeat-ngram-bias-weight)))
                 (repeat-ngram-min-count
                  (max 1
                       (truncate
                        (or (cdr (assq 'repeat-ngram-bias-min-count options))
                            (photon-model-param-get
                             model
                             'wordpiece-generation-repeat-ngram-bias-min-count)))))
                 (repeat-ngram-window
                  (max 2
                       (truncate
                        (photon-model-param-get
                         model 'wordpiece-generation-repeat-ngram-window))))
                 (special-tokens nil)
                 (continuation-tokens nil)
                 (index 0)
                 (walk vocab))
            (while walk
              (when (photon-member-equal-p
                     (car walk) (photon-wordpiece-special-tokens))
                (setq special-tokens (cons index special-tokens)))
              (setq index (1+ index))
              (setq walk (cdr walk)))
            (setq adjusted
                  (photon-logits-add-token-list-bias
                   adjusted special-tokens -1000000000.0))
            (setq index 0)
            (setq walk vocab)
            (when (and (>= repeat-run 1) (> repeat-weight 0.0))
              (setq adjusted
                    (photon-logits-add-token-bias
                     adjusted last-token (* -1.0 repeat-weight repeat-run))))
            (when (> recent-weight 0.0)
              (setq adjusted
                    (photon-logits-add-token-list-bias
                     adjusted
                     (photon-context-suffix context recent-window)
                     (* -1.0 recent-weight))))
            (when (> ngram-bias-weight 0.0)
              (setq adjusted
                    (let* ((ngram-logits
                            (photon-context-anchor-ngram-logits
                             model context))
                           (effective-weight
                            (photon-generation-gated-ngram-bias-weight
                             adjusted ngram-logits ngram-bias-weight
                             ngram-bias-gate-margin)))
                      (setq effective-weight
                            (photon-generation-repeat-scaled-ngram-bias-weight
                             ngram-logits effective-weight history
                             ngram-bias-repeat-scale
                             ngram-bias-repeat-window))
                      (photon-logits-add-scaled
                       adjusted ngram-logits effective-weight))))
            (when (> repeat-ngram-weight 0.0)
              (setq adjusted
                    (photon-logits-add-token-list-bias
                     adjusted
                     (photon-generation-repeat-ngram-bias-candidates
                      history repeat-ngram-window repeat-ngram-min-count)
                     (* -1.0 repeat-ngram-weight))))
            (when (and (>= continuation-run 2)
                       (> continuation-weight 0.0))
              (while walk
                (when (string-prefix-p "##" (car walk))
                  (setq continuation-tokens
                        (cons index continuation-tokens)))
                (setq index (1+ index))
                (setq walk (cdr walk)))
              (setq adjusted
                    (photon-logits-add-token-list-bias
                     adjusted continuation-tokens
                     (* -1.0 continuation-weight continuation-run))))))))
    adjusted))

(defun photon-generate (config prompt steps)
  (let ((tokens prompt)
        (i 0))
    (while (< i steps)
      (setq tokens (append tokens (list (photon-next-token config tokens))))
      (setq i (1+ i)))
    tokens))

(defun photon-model-generate (model prompt steps &optional options)
  (let ((tokens prompt)
        (i 0))
    (while (< i steps)
      (let* ((states (photon-forward-model model tokens))
             (state (photon-prediction-state states tokens))
             (logits (photon-model-readout-logits model state tokens))
             (adjusted-logits (photon-generation-adjust-logits
                               model tokens logits options))
             (next-token (photon-sample-index
                          adjusted-logits options tokens)))
        (setq tokens (append tokens (list next-token))))
      (setq i (1+ i)))
    tokens))

(defun photon-model-next-token-trace-record (model vocab context step
                                                   &optional options history)
  (let* ((states (photon-forward-model model context))
         (state (photon-prediction-state states context))
         (components (photon-model-readout-components model state context))
         (logits (cdr (assq 'full components)))
         (adjusted-logits (photon-generation-adjust-logits
                           model context logits
                           (if history
                               (photon-generation-options-with-history
                                options history)
                             options)))
         (prediction (photon-sample-index adjusted-logits options context)))
    (list (cons 'step step)
          (cons 'options options)
          (cons 'mode (or (cdr (assq 'mode options)) 'greedy))
          (cons 'context context)
          (cons 'context-text
                (if vocab (photon-decode-tokens vocab context) ""))
          (cons 'prediction prediction)
          (cons 'prediction-char
                (if vocab (photon-token-to-char vocab prediction) ""))
          (cons 'top-5
                (if vocab
                    (photon-top-k-logits-with-chars
                     (photon-top-k-logits adjusted-logits 5)
                     vocab)
                  (photon-top-k-logits logits 5)))
          (cons 'raw-top-5
                (if vocab
                    (photon-top-k-logits-with-chars
                     (photon-top-k-logits logits 5)
                     vocab)
                  (photon-top-k-logits logits 5)))
          (cons 'readout-contributions
                (if vocab
                    (photon-readout-contribution-debug
                     vocab components prediction prediction)
                  nil)))))

(defun photon-model-generate-with-trace (model prompt steps &optional options)
  (let ((tokens prompt)
        (trace nil)
        (i 0))
    (while (< i steps)
      (let* ((record (photon-model-next-token-trace-record
                      model nil tokens i options))
             (next-token (cdr (assq 'prediction record))))
        (setq trace (cons record trace))
        (setq tokens (append tokens (list next-token))))
      (setq i (1+ i)))
    (list (cons 'tokens tokens)
          (cons 'trace (nreverse trace)))))

(defun photon-window-append (window value max-size)
  (let ((next (append window (list value))))
    (while (> (length next) max-size)
      (setq next (cdr next)))
    next))

(defun photon-window-slide (window value)
  (append (cdr window) (list value)))

(defun photon-take (items count)
  (let ((result nil)
        (i 0))
    (while (and items (< i count))
      (setq result (cons (car items) result))
      (setq items (cdr items))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-prefix-p (prefix items)
  (let ((ok t))
    (while (and ok prefix)
      (if (or (null items) (/= (car prefix) (car items)))
          (setq ok nil)
        (setq prefix (cdr prefix))
        (setq items (cdr items))))
    ok))

(defun photon-list-equal-prefix-p (left right count)
  (let ((ok t)
        (i 0))
    (while (and ok (< i count))
      (if (or (null left) (null right) (not (equal (car left) (car right))))
          (setq ok nil)
        (setq left (cdr left))
        (setq right (cdr right))
        (setq i (1+ i))))
    ok))

(defun photon-training-pairs (tokens context-size)
  (let ((pairs nil)
        (window nil)
        (remaining tokens)
        (filled 0))
    (while (and remaining (< filled context-size))
      (setq window (cons (car remaining) window))
      (setq remaining (cdr remaining))
      (setq filled (1+ filled)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while remaining
        (setq pairs (cons (cons window (car remaining)) pairs))
        (setq window (photon-window-slide window (car remaining)))
        (setq remaining (cdr remaining))))
    (nreverse pairs)))

(defun photon-make-list-token-reader (tokens)
  (let ((remaining tokens))
    (lambda ()
      (let ((token (car remaining)))
        (setq remaining (cdr remaining))
        token))))

(defun photon-wordpiece-update-scale (model target)
  (let ((tokenizer (photon-model-metadata-get model 'tokenizer))
        (vocab (photon-model-metadata-get model 'vocab)))
    (if (and (eq tokenizer 'wordpiece) vocab)
        (let* ((length (photon-wordpiece-token-length vocab target))
               (counts (photon-model-target-counts model))
               (count (nth target counts))
               (long-extra (max 0 (- length 2)))
               (long-scale (* (photon-model-param-get
                                model 'wordpiece-long-token-update-weight)
                              long-extra))
               (rare-scale (/ (photon-model-param-get
                                model 'wordpiece-rare-token-update-weight)
                               (sqrt (float (1+ count)))))
               (scale (+ 1.0 long-scale rare-scale))
               (max-scale (photon-model-param-get
                           model 'wordpiece-update-scale-max)))
          (min max-scale scale))
      1.0)))

(defun photon-full-readout-distill-update-scale (model target)
  (let ((scale
         (if (> (photon-model-param-get
                 model 'output-head-full-readout-distill-wordpiece-scale-enabled)
                0.0)
             (photon-wordpiece-update-scale model target)
           1.0))
        (vocab (photon-model-metadata-get model 'vocab)))
    (when (and vocab
               (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece))
      (let ((shape (photon-wordpiece-token-shape vocab target)))
        (setq scale
              (* scale
                 (cond
                  ((eq shape 'start-long-piece)
                   (photon-model-param-get
                    model
                    'output-head-full-readout-distill-start-long-scale))
                  ((eq shape 'continuation-long-piece)
                   (photon-model-param-get
                    model
                    'output-head-full-readout-distill-continuation-long-scale))
                  ((eq shape 'start-short-piece)
                   (photon-model-param-get
                    model
                    'output-head-full-readout-distill-start-short-scale))
                  ((eq shape 'continuation-short-piece)
                   (photon-model-param-get
                    model
                    'output-head-full-readout-distill-continuation-short-scale))
                  (t 1.0))))))
    scale))

(defun photon-wordpiece-readout-bias-logits (model source-logits)
  (let ((tokenizer (photon-model-metadata-get model 'tokenizer))
        (vocab (photon-model-metadata-get model 'vocab)))
    (if (and (eq tokenizer 'wordpiece)
             (> (photon-model-param-get model
                                        'wordpiece-readout-bias-enabled)
                0.0)
             vocab source-logits)
        (let ((counts (photon-model-target-counts model))
              (long-weight
               (photon-model-param-get model
                                       'wordpiece-long-readout-bias-weight))
              (rare-weight
               (photon-model-param-get model
                                       'wordpiece-rare-readout-bias-weight))
              (max-bias
               (photon-model-param-get model 'wordpiece-readout-bias-max))
              (result nil)
              (token 0)
              (walk source-logits))
          (while walk
            (let* ((source (max 0.0 (car walk)))
                   (length (photon-wordpiece-token-length vocab token))
                   (count (or (nth token counts) 0))
                   (long-extra (max 0 (- length 2)))
                   (rare-extra (/ 1.0 (sqrt (float (1+ count)))))
                   (bias (* source
                            (min max-bias
                                 (+ (* long-weight long-extra)
                                    (* rare-weight rare-extra))))))
              (setq result (cons bias result)))
            (setq token (1+ token))
            (setq walk (cdr walk)))
          (nreverse result))
      (photon-logits-zeros model))))

(defun photon-update-output-head (model state prediction target learning-rate)
  (let* ((learning-rate (* learning-rate
                           (photon-wordpiece-update-scale model target)))
         (head (photon-model-output-head model))
         (bias (photon-model-output-head-bias model))
         (target-row (nth target head))
         (prediction-row (nth prediction head))
         (delta (photon-vector-scale learning-rate state))
         (new-target-row (photon-vector-add target-row delta))
         (new-prediction-row (photon-vector-add prediction-row
                                                (photon-vector-scale -1.0 delta))))
    (setq head (photon-nth-set head target new-target-row))
    (setq head (photon-nth-set head prediction new-prediction-row))
    (setq bias (photon-nth-set bias target
                               (+ (nth target bias) learning-rate)))
    (setq bias (photon-nth-set bias prediction
                               (- (nth prediction bias) learning-rate)))
    (photon-model-set-output-head model head)
    (photon-model-set-output-head-bias model bias)))

(defun photon-update-target-prototype (model state target learning-rate)
  (let* ((prototypes (photon-model-target-prototypes model))
         (counts (photon-model-target-counts model))
         (row (nth target prototypes))
         (count (nth target counts))
         (rate (/ (* learning-rate
                     (photon-wordpiece-update-scale model target))
                  (sqrt (float (1+ count)))))
         (delta (photon-vector-scale rate
                                     (photon-vector-add
                                      state
                                      (photon-vector-scale -1.0 row)))))
    (setq prototypes
          (photon-nth-set prototypes target
                          (photon-vector-squash
                           (photon-vector-add row delta))))
    (setq counts (photon-nth-set counts target (1+ count)))
    (photon-model-set-target-prototypes model prototypes)
    (photon-model-set-target-counts model counts)))

(defun photon-update-target-prototype-signal (model signal target
                                                    learning-rate)
  (let* ((prototypes (photon-model-target-prototypes model))
         (counts (photon-model-target-counts model))
         (row (nth target prototypes))
         (count (nth target counts))
         (rate (/ (* learning-rate
                     (photon-wordpiece-update-scale model target))
                  (sqrt (float (1+ count)))))
         (delta (photon-vector-scale rate
                                     (photon-vector-add
                                      signal
                                      (photon-vector-scale -1.0 row)))))
    (setq prototypes
          (photon-nth-set prototypes target
                          (photon-vector-squash
                           (photon-vector-add row delta))))
    (setq counts (photon-nth-set counts target (1+ count)))
    (photon-model-set-target-prototypes model prototypes)
    (photon-model-set-target-counts model counts)))

(defun photon-update-target-state-cluster (model state target)
  (let* ((clusters (photon-model-target-state-clusters model))
         (cell (assq target clusters))
         (limit (photon-model-target-state-cluster-limit model))
         (states (if cell (cdr cell) nil)))
    (setq states (photon-target-state-cluster-trim
                  (cons state states)
                  limit))
    (if cell
        (setcdr cell states)
      (setq clusters (cons (cons target states) clusters)))
    (photon-model-set-target-state-clusters model clusters)))

(defun photon-target-state-cluster-add (clusters state target limit)
  (let* ((cell (assq target clusters))
         (states (if cell (cdr cell) nil)))
    (setq states (photon-target-state-cluster-trim
                  (cons state states)
                  limit))
    (if cell
        (setcdr cell states)
      (setq clusters (cons (cons target states) clusters)))
    clusters))

(defun photon-rebuild-target-state-clusters (model tokens context-size)
  (let ((clusters nil)
        (limit (photon-model-target-state-cluster-limit model))
        (window nil)
        (remaining tokens)
        (filled 0))
    (while (and remaining (< filled context-size))
      (setq window (cons (car remaining) window))
      (setq remaining (cdr remaining))
      (setq filled (1+ filled)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while remaining
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (target (car remaining)))
          (setq clusters
                (photon-target-state-cluster-add
                 clusters state target limit))
          (setq window (photon-window-slide window target))
          (setq remaining (cdr remaining)))))
    (photon-model-set-target-state-clusters model clusters)))

(defun photon-rebuild-target-state-clusters-token-reader (model next-token
                                                                context-size)
  (let ((clusters nil)
        (limit (photon-model-target-state-cluster-limit model))
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window)))
          (setq clusters
                (photon-target-state-cluster-add
                 clusters state token limit))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (photon-model-set-target-state-clusters model clusters)))

(defun photon-distill-output-head-from-cluster-token-reader
    (model next-token context-size learning-rate)
  (let ((correct-before 0)
        (correct-after 0)
        (teacher-correct 0)
        (updated 0)
        (total 0)
        (margin-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (margin-target (photon-model-param-get model 'next-token-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (teacher-logits
                (photon-model-output-head-state-cluster-logits model state))
               (teacher-prediction (photon-argmax-index teacher-logits))
               (raw-logits (photon-model-output-head-logits model state))
               (raw-prediction (photon-argmax-index raw-logits))
               (raw-rival
                (photon-best-non-target-index raw-logits token))
               (raw-margin
                (photon-target-margin raw-logits token raw-rival)))
          (setq total (1+ total))
          (setq margin-sum (+ margin-sum raw-margin))
          (when (= raw-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (= teacher-prediction token)
            (setq teacher-correct (1+ teacher-correct))
            (photon-update-output-head-projection
             model state (* 0.05 learning-rate))
            (photon-update-output-head-prototype model state token)
            (when (or (/= raw-prediction token)
                      (< raw-margin margin-target))
              (let* ((gap (max 0.0 (- margin-target raw-margin)))
                     (scale (+ 1.0 (/ gap (max 0.001 margin-target)))))
                (photon-update-output-head
                 model state raw-prediction token
                 (* learning-rate scale))
                (setq updated (1+ updated)))))
          (let* ((after-logits (photon-model-output-head-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'teacher-correct teacher-correct)
          (cons 'updated updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'teacher-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 teacher-correct) total)))
          (cons 'average-margin-before
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-distill-output-head-from-cluster-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-output-head-from-cluster-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-compress-output-head-prototypes-one (model learning-rate)
  (let ((prototypes (photon-model-output-head-prototypes model))
        (correct-before 0)
        (correct-after 0)
        (updated 0)
        (total 0)
        (margin-sum 0.0)
        (margin-target (photon-model-param-get model 'next-token-margin)))
    (while prototypes
      (let ((target (car (car prototypes)))
            (states (cdr (car prototypes))))
        (while states
          (let* ((state (car states))
                 (logits (photon-model-output-head-logits model state))
                 (prediction (photon-argmax-index logits))
                 (rival (photon-best-non-target-index logits target))
                 (margin (photon-target-margin logits target rival)))
            (setq total (1+ total))
            (setq margin-sum (+ margin-sum margin))
            (when (= prediction target)
              (setq correct-before (1+ correct-before)))
            (when (or (/= prediction target)
                      (< margin margin-target))
              (let* ((gap (max 0.0 (- margin-target margin)))
                     (scale (+ 1.0 (/ gap (max 0.001 margin-target)))))
                (photon-update-output-head
                 model state prediction target (* learning-rate scale))
                (photon-update-output-head-projection
                 model state (* 0.02 learning-rate scale))
                (setq updated (1+ updated))))
            (let* ((after-logits (photon-model-output-head-logits model state))
                   (after-prediction (photon-argmax-index after-logits)))
              (when (= after-prediction target)
                (setq correct-after (1+ correct-after)))))
          (setq states (cdr states))))
      (setq prototypes (cdr prototypes)))
    (list (cons 'total total)
          (cons 'updated updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'average-margin-before
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-compress-output-head-prototypes (model learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-compress-output-head-prototypes-one
             model learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-evaluate-linear-output-head-token-reader
    (model next-token context-size)
  (let ((correct 0)
        (total 0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (prediction
                (photon-argmax-index
                 (photon-model-output-head-logits model state))))
          (when (= prediction token)
            (setq correct (1+ correct)))
          (setq total (1+ total))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total))))))

(defun photon-evaluate-output-head-prototype-token-reader
    (model next-token context-size)
  (let ((correct 0)
        (total 0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (prediction
                (photon-argmax-index
                 (photon-model-output-head-prototype-logits model state))))
          (when (= prediction token)
            (setq correct (1+ correct)))
          (setq total (1+ total))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total))))))

(defun photon-evaluate-linear-output-head-shape-token-reader
    (model next-token context-size shape)
  (let ((correct 0)
        (total 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (when (and vocab
                   (eq (photon-wordpiece-token-shape vocab token) shape))
          (let* ((states (photon-forward-model model window))
                 (state (photon-prediction-state states window))
                 (prediction
                  (photon-argmax-index
                   (photon-model-output-head-logits model state))))
            (when (= prediction token)
              (setq correct (1+ correct)))
            (setq total (1+ total))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (list (cons 'shape shape)
          (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total))))))

(defun photon-evaluate-output-head-prototype-shape-token-reader
    (model next-token context-size shape)
  (let ((correct 0)
        (top-3-correct 0)
        (top-5-correct 0)
        (total 0)
        (rank-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (when (and vocab
                   (eq (photon-wordpiece-token-shape vocab token) shape))
          (let* ((states (photon-forward-model model window))
                 (state (photon-prediction-state states window))
                 (logits
                  (photon-model-output-head-prototype-logits model state))
                 (prediction
                  (photon-argmax-index logits))
                 (rank (photon-logit-rank logits token)))
            (when (= prediction token)
              (setq correct (1+ correct)))
            (when (<= rank 3)
              (setq top-3-correct (1+ top-3-correct)))
            (when (<= rank 5)
              (setq top-5-correct (1+ top-5-correct)))
            (setq rank-sum (+ rank-sum rank))
            (setq total (1+ total))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (list (cons 'shape shape)
          (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'top-3-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
          (cons 'top-5-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
          (cons 'average-target-rank
                (if (= total 0) 0.0 (/ rank-sum total))))))

(defun photon-compress-output-head-prototypes-pocket
    (model learning-rate epochs eval-token-reader-factory context-size)
  (let* ((initial-score
          (photon-evaluate-linear-output-head-token-reader
           model (funcall eval-token-reader-factory) context-size))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (best-head (copy-tree (photon-model-output-head model)))
         (best-bias (copy-tree (photon-model-output-head-bias model)))
         (best-projection (copy-tree (photon-model-output-head-projection model)))
         (history nil)
         (epoch 0)
         (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-compress-output-head-prototypes-one
             model learning-rate))
      (let* ((score
              (photon-evaluate-linear-output-head-token-reader
               model (funcall eval-token-reader-factory) context-size))
             (accuracy (cdr (assq 'accuracy-permil score))))
        (when (> accuracy best-accuracy)
          (setq best-accuracy accuracy)
          (setq best-head (copy-tree (photon-model-output-head model)))
          (setq best-bias (copy-tree (photon-model-output-head-bias model)))
          (setq best-projection
                (copy-tree (photon-model-output-head-projection model))))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list (cons 'eval-linear score)
                                        (cons 'best-linear-accuracy-permil
                                              best-accuracy))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head model best-head)
    (photon-model-set-output-head-bias model best-bias)
    (photon-model-set-output-head-projection model best-projection)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'initial-linear initial-score)
          (cons 'best-linear-accuracy-permil best-accuracy))))

(defun photon-distill-token-readout-to-output-head-token-reader
    (model next-token context-size learning-rate)
  (let ((correct-before 0)
        (correct-after 0)
        (teacher-correct 0)
        (updated 0)
        (total 0)
        (margin-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (margin-target (photon-model-param-get model 'next-token-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components (photon-model-readout-components model state window))
               (teacher-logits (cdr (assq 'token-readout components)))
               (teacher-prediction (photon-argmax-index teacher-logits))
               (linear-logits (photon-model-output-head-logits model state))
               (linear-prediction (photon-argmax-index linear-logits))
               (linear-rival
                (photon-best-non-target-index linear-logits token))
               (linear-margin
                (photon-target-margin linear-logits token linear-rival)))
          (setq total (1+ total))
          (setq margin-sum (+ margin-sum linear-margin))
          (when (= linear-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (= teacher-prediction token)
            (setq teacher-correct (1+ teacher-correct))
            (when (or (/= linear-prediction token)
                      (< linear-margin margin-target))
              (let* ((gap (max 0.0 (- margin-target linear-margin)))
                     (scale (+ 1.0 (/ gap (max 0.001 margin-target)))))
                (photon-update-output-head
                 model state linear-prediction token
                 (* learning-rate scale))
                (photon-update-output-head-projection
                 model state (* 0.03 learning-rate scale))
                (setq updated (1+ updated)))))
          (let* ((after-logits (photon-model-output-head-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'teacher-correct teacher-correct)
          (cons 'updated updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'teacher-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 teacher-correct) total)))
          (cons 'average-margin-before
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-distill-token-readout-to-output-head-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-token-readout-to-output-head-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-distill-token-readout-to-output-head-token-reader-pocket
    (model token-reader-factory context-size learning-rate epochs)
  (let* ((initial-score
          (photon-evaluate-linear-output-head-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-prototype-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (best-prototype-accuracy
          (cdr (assq 'accuracy-permil initial-prototype-score)))
         (best-selection-score (+ best-accuracy best-prototype-accuracy))
         (best-head (copy-tree (photon-model-output-head model)))
         (best-bias (copy-tree (photon-model-output-head-bias model)))
         (best-projection (copy-tree (photon-model-output-head-projection model)))
         (history nil)
         (epoch 0)
         (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-token-readout-to-output-head-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (let* ((score
              (photon-evaluate-linear-output-head-token-reader
               model (funcall token-reader-factory) context-size))
             (prototype-score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (prototype-accuracy
              (cdr (assq 'accuracy-permil prototype-score)))
             (selection-score (+ accuracy prototype-accuracy)))
        (when (> selection-score best-selection-score)
          (setq best-accuracy accuracy)
          (setq best-prototype-accuracy prototype-accuracy)
          (setq best-selection-score selection-score)
          (setq best-head (copy-tree (photon-model-output-head model)))
          (setq best-bias (copy-tree (photon-model-output-head-bias model)))
          (setq best-projection
                (copy-tree (photon-model-output-head-projection model))))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list (cons 'eval-linear score)
                                        (cons 'eval-output-head-prototype
                                              prototype-score)
                                        (cons 'best-linear-accuracy-permil
                                              best-accuracy)
                                        (cons 'best-prototype-accuracy-permil
                                              best-prototype-accuracy)
                                        (cons 'best-selection-score
                                              best-selection-score))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head model best-head)
    (photon-model-set-output-head-bias model best-bias)
    (photon-model-set-output-head-projection model best-projection)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'initial-linear initial-score)
          (cons 'initial-output-head-prototype initial-prototype-score)
          (cons 'best-linear-accuracy-permil best-accuracy)
          (cons 'best-prototype-accuracy-permil best-prototype-accuracy)
          (cons 'best-selection-score best-selection-score))))

(defun photon-wordpiece-rescue-target-p (model token)
  (let ((vocab (photon-model-metadata-get model 'vocab))
        (counts (photon-model-target-counts model)))
    (and (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
         vocab
         (or (eq (photon-wordpiece-token-kind vocab token) 'long-piece)
             (<= (or (nth token counts) 0) 2)))))

(defun photon-distill-wordpiece-ngram-rescue-token-reader
    (model next-token context-size learning-rate)
  (let ((total 0)
        (candidates 0)
        (updated 0)
        (correct-before 0)
        (correct-after 0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components (photon-model-readout-components model state window))
               (full-logits (cdr (assq 'full components)))
               (ngram-logits (cdr (assq 'ngram-anchor components)))
               (prediction (photon-argmax-index full-logits))
               (top-3 (photon-top-k-logits ngram-logits 3)))
          (setq total (1+ total))
          (when (= prediction token)
            (setq correct-before (1+ correct-before)))
          (when (and (/= prediction token)
                     (photon-wordpiece-rescue-target-p model token)
                     (photon-top-k-hit-p top-3 token))
            (setq candidates (1+ candidates))
            (photon-update-output-head
             model state prediction token learning-rate)
            (photon-update-target-prototype-signal
             model state token (* 0.50 learning-rate))
            (photon-update-output-head-prototype model state token)
            (setq updated (1+ updated)))
          (let* ((after-components
                  (photon-model-readout-components model state window))
                 (after-prediction
                  (photon-argmax-index (cdr (assq 'full after-components)))))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total))))))

(defun photon-distill-wordpiece-ngram-rescue-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-wordpiece-ngram-rescue-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-final-wordpiece-ngram-rescue-token-reader
    (model next-token context-size learning-rate)
  (let ((total 0)
        (candidates 0)
        (updated 0)
        (correct-before 0)
        (correct-after 0)
        (long-updated 0)
        (rare-updated 0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (logits (photon-model-readout-logits model state window))
               (prediction (photon-argmax-index logits))
               (rival (photon-best-non-target-index logits token))
               (margin (photon-target-margin logits token rival))
               (vocab (photon-model-metadata-get model 'vocab))
               (counts (photon-model-target-counts model))
               (count (or (nth token counts) 0))
               (long-p (and vocab
                            (eq (photon-wordpiece-token-kind vocab token)
                                'long-piece)))
               (rare-p (<= count 2)))
          (setq total (1+ total))
          (when (= prediction token)
            (setq correct-before (1+ correct-before)))
          (when (and (/= prediction token)
                     (photon-wordpiece-rescue-target-p model token))
            (setq candidates (1+ candidates))
            (photon-update-context-anchor-ngram-contrastive
             model window token prediction margin learning-rate)
            (when long-p
              (setq long-updated (1+ long-updated)))
            (when rare-p
              (setq rare-updated (1+ rare-updated)))
            (setq updated (1+ updated)))
          (let* ((after-states (photon-forward-model model window))
                 (after-state
                  (photon-prediction-state after-states window))
                 (after-logits
                  (photon-model-readout-logits model after-state window))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'long-updated long-updated)
          (cons 'rare-updated rare-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total))))))

(defun photon-final-wordpiece-ngram-rescue-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-final-wordpiece-ngram-rescue-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-wordpiece-long-rare-readout-rescue-token-reader
    (model next-token context-size learning-rate)
  (let ((total 0)
        (candidates 0)
        (updated 0)
        (correct-before 0)
        (correct-after 0)
        (long-updated 0)
        (rare-updated 0)
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components
                (photon-model-readout-components model state window))
               (logits (cdr (assq 'full components)))
               (prediction (photon-argmax-index logits))
               (rival (photon-best-non-target-index logits token))
               (margin (photon-target-margin logits token rival))
               (vocab (photon-model-metadata-get model 'vocab))
               (counts (photon-model-target-counts model))
               (count (or (nth token counts) 0))
               (long-p (and vocab
                            (eq (photon-wordpiece-token-kind vocab token)
                                'long-piece)))
               (rare-p (<= count 2)))
          (setq total (1+ total))
          (when (= prediction token)
            (setq correct-before (1+ correct-before)))
          (when (and (/= prediction token)
                     (or long-p rare-p)
                     (photon-wordpiece-rescue-target-p model token))
            (let ((amount (* learning-rate
                             (photon-wordpiece-update-scale model token))))
              (setq candidates (1+ candidates))
              (photon-update-context-anchor-ngram-weighted
               model window token amount)
              (photon-update-context-anchor-ngram-contrastive
               model window token prediction margin learning-rate)
              (when long-p
                (setq long-updated (1+ long-updated)))
              (when rare-p
                (setq rare-updated (1+ rare-updated)))
              (setq updated (1+ updated))))
          (let* ((after-states (photon-forward-model model window))
                 (after-state
                  (photon-prediction-state after-states window))
                 (after-logits
                  (photon-model-readout-logits model after-state window))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'long-updated long-updated)
          (cons 'rare-updated rare-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total))))))

(defun photon-wordpiece-long-rare-readout-rescue-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-wordpiece-long-rare-readout-rescue-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-distill-readout-to-output-head-prototypes-token-reader
    (model next-token context-size &optional teacher-key target-kind)
  (let ((total 0)
        (eligible 0)
        (teacher-correct 0)
        (updated 0)
        (correct-before 0)
        (correct-after 0)
        (window nil)
        (filled 0)
        (token nil)
        (teacher-key (or teacher-key 'full)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
         (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components (photon-model-readout-components model state window))
               (vocab (photon-model-metadata-get model 'vocab))
               (kind (and vocab
                          (photon-wordpiece-token-kind vocab token)))
               (eligible-p (or (null target-kind)
                               (eq kind target-kind)))
               (teacher-logits (or (cdr (assq teacher-key components))
                                   (cdr (assq 'full components))))
               (teacher-prediction (photon-argmax-index teacher-logits))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits)))
          (setq total (1+ total))
          (when eligible-p
            (setq eligible (1+ eligible)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (and eligible-p (= teacher-prediction token))
            (setq teacher-correct (1+ teacher-correct))
            (photon-update-output-head-prototype model state token)
            (setq updated (1+ updated)))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'target-kind (or target-kind 'all))
          (cons 'eligible eligible)
          (cons 'teacher-key teacher-key)
          (cons 'teacher-correct teacher-correct)
          (cons 'updated updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'teacher-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 teacher-correct) total))))))

(defun photon-distill-readout-to-output-head-prototypes-token-reader-epochs
    (model token-reader-factory context-size epochs &optional teacher-key
           target-kind)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-readout-to-output-head-prototypes-token-reader
             model
             (funcall token-reader-factory)
             context-size
             teacher-key
             target-kind))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-distill-ngram-long-to-output-head-prototypes-token-reader
    (model next-token context-size)
  (let ((total 0)
        (long-total 0)
        (ngram-correct 0)
        (candidates 0)
        (updated 0)
        (negative-updated 0)
        (state-path-updated 0)
        (correct-before 0)
        (correct-after 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (bias-weight
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-bias-weight))
        (contrastive-rate
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-contrastive-rate))
        (state-path-rate
         (photon-model-param-get
          model 'output-head-prototype-ngram-long-state-path-rate)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components (photon-model-readout-components model state window))
               (ngram-logits (cdr (assq 'ngram-anchor components)))
               (ngram-prediction
                (if ngram-logits (photon-argmax-index ngram-logits) 0))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits))
               (long-p
                (and vocab
                     (eq (photon-wordpiece-token-kind vocab token)
                         'long-piece))))
          (setq total (1+ total))
          (when long-p
            (setq long-total (1+ long-total)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (= ngram-prediction token)
            (setq ngram-correct (1+ ngram-correct)))
          (when (and long-p
                     (= ngram-prediction token)
                     (/= prototype-prediction token))
            (setq candidates (1+ candidates))
            (photon-update-output-head-prototype
             model
             (photon-ngram-long-output-head-prototype-state
              model state window token)
             token)
            (photon-update-output-head-negative-prototype
             model state prototype-prediction)
            (photon-update-output-head-prototype-bias
             model token bias-weight)
            (when (> contrastive-rate 0.0)
              (photon-update-output-head
               model state prototype-prediction token contrastive-rate)
              (photon-update-output-head-projection
               model state (* 0.03 contrastive-rate))
              (photon-update-embedding-table
               model window prototype-prediction token
               (* 0.25 contrastive-rate))
              (photon-update-layer-projections
               model prototype-prediction token contrastive-rate))
            (when (> state-path-rate 0.0)
              (photon-update-long-piece-state-path
               model window prototype-prediction token state-path-rate)
              (setq state-path-updated (1+ state-path-updated)))
            (setq negative-updated (1+ negative-updated))
            (setq updated (1+ updated)))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'long-total long-total)
          (cons 'ngram-correct ngram-correct)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'negative-updated negative-updated)
          (cons 'state-path-updated state-path-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total))))))

(defun photon-distill-ngram-long-to-output-head-prototypes-token-reader-epochs
    (model token-reader-factory context-size epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-ngram-long-to-output-head-prototypes-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-positive-distill-token-reader
    (model next-token context-size learning-rate)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (updated 0)
        (prototype-updated 0)
        (correct-before 0)
        (correct-after 0)
        (prototype-correct-before 0)
        (prototype-correct-after 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-positive-distill-top-k)))
        (margin-target (photon-model-param-get model 'next-token-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (start-long-p
                (and vocab
                     (eq (photon-wordpiece-token-shape vocab token)
                         'start-long-piece)))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (linear-logits (photon-model-output-head-logits model state))
               (linear-prediction (photon-argmax-index linear-logits))
               (linear-rival
                (photon-best-non-target-index linear-logits token))
               (linear-margin
                (photon-target-margin linear-logits token linear-rival))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits)))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total)))
          (when (= linear-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (= prototype-prediction token)
            (setq prototype-correct-before (1+ prototype-correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit)
            (setq candidates (1+ candidates))
            (photon-update-output-head-prototype model state token)
            (setq prototype-updated (1+ prototype-updated))
            (when (or (/= linear-prediction token)
                      (< linear-margin margin-target))
              (photon-update-output-head
               model state linear-prediction token learning-rate)
              (photon-update-output-head-projection
               model state (* 0.03 learning-rate))
              (setq updated (1+ updated))))
          (let* ((after-logits (photon-model-output-head-logits model state))
                 (after-prediction (photon-argmax-index after-logits))
                 (after-prototype-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prototype-prediction
                  (photon-argmax-index after-prototype-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after)))
            (when (= after-prototype-prediction token)
              (setq prototype-correct-after
                    (1+ prototype-correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'prototype-updated prototype-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'prototype-correct-before prototype-correct-before)
          (cons 'prototype-correct-after prototype-correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'prototype-accuracy-before-permil
                (if (= total 0)
                    0
                  (/ (* 1000 prototype-correct-before) total)))
          (cons 'prototype-accuracy-after-permil
                (if (= total 0)
                    0
                  (/ (* 1000 prototype-correct-after) total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-positive-distill-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-positive-distill-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-projected-prototype-distill-token-reader
    (model next-token context-size)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (projected-prototype-updated 0)
        (correct-before 0)
        (correct-after 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (bias-weight
         (photon-model-param-get
          model 'output-head-start-long-projected-prototype-distill-weight))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-projected-prototype-distill-top-k))))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (start-long-p
                (and vocab
                     (eq (photon-wordpiece-token-shape vocab token)
                         'start-long-piece)))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits)))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit)
            (setq candidates (1+ candidates))
            (photon-update-output-head-projected-prototype
             model state token)
            (when (> bias-weight 0.0)
              (photon-update-output-head-prototype-bias
               model token bias-weight))
            (setq projected-prototype-updated
                  (1+ projected-prototype-updated)))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'bias-weight bias-weight)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'candidates candidates)
          (cons 'projected-prototype-updated projected-prototype-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-projected-prototype-distill-token-reader-epochs
    (model token-reader-factory context-size epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-projected-prototype-distill-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-prototype-merge-state (base state rate)
  (if base
      (photon-vector-squash
       (photon-vector-add
        (photon-vector-scale (- 1.0 rate) base)
        (photon-vector-scale rate state)))
    state))

(defun photon-start-long-prototype-separated-state (state rival-state rate)
  (if (and rival-state (> rate 0.0))
      (photon-vector-squash
       (photon-vector-add
        (photon-vector-add
         state
         (photon-vector-scale rate state))
        (photon-vector-scale
         (- rate) rival-state)))
    state))

(defun photon-start-long-prototype-merge-token-reader
    (model next-token context-size)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (merged 0)
        (target-bias-updated 0)
        (rival-bias-updated 0)
        (rival-negative-updated 0)
        (correct-before 0)
        (correct-after 0)
        (rank-before-sum 0.0)
        (rank-after-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-rate))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-top-k)))
        (target-margin
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-margin))
        (target-bias-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-target-bias-rate))
        (rival-bias-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-rival-bias-rate))
        (score-bonus-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-score-bonus-rate))
        (rival-negative-weight
         (photon-model-param-get
          model
          'output-head-start-long-prototype-merge-rival-negative-weight))
        (rival-limit
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-rivals))))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (shape (and vocab
                           (photon-wordpiece-token-shape vocab token)))
               (start-long-p (eq shape 'start-long-piece))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction (photon-argmax-index prototype-logits))
               (projected-state
                (photon-model-output-head-projected-state model state))
               (same-shape-rival
                (and start-long-p
                     (photon-best-same-wordpiece-shape-rival-index
                      model prototype-logits token shape)))
               (same-shape-rivals
                (and start-long-p
                     (photon-top-same-wordpiece-shape-rival-indices
                      model prototype-logits token shape rival-limit)))
               (rank-before
                (and start-long-p
                     (photon-logit-rank prototype-logits token)))
               (margin-before
                (if same-shape-rival
                    (photon-target-margin
                     prototype-logits token same-shape-rival)
                  0.0)))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total))
            (setq rank-before-sum (+ rank-before-sum rank-before)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit same-shape-rival
                     (< margin-before target-margin)
                     (> rate 0.0))
            (let* ((base-state
                    (photon-output-head-prototype-best-state
                     model projected-state token))
                   (merged-state
                    (photon-start-long-prototype-merge-state
                     base-state state rate)))
              (setq candidates (1+ candidates))
              (photon-update-output-head-prototype
               model merged-state token)
              (setq merged (1+ merged))
              (when (> target-bias-rate 0.0)
                (photon-update-output-head-prototype-bias
                 model token target-bias-rate)
                (setq target-bias-updated
                      (1+ target-bias-updated)))
              (when (> rival-bias-rate 0.0)
                (let ((rivals (or same-shape-rivals
                                  (list same-shape-rival))))
                  (while rivals
                    (photon-update-output-head-prototype-bias
                     model (car rivals) (- rival-bias-rate))
                    (setq rival-bias-updated
                          (1+ rival-bias-updated))
                    (setq rivals (cdr rivals)))))
              (when (> rival-negative-weight 0.0)
                (let ((rivals (or same-shape-rivals
                                  (list same-shape-rival))))
                  (while rivals
                    (photon-update-output-head-negative-prototype
                     model state (car rivals))
                    (setq rival-negative-updated
                          (1+ rival-negative-updated))
                    (setq rivals (cdr rivals)))))))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits))
                 (rank-after
                  (and start-long-p
                       (photon-logit-rank after-logits token))))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after)))
            (when start-long-p
              (setq rank-after-sum (+ rank-after-sum rank-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'target-margin target-margin)
          (cons 'merge-rate rate)
          (cons 'target-bias-rate target-bias-rate)
          (cons 'rival-bias-rate rival-bias-rate)
          (cons 'rival-negative-weight rival-negative-weight)
          (cons 'rivals rival-limit)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'candidates candidates)
          (cons 'merged merged)
          (cons 'target-bias-updated target-bias-updated)
          (cons 'rival-bias-updated rival-bias-updated)
          (cons 'rival-negative-updated rival-negative-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'average-target-rank-before
                (if (= start-long-total 0)
                    0.0
                  (/ rank-before-sum start-long-total)))
          (cons 'average-target-rank-after
                (if (= start-long-total 0)
                    0.0
                  (/ rank-after-sum start-long-total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-prototype-merge-token-reader-epochs
    (model token-reader-factory context-size epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-prototype-merge-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-prototype-merge-candidates-token-reader
    (model next-token context-size)
  (let ((candidates nil)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-rate))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-top-k)))
        (target-margin
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-margin))
        (candidate-limit
         (truncate
          (photon-model-param-get
           model
           'output-head-start-long-prototype-merge-candidate-limit)))
        (rival-limit
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-rivals))))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (shape (and vocab
                           (photon-wordpiece-token-shape vocab token)))
               (start-long-p (eq shape 'start-long-piece))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (and start-long-p
                     (photon-model-output-head-prototype-logits model state)))
               (projected-state
                (and start-long-p
                     (photon-model-output-head-projected-state model state)))
               (same-shape-rival
                (and prototype-logits
                     (photon-best-same-wordpiece-shape-rival-index
                      model prototype-logits token shape)))
               (same-shape-rivals
                (and prototype-logits
                     (photon-top-same-wordpiece-shape-rival-indices
                      model prototype-logits token shape rival-limit)))
               (rank-before
                (and start-long-p
                     (photon-logit-rank prototype-logits token)))
               (margin-before
                (if same-shape-rival
                    (photon-target-margin
                     prototype-logits token same-shape-rival)
                  0.0)))
          (when start-long-p
            (setq start-long-total (1+ start-long-total)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit same-shape-rival
                     (< margin-before target-margin)
                     (< (length candidates) candidate-limit)
                     (> rate 0.0))
            (let* ((base-state
                    (photon-output-head-prototype-best-state
                     model projected-state token))
                   (merged-state
                    (photon-start-long-prototype-merge-state
                     base-state state rate)))
              (setq candidates
                    (cons (list (cons 'target token)
                                (cons 'rivals
                                      (or same-shape-rivals
                                          (list same-shape-rival)))
                                (cons 'state state)
                                (cons 'merged-state merged-state)
                                (cons 'rank-before rank-before)
                                (cons 'margin-before margin-before))
                          candidates)))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (setq candidates (nreverse candidates))
    (list (cons 'candidates candidates)
          (cons 'candidate-count (length candidates))
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'target-margin target-margin)
          (cons 'merge-rate rate)
          (cons 'candidate-limit candidate-limit)
          (cons 'rivals rival-limit)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-prototype-merge-rival-counts (candidates)
  (let ((counts nil))
    (while candidates
      (let ((rivals (cdr (assq 'rivals (car candidates)))))
        (while rivals
          (setq counts (photon-token-counts-inc counts (car rivals)))
          (setq rivals (cdr rivals))))
      (setq candidates (cdr candidates)))
    counts))

(defun photon-start-long-prototype-merge-dominant-rivals (candidates min-count)
  (let ((counts (photon-start-long-prototype-merge-rival-counts candidates))
        (dominant nil))
    (while counts
      (when (>= (cdr (car counts)) min-count)
        (setq dominant (cons (car counts) dominant)))
      (setq counts (cdr counts)))
    dominant))

(defun photon-start-long-prototype-merge-dominant-rival-p
    (token dominant-rivals)
  (assq token dominant-rivals))

(defun photon-apply-start-long-prototype-merge-candidate
    (model candidate &optional dominant-rivals)
  (let ((target (cdr (assq 'target candidate)))
        (rivals (cdr (assq 'rivals candidate)))
        (state (cdr (assq 'state candidate)))
        (merged-state (cdr (assq 'merged-state candidate)))
        (target-bias-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-target-bias-rate))
        (rival-bias-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-rival-bias-rate))
        (score-bonus-rate
         (photon-model-param-get
          model 'output-head-start-long-prototype-merge-score-bonus-rate))
        (dominant-rival-bias-rate
         (photon-model-param-get
          model
          'output-head-start-long-prototype-merge-dominant-rival-bias-rate))
        (state-separation-rate
         (photon-model-param-get
          model
          'output-head-start-long-prototype-merge-state-separation-rate))
        (dense-head-rate
         (photon-model-param-get
          model
          'output-head-start-long-prototype-merge-dense-head-rate))
        (rival-negative-weight
         (photon-model-param-get
          model
          'output-head-start-long-prototype-merge-rival-negative-weight))
        (target-bias-updated 0)
        (rival-bias-updated 0)
        (score-bonus-updated 0)
        (dominant-rival-bias-updated 0)
        (state-separation-updated 0)
        (dense-head-updated 0)
        (rival-negative-updated 0))
    (photon-update-output-head-prototype model merged-state target)
    (when (> target-bias-rate 0.0)
      (photon-update-output-head-prototype-bias
       model target target-bias-rate)
      (setq target-bias-updated (1+ target-bias-updated)))
    (when (> score-bonus-rate 0.0)
      (let* ((rank (or (cdr (assq 'rank-before candidate)) 1))
             (scale (max 1.0 (/ (float rank) 3.0))))
        (photon-update-output-head-prototype-score-bonus
         model target (* score-bonus-rate scale))
        (setq score-bonus-updated (1+ score-bonus-updated))))
    (while rivals
      (when (> rival-bias-rate 0.0)
        (photon-update-output-head-prototype-bias
         model (car rivals) (- rival-bias-rate))
        (setq rival-bias-updated (1+ rival-bias-updated)))
      (when (and (> dominant-rival-bias-rate 0.0)
                 (photon-start-long-prototype-merge-dominant-rival-p
                  (car rivals) dominant-rivals))
        (photon-update-output-head-prototype-bias
         model (car rivals) (- dominant-rival-bias-rate))
        (setq dominant-rival-bias-updated
              (1+ dominant-rival-bias-updated)))
      (when (> state-separation-rate 0.0)
        (let* ((rival-state
                (photon-output-head-prototype-best-state
                 model state (car rivals)))
               (separated-state
                (photon-start-long-prototype-separated-state
                 state rival-state state-separation-rate)))
          (when rival-state
            (photon-update-output-head-prototype
             model separated-state target)
            (setq state-separation-updated
                  (1+ state-separation-updated)))))
      (when (> dense-head-rate 0.0)
        (photon-update-output-head model state (car rivals)
                                   target dense-head-rate)
        (setq dense-head-updated (1+ dense-head-updated)))
      (when (> rival-negative-weight 0.0)
        (photon-update-output-head-negative-prototype
         model state (car rivals))
        (setq rival-negative-updated
              (1+ rival-negative-updated)))
      (setq rivals (cdr rivals)))
    (list (cons 'target-bias-updated target-bias-updated)
          (cons 'rival-bias-updated rival-bias-updated)
          (cons 'score-bonus-updated score-bonus-updated)
          (cons 'dominant-rival-bias-updated
                dominant-rival-bias-updated)
          (cons 'state-separation-updated state-separation-updated)
          (cons 'dense-head-updated dense-head-updated)
          (cons 'rival-negative-updated rival-negative-updated))))

(defun photon-start-long-prototype-merge-accept-score-p
    (accuracy start-long-accuracy start-long-top-3 start-long-top-5
              start-long-rank best-accuracy accuracy-floor
              best-start-long-accuracy best-start-long-top-3
              best-start-long-top-5 best-start-long-rank)
  (or (> start-long-accuracy best-start-long-accuracy)
      (and (= start-long-accuracy best-start-long-accuracy)
           (> accuracy best-accuracy))
      (and (= start-long-accuracy best-start-long-accuracy)
           (>= accuracy accuracy-floor)
           (> start-long-top-3 best-start-long-top-3))
      (and (= start-long-accuracy best-start-long-accuracy)
           (= start-long-top-3 best-start-long-top-3)
           (>= accuracy accuracy-floor)
           (> start-long-top-5 best-start-long-top-5))
      (and (= start-long-accuracy best-start-long-accuracy)
           (>= accuracy accuracy-floor)
           (= start-long-top-3 best-start-long-top-3)
           (= start-long-top-5 best-start-long-top-5)
           (< start-long-rank best-start-long-rank))))

(defun photon-start-long-prototype-merge-rival-line-search
    (model token-reader-factory context-size candidate best-accuracy
           accuracy-floor best-start-long-accuracy best-start-long-top-3
           best-start-long-top-5 best-start-long-rank)
  (let* ((step
          (photon-model-param-get
           model
           'output-head-start-long-prototype-merge-rival-line-search-step))
         (max-bias
          (photon-model-param-get
           model
           'output-head-start-long-prototype-merge-rival-line-search-max))
         (rivals (cdr (assq 'rivals candidate)))
         (initial-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-line-bias initial-bias)
         (best-line-amount 0.0)
         (best-line-accuracy best-accuracy)
         (best-line-start-long-accuracy best-start-long-accuracy)
         (best-line-start-long-top-3 best-start-long-top-3)
         (best-line-start-long-top-5 best-start-long-top-5)
         (best-line-start-long-rank best-start-long-rank)
         (attempted 0)
         (accepted 0)
         (amount step))
    (while (and (> step 0.0) (<= amount max-bias))
      (setq attempted (1+ attempted))
      (photon-model-set-output-head-prototype-bias
       model (copy-tree initial-bias))
      (let ((scan rivals))
        (while scan
          (photon-update-output-head-prototype-bias
           model (car scan) (- amount))
          (setq scan (cdr scan))))
      (let* ((score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-output-head-prototype-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score)))
             (start-long-top-3
              (cdr (assq 'top-3-accuracy-permil start-long-score)))
             (start-long-top-5
              (cdr (assq 'top-5-accuracy-permil start-long-score)))
             (start-long-rank
              (cdr (assq 'average-target-rank start-long-score))))
        (when (photon-start-long-prototype-merge-accept-score-p
               accuracy start-long-accuracy start-long-top-3 start-long-top-5
               start-long-rank best-line-accuracy accuracy-floor
               best-line-start-long-accuracy best-line-start-long-top-3
               best-line-start-long-top-5 best-line-start-long-rank)
          (setq accepted (1+ accepted))
          (setq best-line-amount amount)
          (setq best-line-accuracy accuracy)
          (setq best-line-start-long-accuracy start-long-accuracy)
          (setq best-line-start-long-top-3 start-long-top-3)
          (setq best-line-start-long-top-5 start-long-top-5)
          (setq best-line-start-long-rank start-long-rank)
          (setq best-line-bias
                (copy-tree
                 (photon-model-output-head-prototype-bias model)))))
      (setq amount (+ amount step)))
    (photon-model-set-output-head-prototype-bias
     model (copy-tree best-line-bias))
    (list (cons 'attempted attempted)
          (cons 'accepted accepted)
          (cons 'amount best-line-amount)
          (cons 'accuracy best-line-accuracy)
          (cons 'start-long-accuracy best-line-start-long-accuracy)
          (cons 'start-long-top-3 best-line-start-long-top-3)
          (cons 'start-long-top-5 best-line-start-long-top-5)
          (cons 'start-long-rank best-line-start-long-rank))))

(defun photon-start-long-prototype-merge-candidate-search
    (model token-reader-factory context-size)
  (let* ((candidate-report
          (photon-start-long-prototype-merge-candidates-token-reader
           model (funcall token-reader-factory) context-size))
         (candidates (cdr (assq 'candidates candidate-report)))
         (dominant-rival-min-count
          (truncate
           (photon-model-param-get
            model
            'output-head-start-long-prototype-merge-dominant-rival-min-count)))
         (dominant-rivals
          (photon-start-long-prototype-merge-dominant-rivals
           candidates dominant-rival-min-count))
         (initial-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-output-head-prototype-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (global-tolerance
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-global-tolerance))
         (accuracy-floor (max 0 (- best-accuracy global-tolerance)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-start-long-top-3
          (cdr (assq 'top-3-accuracy-permil initial-start-long-score)))
         (best-start-long-top-5
          (cdr (assq 'top-5-accuracy-permil initial-start-long-score)))
         (best-start-long-rank
          (cdr (assq 'average-target-rank initial-start-long-score)))
         (best-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
         (best-prototype-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-negative-prototypes
          (copy-tree (photon-model-output-head-negative-prototypes model)))
         (best-prototype-score-bonus
          (copy-tree (photon-model-output-head-prototype-score-bonus model)))
         (best-output-head
          (copy-tree (photon-model-output-head model)))
         (best-output-head-bias
          (copy-tree (photon-model-output-head-bias model)))
         (tested 0)
         (accepted 0)
         (rejected 0)
         (target-bias-updated 0)
         (rival-bias-updated 0)
         (score-bonus-updated 0)
         (dominant-rival-bias-updated 0)
         (state-separation-updated 0)
         (dense-head-updated 0)
         (rival-negative-updated 0)
         (line-search-attempted 0)
         (line-search-accepted 0))
    (while candidates
      (setq tested (1+ tested))
      (photon-model-set-output-head-prototypes
       model (copy-tree best-prototypes))
      (photon-model-set-output-head-prototype-bias
       model (copy-tree best-prototype-bias))
      (photon-model-set-output-head-negative-prototypes
       model (copy-tree best-negative-prototypes))
      (photon-model-set-output-head-prototype-score-bonus
       model (copy-tree best-prototype-score-bonus))
      (photon-model-set-output-head model (copy-tree best-output-head))
      (photon-model-set-output-head-bias
       model (copy-tree best-output-head-bias))
      (let ((apply-report
             (photon-apply-start-long-prototype-merge-candidate
              model (car candidates) dominant-rivals)))
        (let* ((score
                (photon-evaluate-output-head-prototype-token-reader
                 model (funcall token-reader-factory) context-size))
               (start-long-score
                (photon-evaluate-output-head-prototype-shape-token-reader
                 model (funcall token-reader-factory) context-size
                 'start-long-piece))
               (accuracy (cdr (assq 'accuracy-permil score)))
               (start-long-accuracy
                (cdr (assq 'accuracy-permil start-long-score)))
               (start-long-top-3
                (cdr (assq 'top-3-accuracy-permil start-long-score)))
               (start-long-top-5
                (cdr (assq 'top-5-accuracy-permil start-long-score)))
               (start-long-rank
                (cdr (assq 'average-target-rank start-long-score))))
          (when (> (photon-model-param-get
                    model
                    'output-head-start-long-prototype-merge-rival-line-search-step)
                   0.0)
            (let ((line-report
                   (photon-start-long-prototype-merge-rival-line-search
                    model token-reader-factory context-size (car candidates)
                    accuracy accuracy-floor start-long-accuracy
                    start-long-top-3 start-long-top-5 start-long-rank)))
              (setq line-search-attempted
                    (+ line-search-attempted
                       (cdr (assq 'attempted line-report))))
              (setq line-search-accepted
                    (+ line-search-accepted
                       (cdr (assq 'accepted line-report))))
              (setq accuracy (cdr (assq 'accuracy line-report)))
              (setq start-long-accuracy
                    (cdr (assq 'start-long-accuracy line-report)))
              (setq start-long-top-3
                    (cdr (assq 'start-long-top-3 line-report)))
              (setq start-long-top-5
                    (cdr (assq 'start-long-top-5 line-report)))
              (setq start-long-rank
                    (cdr (assq 'start-long-rank line-report)))))
          (if (photon-start-long-prototype-merge-accept-score-p
               accuracy start-long-accuracy start-long-top-3 start-long-top-5
               start-long-rank best-accuracy accuracy-floor
               best-start-long-accuracy best-start-long-top-3
               best-start-long-top-5 best-start-long-rank)
              (progn
                (setq accepted (1+ accepted))
                (setq target-bias-updated
                      (+ target-bias-updated
                         (cdr (assq 'target-bias-updated apply-report))))
                (setq rival-bias-updated
                      (+ rival-bias-updated
                         (cdr (assq 'rival-bias-updated apply-report))))
                (setq score-bonus-updated
                      (+ score-bonus-updated
                         (cdr (assq 'score-bonus-updated apply-report))))
                (setq dominant-rival-bias-updated
                      (+ dominant-rival-bias-updated
                         (cdr (assq 'dominant-rival-bias-updated
                                    apply-report))))
                (setq state-separation-updated
                      (+ state-separation-updated
                         (cdr (assq 'state-separation-updated
                                    apply-report))))
                (setq dense-head-updated
                      (+ dense-head-updated
                         (cdr (assq 'dense-head-updated apply-report))))
                (setq rival-negative-updated
                      (+ rival-negative-updated
                         (cdr (assq 'rival-negative-updated
                                    apply-report))))
                (setq best-accuracy accuracy)
                (setq best-start-long-accuracy start-long-accuracy)
                (setq best-start-long-top-3 start-long-top-3)
                (setq best-start-long-top-5 start-long-top-5)
                (setq best-start-long-rank start-long-rank)
                (setq best-prototypes
                      (copy-tree
                       (photon-model-output-head-prototypes model)))
                (setq best-prototype-bias
                      (copy-tree
                       (photon-model-output-head-prototype-bias model)))
                (setq best-negative-prototypes
                      (copy-tree
                       (photon-model-output-head-negative-prototypes
                        model)))
                (setq best-prototype-score-bonus
                      (copy-tree
                       (photon-model-output-head-prototype-score-bonus
                        model)))
                (setq best-output-head
                      (copy-tree (photon-model-output-head model)))
                (setq best-output-head-bias
                      (copy-tree (photon-model-output-head-bias model))))
            (setq rejected (1+ rejected)))))
      (setq candidates (cdr candidates)))
    (photon-model-set-output-head-prototypes model best-prototypes)
    (photon-model-set-output-head-prototype-bias model best-prototype-bias)
    (photon-model-set-output-head-negative-prototypes
     model best-negative-prototypes)
    (photon-model-set-output-head-prototype-score-bonus
     model best-prototype-score-bonus)
    (photon-model-set-output-head model best-output-head)
    (photon-model-set-output-head-bias model best-output-head-bias)
    (list (cons 'candidate-count
                (cdr (assq 'candidate-count candidate-report)))
          (cons 'tested tested)
          (cons 'accepted accepted)
          (cons 'rejected rejected)
          (cons 'target-bias-updated target-bias-updated)
          (cons 'rival-bias-updated rival-bias-updated)
          (cons 'score-bonus-updated score-bonus-updated)
          (cons 'dominant-rival-bias-updated
                dominant-rival-bias-updated)
          (cons 'state-separation-updated state-separation-updated)
          (cons 'dense-head-updated dense-head-updated)
          (cons 'rival-negative-updated rival-negative-updated)
          (cons 'line-search-attempted line-search-attempted)
          (cons 'line-search-accepted line-search-accepted)
          (cons 'initial-output-head-prototype initial-score)
          (cons 'initial-start-long-prototype initial-start-long-score)
          (cons 'best-prototype-accuracy-permil best-accuracy)
          (cons 'accuracy-floor-permil accuracy-floor)
          (cons 'global-tolerance-permil global-tolerance)
          (cons 'dominant-rival-min-count dominant-rival-min-count)
          (cons 'dominant-rivals dominant-rivals)
          (cons 'best-start-long-prototype-accuracy-permil
                best-start-long-accuracy)
          (cons 'best-start-long-top-3-accuracy-permil
                best-start-long-top-3)
          (cons 'best-start-long-top-5-accuracy-permil
                best-start-long-top-5)
          (cons 'best-start-long-average-target-rank
                best-start-long-rank)
          (cons 'candidate-report candidate-report))))

(defun photon-start-long-prototype-merge-pocket
    (model token-reader-factory context-size epochs)
  (let* ((initial-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-output-head-prototype-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (global-tolerance
          (photon-model-param-get
           model 'output-head-start-long-prototype-merge-global-tolerance))
         (accuracy-floor (max 0 (- best-accuracy global-tolerance)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-start-long-top-3
          (cdr (assq 'top-3-accuracy-permil initial-start-long-score)))
         (best-start-long-top-5
          (cdr (assq 'top-5-accuracy-permil initial-start-long-score)))
         (best-start-long-rank
          (cdr (assq 'average-target-rank initial-start-long-score)))
         (best-prototypes
          (copy-tree (photon-model-output-head-prototypes model)))
         (best-prototype-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-negative-prototypes
          (copy-tree (photon-model-output-head-negative-prototypes model)))
         (history nil)
         (epoch 0)
         (last nil)
         (accepted 0)
         (rejected 0))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-prototype-merge-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (let* ((score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-output-head-prototype-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score)))
             (start-long-top-3
              (cdr (assq 'top-3-accuracy-permil start-long-score)))
             (start-long-top-5
              (cdr (assq 'top-5-accuracy-permil start-long-score)))
             (start-long-rank
              (cdr (assq 'average-target-rank start-long-score))))
        (if (or (> start-long-accuracy best-start-long-accuracy)
                (and (= start-long-accuracy best-start-long-accuracy)
                     (> accuracy best-accuracy))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-3 best-start-long-top-3))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (= start-long-top-3 best-start-long-top-3)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-5 best-start-long-top-5))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (= start-long-top-3 best-start-long-top-3)
                     (= start-long-top-5 best-start-long-top-5)
                     (< start-long-rank best-start-long-rank)))
            (progn
              (setq accepted (1+ accepted))
              (setq best-accuracy accuracy)
              (setq best-start-long-accuracy start-long-accuracy)
              (setq best-start-long-top-3 start-long-top-3)
              (setq best-start-long-top-5 start-long-top-5)
              (setq best-start-long-rank start-long-rank)
              (setq best-prototypes
                    (copy-tree
                     (photon-model-output-head-prototypes model)))
              (setq best-prototype-bias
                    (copy-tree
                     (photon-model-output-head-prototype-bias model)))
              (setq best-negative-prototypes
                    (copy-tree
                     (photon-model-output-head-negative-prototypes model))))
          (setq rejected (1+ rejected))
          (photon-model-set-output-head-prototypes
           model (copy-tree best-prototypes))
          (photon-model-set-output-head-prototype-bias
           model (copy-tree best-prototype-bias))
          (photon-model-set-output-head-negative-prototypes
           model (copy-tree best-negative-prototypes)))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list
                                   (cons 'eval-output-head-prototype score)
                                   (cons 'eval-start-long-prototype
                                         start-long-score)
                                   (cons 'accepted-so-far accepted)
                                   (cons 'rejected-so-far rejected)
                                   (cons 'best-prototype-accuracy-permil
                                         best-accuracy)
                                   (cons
                                    'best-start-long-prototype-accuracy-permil
                                    best-start-long-accuracy)
                                   (cons 'best-start-long-top-3-accuracy-permil
                                         best-start-long-top-3)
                                   (cons 'best-start-long-top-5-accuracy-permil
                                         best-start-long-top-5)
                                   (cons 'best-start-long-average-target-rank
                                         best-start-long-rank))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head-prototypes model best-prototypes)
    (photon-model-set-output-head-prototype-bias model best-prototype-bias)
    (photon-model-set-output-head-negative-prototypes
     model best-negative-prototypes)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'accepted accepted)
          (cons 'rejected rejected)
          (cons 'initial-output-head-prototype initial-score)
          (cons 'initial-start-long-prototype initial-start-long-score)
          (cons 'best-prototype-accuracy-permil best-accuracy)
          (cons 'accuracy-floor-permil accuracy-floor)
          (cons 'global-tolerance-permil global-tolerance)
          (cons 'best-start-long-prototype-accuracy-permil
                best-start-long-accuracy)
          (cons 'best-start-long-top-3-accuracy-permil
                best-start-long-top-3)
          (cons 'best-start-long-top-5-accuracy-permil
                best-start-long-top-5)
          (cons 'best-start-long-average-target-rank
                best-start-long-rank))))

(defun photon-start-long-projection-separation-token-reader
    (model next-token context-size)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (updated 0)
        (score-bonus-updated 0)
        (rival-bias-updated 0)
        (correct-before 0)
        (correct-after 0)
        (rank-before-sum 0.0)
        (rank-after-sum 0.0)
        (margin-before-sum 0.0)
        (margin-after-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (rate
         (photon-model-param-get
          model 'output-head-start-long-projection-separation-rate))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-projection-separation-top-k)))
        (target-margin
         (photon-model-param-get
          model 'output-head-start-long-projection-separation-margin))
        (score-bonus-rate
         (photon-model-param-get
          model 'output-head-start-long-projection-separation-score-bonus-rate))
        (rival-bias-rate
         (photon-model-param-get
          model 'output-head-start-long-projection-separation-rival-bias-rate)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (shape (and vocab
                           (photon-wordpiece-token-shape vocab token)))
               (start-long-p (eq shape 'start-long-piece))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction (photon-argmax-index prototype-logits))
               (projected-state
                (photon-model-output-head-projected-state model state))
               (same-shape-rival
                (and start-long-p
                     (photon-best-same-wordpiece-shape-rival-index
                      model prototype-logits token shape)))
               (rank-before
                (and start-long-p
                     (photon-logit-rank prototype-logits token)))
               (margin-before
                (if same-shape-rival
                    (photon-target-margin
                     prototype-logits token same-shape-rival)
                  0.0)))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total))
            (setq rank-before-sum (+ rank-before-sum rank-before))
            (setq margin-before-sum (+ margin-before-sum margin-before)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit same-shape-rival
                     (< margin-before target-margin)
                     (> rate 0.0))
            (let* ((target-state
                    (photon-output-head-prototype-best-state
                     model projected-state token))
                   (rival-state
                    (photon-output-head-prototype-best-state
                     model projected-state same-shape-rival)))
              (when (and target-state rival-state)
                (let* ((target-projected
                        (photon-model-output-head-projected-state
                         model target-state))
                       (rival-projected
                        (photon-model-output-head-projected-state
                         model rival-state))
                       (direction
                        (photon-vector-add
                         target-projected
                         (photon-vector-scale -1.0 rival-projected)))
                       (gap (max 0.0 (- target-margin margin-before)))
                       (scale (+ 1.0
                                 (/ gap (max 0.001 target-margin)))))
                  (setq candidates (1+ candidates))
                  (photon-update-output-head-projection-direction
                   model direction (* rate scale))
                  (when (> score-bonus-rate 0.0)
                    (photon-update-output-head-prototype-score-bonus
                     model token (* score-bonus-rate scale))
                    (setq score-bonus-updated
                          (1+ score-bonus-updated)))
                  (when (> rival-bias-rate 0.0)
                    (photon-update-output-head-prototype-bias
                     model same-shape-rival (- (* rival-bias-rate scale)))
                    (setq rival-bias-updated
                          (1+ rival-bias-updated)))
                  (setq updated (1+ updated))))))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits))
                 (rank-after
                  (and start-long-p
                       (photon-logit-rank after-logits token)))
                 (same-shape-rival-after
                  (and start-long-p
                       (photon-best-same-wordpiece-shape-rival-index
                        model after-logits token shape)))
                 (margin-after
                  (if same-shape-rival-after
                      (photon-target-margin
                       after-logits token same-shape-rival-after)
                    0.0)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after)))
            (when start-long-p
              (setq rank-after-sum (+ rank-after-sum rank-after))
              (setq margin-after-sum (+ margin-after-sum margin-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'target-margin target-margin)
          (cons 'rate rate)
          (cons 'score-bonus-rate score-bonus-rate)
          (cons 'rival-bias-rate rival-bias-rate)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'score-bonus-updated score-bonus-updated)
          (cons 'rival-bias-updated rival-bias-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'average-target-rank-before
                (if (= start-long-total 0)
                    0.0
                  (/ rank-before-sum start-long-total)))
          (cons 'average-target-rank-after
                (if (= start-long-total 0)
                    0.0
                  (/ rank-after-sum start-long-total)))
          (cons 'average-same-shape-margin-before
                (if (= start-long-total 0)
                    0.0
                  (/ margin-before-sum start-long-total)))
          (cons 'average-same-shape-margin-after
                (if (= start-long-total 0)
                    0.0
                  (/ margin-after-sum start-long-total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-projection-separation-token-reader-epochs
    (model token-reader-factory context-size epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-projection-separation-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-projection-separation-pocket
    (model token-reader-factory context-size epochs)
  (let* ((initial-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-output-head-prototype-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (global-tolerance
          (photon-model-param-get
           model
           'output-head-start-long-projection-separation-global-tolerance))
         (accuracy-floor (max 0 (- best-accuracy global-tolerance)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-start-long-top-3
          (cdr (assq 'top-3-accuracy-permil initial-start-long-score)))
         (best-start-long-top-5
          (cdr (assq 'top-5-accuracy-permil initial-start-long-score)))
         (best-start-long-rank
          (cdr (assq 'average-target-rank initial-start-long-score)))
         (best-projection
          (copy-tree (photon-model-output-head-projection model)))
         (best-prototype-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-prototype-score-bonus
          (copy-tree (photon-model-output-head-prototype-score-bonus model)))
         (history nil)
         (epoch 0)
         (last nil)
         (accepted 0)
         (rejected 0))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-projection-separation-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (let* ((score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-output-head-prototype-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score)))
             (start-long-top-3
              (cdr (assq 'top-3-accuracy-permil start-long-score)))
             (start-long-top-5
              (cdr (assq 'top-5-accuracy-permil start-long-score)))
             (start-long-rank
              (cdr (assq 'average-target-rank start-long-score))))
        (if (or (> start-long-accuracy best-start-long-accuracy)
                (and (= start-long-accuracy best-start-long-accuracy)
                     (> accuracy best-accuracy))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-3 best-start-long-top-3))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (= start-long-top-3 best-start-long-top-3)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-5 best-start-long-top-5))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (= start-long-top-3 best-start-long-top-3)
                     (= start-long-top-5 best-start-long-top-5)
                     (< start-long-rank best-start-long-rank)))
            (progn
              (setq accepted (1+ accepted))
              (setq best-accuracy accuracy)
              (setq best-start-long-accuracy start-long-accuracy)
              (setq best-start-long-top-3 start-long-top-3)
              (setq best-start-long-top-5 start-long-top-5)
              (setq best-start-long-rank start-long-rank)
              (setq best-projection
                    (copy-tree (photon-model-output-head-projection model)))
              (setq best-prototype-bias
                    (copy-tree
                     (photon-model-output-head-prototype-bias model)))
              (setq best-prototype-score-bonus
                    (copy-tree
                     (photon-model-output-head-prototype-score-bonus model))))
          (setq rejected (1+ rejected))
          (photon-model-set-output-head-projection
           model (copy-tree best-projection))
          (photon-model-set-output-head-prototype-bias
           model (copy-tree best-prototype-bias))
          (photon-model-set-output-head-prototype-score-bonus
           model (copy-tree best-prototype-score-bonus)))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list
                                   (cons 'eval-output-head-prototype score)
                                   (cons 'eval-start-long-prototype
                                         start-long-score)
                                   (cons 'accepted-so-far accepted)
                                   (cons 'rejected-so-far rejected)
                                   (cons 'best-prototype-accuracy-permil
                                         best-accuracy)
                                   (cons
                                    'best-start-long-prototype-accuracy-permil
                                    best-start-long-accuracy)
                                   (cons 'best-start-long-top-3-accuracy-permil
                                         best-start-long-top-3)
                                   (cons 'best-start-long-top-5-accuracy-permil
                                         best-start-long-top-5)
                                   (cons 'best-start-long-average-target-rank
                                         best-start-long-rank))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head-projection model best-projection)
    (photon-model-set-output-head-prototype-bias model best-prototype-bias)
    (photon-model-set-output-head-prototype-score-bonus
     model best-prototype-score-bonus)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'accepted accepted)
          (cons 'rejected rejected)
          (cons 'initial-output-head-prototype initial-score)
          (cons 'initial-start-long-prototype initial-start-long-score)
          (cons 'best-prototype-accuracy-permil best-accuracy)
          (cons 'accuracy-floor-permil accuracy-floor)
          (cons 'global-tolerance-permil global-tolerance)
          (cons 'best-start-long-prototype-accuracy-permil
                best-start-long-accuracy)
          (cons 'best-start-long-top-3-accuracy-permil
                best-start-long-top-3)
          (cons 'best-start-long-top-5-accuracy-permil
                best-start-long-top-5)
          (cons 'best-start-long-average-target-rank
                best-start-long-rank))))

(defun photon-start-long-rival-negative-distill-token-reader
    (model next-token context-size)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (negative-updated 0)
        (correct-before 0)
        (correct-after 0)
        (target-rank-before-sum 0.0)
        (target-rank-after-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-rival-negative-distill-top-k)))
        (rival-count
         (max 1
              (truncate
               (photon-model-param-get
                model
                'output-head-start-long-rival-negative-distill-rivals)))))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (start-long-p
                (and vocab
                     (eq (photon-wordpiece-token-shape vocab token)
                         'start-long-piece)))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits))
               (rank-before
                (and start-long-p
                     (photon-logit-rank prototype-logits token))))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total))
            (setq target-rank-before-sum
                  (+ target-rank-before-sum rank-before)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit
                     (/= prototype-prediction token))
            (setq candidates (1+ candidates))
            (let ((rivals (photon-top-k-logits
                           prototype-logits
                           (min (length prototype-logits)
                                (1+ rival-count))))
                  (seen 0))
              (while (and rivals (< seen rival-count))
                (unless (= (car (car rivals)) token)
                  (photon-update-output-head-negative-prototype
                   model state (car (car rivals)))
                  (setq negative-updated (1+ negative-updated))
                  (setq seen (1+ seen)))
                (setq rivals (cdr rivals)))))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits))
                 (rank-after
                  (and start-long-p
                       (photon-logit-rank after-logits token))))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after)))
            (when start-long-p
              (setq target-rank-after-sum
                    (+ target-rank-after-sum rank-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'rivals rival-count)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'candidates candidates)
          (cons 'negative-updated negative-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'average-target-rank-before
                (if (= start-long-total 0)
                    0.0
                  (/ target-rank-before-sum start-long-total)))
          (cons 'average-target-rank-after
                (if (= start-long-total 0)
                    0.0
                  (/ target-rank-after-sum start-long-total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-rival-negative-distill-token-reader-epochs
    (model token-reader-factory context-size epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-rival-negative-distill-token-reader
             model
             (funcall token-reader-factory)
             context-size))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-best-same-wordpiece-shape-rival-index
    (model logits target shape)
  (let ((vocab (photon-model-metadata-get model 'vocab))
        (best-index nil)
        (best-value nil)
        (index 0)
        (walk logits))
    (while walk
      (when (and (/= index target)
                 vocab
                 (eq (photon-wordpiece-token-shape vocab index) shape)
                 (or (null best-value) (> (car walk) best-value)))
        (setq best-index index)
        (setq best-value (car walk)))
      (setq walk (cdr walk))
      (setq index (1+ index)))
    best-index))

(defun photon-top-same-wordpiece-shape-rival-indices
    (model logits target shape limit)
  (let ((vocab (photon-model-metadata-get model 'vocab))
        (scores nil)
        (index 0)
        (walk logits))
    (while walk
      (when (and (/= index target)
                 vocab
                 (eq (photon-wordpiece-token-shape vocab index) shape))
        (setq scores (cons (cons index (car walk)) scores)))
      (setq walk (cdr walk))
      (setq index (1+ index)))
    (mapcar
     #'car
     (photon-take
      (sort scores (lambda (a b) (> (cdr a) (cdr b))))
      (max 1 limit)))))

(defun photon-start-long-same-shape-margin-distill-token-reader
    (model next-token context-size learning-rate)
  (let ((total 0)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (candidates 0)
        (updated 0)
        (bias-updated 0)
        (correct-before 0)
        (correct-after 0)
        (target-rank-before-sum 0.0)
        (target-rank-after-sum 0.0)
        (margin-before-sum 0.0)
        (margin-after-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-same-shape-margin-distill-top-k)))
        (rival-limit
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-same-shape-margin-distill-rivals)))
        (bias-weight
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-margin-distill-bias-weight))
        (adaptive-bias-max
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-margin-distill-adaptive-bias-max))
        (score-bonus-max
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-top1-search-score-bonus-max))
        (context-bonus-weight
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-top1-search-context-bonus-weight))
        (target-margin
         (photon-model-param-get
          model 'output-head-start-long-same-shape-margin-distill-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (shape (and vocab
                           (photon-wordpiece-token-shape vocab token)))
               (start-long-p (eq shape 'start-long-piece))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (photon-model-output-head-prototype-logits model state))
               (prototype-prediction
                (photon-argmax-index prototype-logits))
               (same-shape-rival
                (and start-long-p
                     (photon-best-same-wordpiece-shape-rival-index
                      model prototype-logits token shape)))
               (same-shape-rivals
                (and start-long-p
                     (photon-top-same-wordpiece-shape-rival-indices
                      model prototype-logits token shape rival-limit)))
               (rank-before
                (and start-long-p
                     (photon-logit-rank prototype-logits token)))
               (margin-before
                (if same-shape-rival
                    (photon-target-margin
                     prototype-logits token same-shape-rival)
                  0.0)))
          (setq total (1+ total))
          (when start-long-p
            (setq start-long-total (1+ start-long-total))
            (setq target-rank-before-sum
                  (+ target-rank-before-sum rank-before))
            (setq margin-before-sum
                  (+ margin-before-sum margin-before)))
          (when (= prototype-prediction token)
            (setq correct-before (1+ correct-before)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit same-shape-rival
                     (< margin-before target-margin))
            (setq candidates (1+ candidates))
            (let ((bias-step
                   (if (> adaptive-bias-max 0.0)
                       (min adaptive-bias-max
                            (/ (- target-margin margin-before) 2.0))
                     bias-weight))
                  (rivals (or same-shape-rivals
                              (list same-shape-rival))))
              (when (> bias-step 0.0)
                (photon-update-output-head-prototype-bias
                 model token bias-step)
                (while rivals
                  (photon-update-output-head-prototype-bias
                   model (car rivals) (- bias-step))
                  (setq rivals (cdr rivals)))
                (setq bias-updated (1+ bias-updated))))
            (when (> learning-rate 0.0)
              (photon-update-output-head
               model state same-shape-rival token learning-rate)
              (photon-update-output-head-projection
               model state (* 0.03 learning-rate))
              (setq updated (1+ updated))))
          (let* ((after-logits
                  (photon-model-output-head-prototype-logits model state))
                 (after-prediction (photon-argmax-index after-logits))
                 (rank-after
                  (and start-long-p
                       (photon-logit-rank after-logits token)))
                 (same-shape-rival-after
                  (and start-long-p
                       (photon-best-same-wordpiece-shape-rival-index
                        model after-logits token shape)))
                 (margin-after
                  (if same-shape-rival-after
                      (photon-target-margin
                       after-logits token same-shape-rival-after)
                    0.0)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after)))
            (when start-long-p
              (setq target-rank-after-sum
                    (+ target-rank-after-sum rank-after))
              (setq margin-after-sum
                    (+ margin-after-sum margin-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'rivals rival-limit)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'target-margin target-margin)
          (cons 'learning-rate learning-rate)
          (cons 'bias-weight bias-weight)
          (cons 'adaptive-bias-max adaptive-bias-max)
          (cons 'candidates candidates)
          (cons 'updated updated)
          (cons 'bias-updated bias-updated)
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'average-target-rank-before
                (if (= start-long-total 0)
                    0.0
                  (/ target-rank-before-sum start-long-total)))
          (cons 'average-target-rank-after
                (if (= start-long-total 0)
                    0.0
                  (/ target-rank-after-sum start-long-total)))
          (cons 'average-same-shape-margin-before
                (if (= start-long-total 0)
                    0.0
                  (/ margin-before-sum start-long-total)))
          (cons 'average-same-shape-margin-after
                (if (= start-long-total 0)
                    0.0
                  (/ margin-after-sum start-long-total)))
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-start-long-same-shape-margin-distill-token-reader-epochs
    (model token-reader-factory context-size learning-rate epochs)
  (let ((history nil)
        (epoch 0)
        (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-same-shape-margin-distill-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (setq history (cons (cons epoch last) history))
      (setq epoch (1+ epoch)))
    (list (cons 'history (nreverse history))
          (cons 'last last))))

(defun photon-start-long-same-shape-bias-candidates-token-reader
    (model next-token context-size)
  (let ((candidates nil)
        (start-long-total 0)
        (ngram-top-k-hit 0)
        (window nil)
        (filled 0)
        (token nil)
        (vocab (photon-model-metadata-get model 'vocab))
        (top-k
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-same-shape-margin-distill-top-k)))
        (rival-limit
         (truncate
          (photon-model-param-get
           model 'output-head-start-long-same-shape-margin-distill-rivals)))
        (bias-weight
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-margin-distill-bias-weight))
        (adaptive-bias-max
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-margin-distill-adaptive-bias-max))
        (score-bonus-max
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-top1-search-score-bonus-max))
        (context-bonus-weight
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-top1-search-context-bonus-weight))
        (target-margin
         (photon-model-param-get
          model 'output-head-start-long-same-shape-margin-distill-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (shape (and vocab
                           (photon-wordpiece-token-shape vocab token)))
               (start-long-p (eq shape 'start-long-piece))
               (ngram-logits
                (and start-long-p
                     (photon-context-anchor-ngram-logits model window)))
               (ngram-hit
                (and ngram-logits
                     (photon-top-k-hit-p
                      (photon-top-k-logits ngram-logits (max 1 top-k))
                      token)))
               (prototype-logits
                (and start-long-p
                     (photon-model-output-head-prototype-logits model state)))
               (same-shape-rival
                (and prototype-logits
                     (photon-best-same-wordpiece-shape-rival-index
                      model prototype-logits token shape)))
               (same-shape-rivals
                (and prototype-logits
                     (photon-top-same-wordpiece-shape-rival-indices
                      model prototype-logits token shape rival-limit)))
               (margin-before
                (if same-shape-rival
                    (photon-target-margin
                     prototype-logits token same-shape-rival)
                  0.0))
               (bias-step
                (if (> adaptive-bias-max 0.0)
                    (min adaptive-bias-max
                         (/ (- target-margin margin-before) 2.0))
                  bias-weight))
               (score-step
                (if (> score-bonus-max 0.0)
                    (min score-bonus-max
                         (max 0.0 (- target-margin margin-before)))
                  0.0)))
          (when start-long-p
            (setq start-long-total (1+ start-long-total)))
          (when ngram-hit
            (setq ngram-top-k-hit (1+ ngram-top-k-hit)))
          (when (and start-long-p ngram-hit same-shape-rival
                     (< margin-before target-margin)
                     (or (> bias-step 0.0)
                         (> score-step 0.0)
                         (> context-bonus-weight 0.0)))
            (setq candidates
                  (cons (list (cons 'target token)
                              (cons 'rivals
                                    (or same-shape-rivals
                                        (list same-shape-rival)))
                              (cons 'state state)
                              (cons 'bias-step bias-step)
                              (cons 'score-bonus-step score-step)
                              (cons 'margin-before margin-before))
                        candidates))))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (setq candidates (nreverse candidates))
    (list (cons 'candidates candidates)
          (cons 'candidate-count (length candidates))
          (cons 'start-long-total start-long-total)
          (cons 'ngram-top-k top-k)
          (cons 'rivals rival-limit)
          (cons 'score-bonus-max score-bonus-max)
          (cons 'context-bonus-weight context-bonus-weight)
          (cons 'ngram-top-k-hit ngram-top-k-hit)
          (cons 'start-long-ngram-hit-permil
                (if (= start-long-total 0)
                    0
                  (/ (* 1000 ngram-top-k-hit) start-long-total))))))

(defun photon-apply-start-long-same-shape-bias-candidate (model candidate)
  (let ((target (cdr (assq 'target candidate)))
        (rivals (cdr (assq 'rivals candidate)))
        (state (cdr (assq 'state candidate)))
        (bias-step (cdr (assq 'bias-step candidate)))
        (score-step (or (cdr (assq 'score-bonus-step candidate)) 0.0))
        (context-bonus-weight
         (photon-model-param-get
          model
          'output-head-start-long-same-shape-top1-search-context-bonus-weight)))
    (when (> bias-step 0.0)
      (photon-update-output-head-prototype-bias model target bias-step))
    (when (> score-step 0.0)
      (photon-update-output-head-prototype-score-bonus
       model target score-step))
    (when (and (> context-bonus-weight 0.0) state)
      (let ((old-weight
             (photon-model-param-get
              model 'output-head-prototype-context-bonus-weight)))
        (photon-model-param-set
         model 'output-head-prototype-context-bonus-weight
         (max old-weight context-bonus-weight)))
      (photon-update-output-head-context-bonus-prototype
       model state target))
    (while rivals
      (when (> bias-step 0.0)
        (photon-update-output-head-prototype-bias
         model (car rivals) (- bias-step)))
      (setq rivals (cdr rivals)))))

(defun photon-start-long-same-shape-top1-bias-search
    (model token-reader-factory context-size)
  (let* ((candidate-report
          (photon-start-long-same-shape-bias-candidates-token-reader
           model (funcall token-reader-factory) context-size))
         (candidates (cdr (assq 'candidates candidate-report)))
         (initial-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-output-head-prototype-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (global-tolerance
          (photon-model-param-get
           model
           'output-head-start-long-same-shape-top1-search-global-tolerance))
         (accuracy-floor
          (max 0 (- best-accuracy global-tolerance)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-start-long-top-3
          (cdr (assq 'top-3-accuracy-permil initial-start-long-score)))
         (best-start-long-top-5
          (cdr (assq 'top-5-accuracy-permil initial-start-long-score)))
         (best-start-long-rank
          (cdr (assq 'average-target-rank initial-start-long-score)))
         (best-prototype-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-prototype-score-bonus
          (copy-tree (photon-model-output-head-prototype-score-bonus model)))
         (best-context-bonus-prototypes
          (copy-tree
           (photon-model-output-head-context-bonus-prototypes model)))
         (best-context-bonus-weight
          (photon-model-param-get
           model 'output-head-prototype-context-bonus-weight))
         (tested 0)
         (accepted 0)
         (rejected 0))
    (while candidates
      (setq tested (1+ tested))
      (photon-model-set-output-head-prototype-bias
       model (copy-tree best-prototype-bias))
      (photon-model-set-output-head-prototype-score-bonus
       model (copy-tree best-prototype-score-bonus))
      (photon-model-set-output-head-context-bonus-prototypes
       model (copy-tree best-context-bonus-prototypes))
      (photon-model-param-set
       model 'output-head-prototype-context-bonus-weight
       best-context-bonus-weight)
      (photon-apply-start-long-same-shape-bias-candidate
       model (car candidates))
      (let* ((score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-output-head-prototype-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score)))
             (start-long-top-3
              (cdr (assq 'top-3-accuracy-permil start-long-score)))
             (start-long-top-5
              (cdr (assq 'top-5-accuracy-permil start-long-score)))
             (start-long-rank
              (cdr (assq 'average-target-rank start-long-score))))
        (if (or (> start-long-accuracy best-start-long-accuracy)
                (and (= start-long-accuracy best-start-long-accuracy)
                     (> accuracy best-accuracy))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-3 best-start-long-top-3))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (= start-long-top-3 best-start-long-top-3)
                     (>= accuracy accuracy-floor)
                     (> start-long-top-5 best-start-long-top-5))
                (and (= start-long-accuracy best-start-long-accuracy)
                     (>= accuracy accuracy-floor)
                     (= start-long-top-3 best-start-long-top-3)
                     (= start-long-top-5 best-start-long-top-5)
                     (< start-long-rank best-start-long-rank)))
            (progn
              (setq accepted (1+ accepted))
              (setq best-accuracy accuracy)
              (setq best-start-long-accuracy start-long-accuracy)
              (setq best-start-long-top-3 start-long-top-3)
              (setq best-start-long-top-5 start-long-top-5)
              (setq best-start-long-rank start-long-rank)
              (setq best-prototype-bias
                    (copy-tree
                     (photon-model-output-head-prototype-bias model)))
              (setq best-prototype-score-bonus
                    (copy-tree
                     (photon-model-output-head-prototype-score-bonus model)))
              (setq best-context-bonus-prototypes
                    (copy-tree
                     (photon-model-output-head-context-bonus-prototypes
                      model)))
              (setq best-context-bonus-weight
                    (photon-model-param-get
                     model 'output-head-prototype-context-bonus-weight)))
          (setq rejected (1+ rejected))))
      (setq candidates (cdr candidates)))
    (photon-model-set-output-head-prototype-bias model best-prototype-bias)
    (photon-model-set-output-head-prototype-score-bonus
     model best-prototype-score-bonus)
    (photon-model-set-output-head-context-bonus-prototypes
     model best-context-bonus-prototypes)
    (photon-model-param-set
     model 'output-head-prototype-context-bonus-weight
     best-context-bonus-weight)
    (list (cons 'candidate-count
                (cdr (assq 'candidate-count candidate-report)))
          (cons 'tested tested)
          (cons 'accepted accepted)
          (cons 'rejected rejected)
          (cons 'initial-output-head-prototype initial-score)
          (cons 'initial-start-long-prototype initial-start-long-score)
          (cons 'best-prototype-accuracy-permil best-accuracy)
          (cons 'accuracy-floor-permil accuracy-floor)
          (cons 'global-tolerance-permil global-tolerance)
          (cons 'best-start-long-prototype-accuracy-permil
                best-start-long-accuracy)
          (cons 'best-start-long-top-3-accuracy-permil
                best-start-long-top-3)
          (cons 'best-start-long-top-5-accuracy-permil
                best-start-long-top-5)
          (cons 'best-start-long-average-target-rank
                best-start-long-rank)
          (cons 'candidate-report candidate-report))))

(defun photon-start-long-same-shape-margin-distill-pocket
    (model token-reader-factory context-size learning-rate epochs)
  (let* ((initial-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-output-head-prototype-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-prototype-bias
          (copy-tree (photon-model-output-head-prototype-bias model)))
         (best-prototype-score-bonus
          (copy-tree (photon-model-output-head-prototype-score-bonus model)))
         (best-context-bonus-prototypes
          (copy-tree
           (photon-model-output-head-context-bonus-prototypes model)))
         (best-context-bonus-weight
          (photon-model-param-get
           model 'output-head-prototype-context-bonus-weight))
         (best-head (copy-tree (photon-model-output-head model)))
         (best-bias (copy-tree (photon-model-output-head-bias model)))
         (best-projection
          (copy-tree (photon-model-output-head-projection model)))
         (history nil)
         (epoch 0)
         (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-start-long-same-shape-margin-distill-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate))
      (let* ((score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-output-head-prototype-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score))))
        (when (or (> start-long-accuracy best-start-long-accuracy)
                  (and (= start-long-accuracy best-start-long-accuracy)
                       (>= accuracy best-accuracy)))
          (setq best-accuracy accuracy)
          (setq best-start-long-accuracy start-long-accuracy)
          (setq best-prototype-bias
                (copy-tree (photon-model-output-head-prototype-bias model)))
          (setq best-prototype-score-bonus
                (copy-tree
                 (photon-model-output-head-prototype-score-bonus model)))
          (setq best-context-bonus-prototypes
                (copy-tree
                 (photon-model-output-head-context-bonus-prototypes model)))
          (setq best-context-bonus-weight
                (photon-model-param-get
                 model 'output-head-prototype-context-bonus-weight))
          (setq best-head (copy-tree (photon-model-output-head model)))
          (setq best-bias (copy-tree (photon-model-output-head-bias model)))
          (setq best-projection
                (copy-tree (photon-model-output-head-projection model))))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list
                                   (cons 'eval-output-head-prototype score)
                                   (cons 'eval-start-long-prototype
                                         start-long-score)
                                   (cons 'best-prototype-accuracy-permil
                                         best-accuracy)
                                   (cons
                                    'best-start-long-prototype-accuracy-permil
                                    best-start-long-accuracy))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head-prototype-bias model best-prototype-bias)
    (photon-model-set-output-head-prototype-score-bonus
     model best-prototype-score-bonus)
    (photon-model-set-output-head-context-bonus-prototypes
     model best-context-bonus-prototypes)
    (photon-model-param-set
     model 'output-head-prototype-context-bonus-weight
     best-context-bonus-weight)
    (photon-model-set-output-head model best-head)
    (photon-model-set-output-head-bias model best-bias)
    (photon-model-set-output-head-projection model best-projection)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'initial-output-head-prototype initial-score)
          (cons 'initial-start-long-prototype initial-start-long-score)
          (cons 'best-prototype-accuracy-permil best-accuracy)
          (cons 'best-start-long-prototype-accuracy-permil
                best-start-long-accuracy))))

(defun photon-distill-full-readout-to-output-head-token-reader
    (model next-token context-size learning-rate &optional teacher-key)
  (let ((total 0)
        (teacher-correct 0)
        (updated 0)
        (scaled-updated 0)
        (update-scale-sum 0.0)
        (correct-before 0)
        (correct-after 0)
        (margin-sum 0.0)
        (window nil)
        (filled 0)
        (token nil)
        (teacher-key (or teacher-key 'full))
        (margin-target (photon-model-param-get model 'next-token-margin)))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components (photon-model-readout-components model state window))
               (teacher-logits (or (cdr (assq teacher-key components))
                                   (cdr (assq 'full components))))
               (teacher-prediction (photon-argmax-index teacher-logits))
               (linear-logits (photon-model-output-head-logits model state))
               (linear-prediction (photon-argmax-index linear-logits))
               (linear-rival
                (photon-best-non-target-index linear-logits token))
               (linear-margin
                (photon-target-margin linear-logits token linear-rival)))
          (setq total (1+ total))
          (setq margin-sum (+ margin-sum linear-margin))
          (when (= linear-prediction token)
            (setq correct-before (1+ correct-before)))
          (when (= teacher-prediction token)
            (setq teacher-correct (1+ teacher-correct))
            (when (or (/= linear-prediction token)
                      (< linear-margin margin-target))
              (let ((update-scale
                     (photon-full-readout-distill-update-scale model token)))
                (photon-update-output-head
                 model state linear-prediction token
                 (* learning-rate update-scale))
                (photon-update-output-head-projection
                 model state (* 0.03 learning-rate update-scale))
                (when (> update-scale 1.0)
                  (setq scaled-updated (1+ scaled-updated)))
                (setq update-scale-sum
                      (+ update-scale-sum update-scale)))
              (setq updated (1+ updated))))
          (let* ((after-logits (photon-model-output-head-logits model state))
                 (after-prediction (photon-argmax-index after-logits)))
            (when (= after-prediction token)
              (setq correct-after (1+ correct-after))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (list (cons 'total total)
          (cons 'teacher-key teacher-key)
          (cons 'teacher-correct teacher-correct)
          (cons 'updated updated)
          (cons 'scaled-updated scaled-updated)
          (cons 'average-update-scale
                (if (= updated 0) 0.0 (/ update-scale-sum updated)))
          (cons 'correct-before correct-before)
          (cons 'correct-after correct-after)
          (cons 'accuracy-before-permil
                (if (= total 0) 0 (/ (* 1000 correct-before) total)))
          (cons 'accuracy-after-permil
                (if (= total 0) 0 (/ (* 1000 correct-after) total)))
          (cons 'teacher-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 teacher-correct) total)))
          (cons 'average-margin-before
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-distill-full-readout-to-output-head-token-reader-pocket
    (model token-reader-factory context-size learning-rate epochs
           &optional teacher-key)
  (let* ((initial-score
          (photon-evaluate-linear-output-head-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-prototype-score
          (photon-evaluate-output-head-prototype-token-reader
           model (funcall token-reader-factory) context-size))
         (initial-start-long-score
          (photon-evaluate-linear-output-head-shape-token-reader
           model (funcall token-reader-factory) context-size
           'start-long-piece))
         (start-long-selection-weight
          (photon-model-param-get
           model
           'output-head-full-readout-distill-selection-start-long-weight))
         (best-accuracy (cdr (assq 'accuracy-permil initial-score)))
         (best-prototype-accuracy
          (cdr (assq 'accuracy-permil initial-prototype-score)))
         (best-start-long-accuracy
          (cdr (assq 'accuracy-permil initial-start-long-score)))
         (best-selection-score
          (+ best-accuracy
             best-prototype-accuracy
             (* start-long-selection-weight best-start-long-accuracy)))
         (best-head (copy-tree (photon-model-output-head model)))
         (best-bias (copy-tree (photon-model-output-head-bias model)))
         (best-projection (copy-tree (photon-model-output-head-projection model)))
         (history nil)
         (epoch 0)
         (last nil))
    (while (< epoch epochs)
      (setq last
            (photon-distill-full-readout-to-output-head-token-reader
             model
             (funcall token-reader-factory)
             context-size
             learning-rate
             teacher-key))
      (let* ((score
              (photon-evaluate-linear-output-head-token-reader
               model (funcall token-reader-factory) context-size))
             (prototype-score
              (photon-evaluate-output-head-prototype-token-reader
               model (funcall token-reader-factory) context-size))
             (start-long-score
              (photon-evaluate-linear-output-head-shape-token-reader
               model (funcall token-reader-factory) context-size
               'start-long-piece))
             (accuracy (cdr (assq 'accuracy-permil score)))
             (prototype-accuracy
              (cdr (assq 'accuracy-permil prototype-score)))
             (start-long-accuracy
              (cdr (assq 'accuracy-permil start-long-score)))
             (selection-score
              (+ accuracy
                 prototype-accuracy
                 (* start-long-selection-weight start-long-accuracy))))
        (when (> selection-score best-selection-score)
          (setq best-accuracy accuracy)
          (setq best-prototype-accuracy prototype-accuracy)
          (setq best-start-long-accuracy start-long-accuracy)
          (setq best-selection-score selection-score)
          (setq best-head (copy-tree (photon-model-output-head model)))
          (setq best-bias (copy-tree (photon-model-output-head-bias model)))
          (setq best-projection
                (copy-tree (photon-model-output-head-projection model))))
        (setq history
              (cons (cons epoch
                          (append last
                                  (list (cons 'eval-linear score)
                                        (cons 'eval-output-head-prototype
                                              prototype-score)
                                        (cons 'eval-start-long-linear
                                              start-long-score)
                                        (cons 'best-linear-accuracy-permil
                                              best-accuracy)
                                        (cons 'best-prototype-accuracy-permil
                                              best-prototype-accuracy)
                                        (cons 'best-start-long-linear-accuracy-permil
                                              best-start-long-accuracy)
                                        (cons 'best-selection-score
                                              best-selection-score))))
                    history)))
      (setq epoch (1+ epoch)))
    (photon-model-set-output-head model best-head)
    (photon-model-set-output-head-bias model best-bias)
    (photon-model-set-output-head-projection model best-projection)
    (list (cons 'history (nreverse history))
          (cons 'last last)
          (cons 'initial-linear initial-score)
          (cons 'initial-output-head-prototype initial-prototype-score)
          (cons 'initial-start-long-linear initial-start-long-score)
          (cons 'best-linear-accuracy-permil best-accuracy)
          (cons 'best-prototype-accuracy-permil best-prototype-accuracy)
          (cons 'best-start-long-linear-accuracy-permil
                best-start-long-accuracy)
          (cons 'best-selection-score best-selection-score))))

(defun photon-update-embedding-table (model context prediction target learning-rate)
  (let* ((table (photon-model-embedding-table model))
         (head (photon-model-output-head model))
         (target-row (nth target head))
         (prediction-row (nth prediction head))
         (signal (photon-vector-scale
                  (/ learning-rate (max 1 (length context)))
                  (photon-vector-add target-row
                                     (photon-vector-scale -1.0 prediction-row)))))
    (while context
      (let* ((token (car context))
             (embedding (nth token table)))
        (setq table
              (photon-nth-set table token
                              (photon-vector-squash
                               (photon-vector-add embedding signal)))))
      (setq context (cdr context)))
    (photon-model-set-embedding-table model table)
    (photon-model-clear-cache model)))

(defun photon-update-embedding-table-from-signal
    (model context signal learning-rate)
  (let* ((table (photon-model-embedding-table model))
         (delta
          (photon-vector-scale
           (/ learning-rate (max 1 (length context)))
           signal)))
    (while context
      (let* ((token (car context))
             (embedding (nth token table)))
        (setq table
              (photon-nth-set table token
                              (photon-vector-squash
                               (photon-vector-add embedding delta)))))
      (setq context (cdr context)))
    (photon-model-set-embedding-table model table)
    (photon-model-clear-cache model)))

(defun photon-update-architecture-params (model learning-rate)
  (let ((step (* 0.01 learning-rate)))
    (photon-model-param-add model 'chunk-scale step 0.50 1.50)
    (photon-model-param-add model 'context-current step 0.20 1.20)
    (photon-model-param-add model 'context-previous (* 0.5 step) 0.10 1.00)
    (photon-model-param-add model 'converter-scale (* 0.5 step) 0.50 1.50)
    (photon-model-param-add model 'decoder-prev (* 0.5 step) 0.05 0.90)
    (photon-model-param-add model 'decoder-token-mix (* 0.5 step) 0.50 2.00)))

(defun photon-update-layer-gain (model key signal learning-rate)
  (let* ((gain (photon-model-gain-get model key))
         (delta (photon-vector-scale learning-rate signal)))
    (photon-model-gain-set
     model key
     (photon-vector-squash (photon-vector-add gain delta)))))

(defun photon-update-layer-gains (model prediction target learning-rate)
  (let* ((head (photon-model-output-head model))
         (target-row (nth target head))
         (prediction-row (nth prediction head))
         (signal (photon-vector-add target-row
                                    (photon-vector-scale -1.0 prediction-row)))
         (rate (* 0.05 learning-rate)))
    (photon-update-layer-gain model 'chunk-gain signal rate)
    (photon-update-layer-gain model 'context-current-gain signal rate)
    (photon-update-layer-gain model 'context-previous-gain signal (* 0.5 rate))
    (photon-update-layer-gain model 'converter-gain signal (* 0.5 rate))
    (photon-update-layer-gain model 'decoder-prev-gain signal (* 0.5 rate))))

(defun photon-update-projection-matrix (model key signal learning-rate)
  (let ((projection (photon-model-projection-get model key))
        (delta (photon-vector-scale learning-rate signal)))
    (if (and (consp projection)
             (eq (cdr (assq 'kind projection)) 'low-rank))
        (let ((left (cdr (assq 'left projection)))
              (right (cdr (assq 'right projection)))
              (diag (cdr (assq 'diag projection))))
          (setcdr (assq 'left projection)
                  (photon-vector-squash (photon-vector-add left delta)))
          (setcdr (assq 'right projection)
                  (photon-vector-squash
                   (photon-vector-add right (photon-vector-scale 0.5 delta))))
          (setcdr (assq 'diag projection)
                  (photon-vector-squash
                   (photon-vector-add diag (photon-vector-scale 0.1 delta))))
          (photon-model-projection-set model key projection))
      (let ((matrix projection)
            (updated nil))
        (while matrix
          (setq updated
                (cons (photon-vector-squash
                       (photon-vector-add (car matrix) delta))
                      updated))
          (setq matrix (cdr matrix)))
        (photon-model-projection-set model key (nreverse updated))))))

(defun photon-update-layer-projections (model prediction target learning-rate)
  (let* ((head (photon-model-output-head model))
         (target-row (nth target head))
         (prediction-row (nth prediction head))
         (signal (photon-vector-add target-row
                                    (photon-vector-scale -1.0 prediction-row)))
         (rate (* 0.01 learning-rate)))
    (photon-update-projection-matrix model 'chunk-proj signal rate)
    (photon-update-projection-matrix model 'context-proj signal rate)
    (photon-update-projection-matrix model 'converter-proj signal (* 0.5 rate))
    (photon-update-projection-matrix model 'decoder-proj signal (* 0.5 rate))))

(defun photon-update-long-piece-state-path
    (model context prediction target learning-rate)
  (let* ((target-embedding (photon-model-token-embedding model target))
         (prediction-embedding
          (photon-model-token-embedding model prediction))
         (signal
          (photon-vector-add
           target-embedding
           (photon-vector-scale -1.0 prediction-embedding))))
    (photon-update-embedding-table-from-signal
     model context signal (* 0.50 learning-rate))
    (photon-update-layer-gain
     model 'context-current-gain signal (* 0.08 learning-rate))
    (photon-update-layer-gain
     model 'decoder-prev-gain signal (* 0.04 learning-rate))
    (photon-update-projection-matrix
     model 'context-proj signal (* 0.03 learning-rate))
    (photon-update-projection-matrix
     model 'decoder-proj signal (* 0.015 learning-rate))
    (photon-model-clear-cache model)))

(defun photon-gradient-accumulate-row (grad row-index delta)
  (let ((cell (assq row-index grad)))
    (if cell
        (setcdr cell (photon-vector-add (cdr cell) delta))
      (setq grad (cons (cons row-index delta) grad)))
    grad))

(defun photon-gradient-accumulate-miss (grad state prediction target)
  (setq grad (photon-gradient-accumulate-row grad target state))
  (photon-gradient-accumulate-row grad prediction
                                  (photon-vector-scale -1.0 state)))

(defun photon-apply-output-gradient (model grad learning-rate batch-count)
  (let ((head (photon-model-output-head model))
        (scale (if (> batch-count 0)
                   (/ (float learning-rate) batch-count)
                 learning-rate)))
    (while grad
      (let* ((row-index (car (car grad)))
             (row (nth row-index head))
             (delta (photon-vector-scale scale (cdr (car grad)))))
        (setq head (photon-nth-set head row-index
                                   (photon-vector-add row delta))))
      (setq grad (cdr grad)))
    (photon-model-set-output-head model head)))

(defun photon-update-embedding-token (model token signal)
  (let* ((table (photon-model-embedding-table model))
         (embedding (nth token table)))
    (setq table
          (photon-nth-set table token
                          (photon-vector-squash
                           (photon-vector-add embedding signal))))
    (photon-model-set-embedding-table model table)))

(defun photon-update-reconstruction-from-report (model context report learning-rate)
  (let ((states (cdr (assq 'states report)))
        (table (photon-model-embedding-table model))
        (head (photon-model-reconstruction-head model))
        (grad nil)
        (signal-sum nil)
        (misses 0)
        (changed nil)
        (rate (* learning-rate (photon-model-param-get model 'reconstruction-weight))))
    (while (and context states)
      (let* ((target (car context))
             (state (car states))
             (prediction (photon-model-reconstruction-token model state)))
        (when (/= prediction target)
          (setq changed t)
          (setq misses (1+ misses))
          (setq grad
                (photon-gradient-accumulate-miss grad state prediction target))
          (setq signal-sum
                (if signal-sum
                    (photon-vector-add signal-sum state)
                  state))
          (let* ((target-row (nth target head))
                 (prediction-row (nth prediction head))
                 (signal (photon-vector-scale
                          (* 0.50 rate)
                          (photon-vector-add
                           target-row
                           (photon-vector-scale -1.0 prediction-row)))))
            (setq table
                  (photon-nth-set
                   table target
                   (photon-vector-squash
                   (photon-vector-add (nth target table) signal)))))))
      (setq context (cdr context))
      (setq states (cdr states)))
    (when changed
      (let ((scale (/ (float rate) (max 1 misses))))
        (while grad
          (let* ((row-index (car (car grad)))
                 (row (nth row-index head))
                 (delta (photon-vector-scale scale (cdr (car grad)))))
            (setq head (photon-nth-set head row-index
                                       (photon-vector-add row delta))))
          (setq grad (cdr grad))))
      (photon-model-set-reconstruction-head model head)
      (photon-model-set-embedding-table model table)
      (photon-model-param-add model 'decoder-token-mix
                              (* rate (/ (float misses)
                                         (max 1
                                              (cdr (assq 'reconstruction-total
                                                         report)))))
                              0.50 2.00)
      (let ((signal (photon-vector-scale (/ 1.0 (max 1 misses)) signal-sum)))
        (photon-update-layer-gain model 'converter-gain signal (* 0.20 rate))
        (photon-update-layer-gain model 'decoder-prev-gain signal (* 0.20 rate))
        (photon-update-projection-matrix model 'converter-proj signal (* 0.05 rate))
        (photon-update-projection-matrix model 'decoder-proj signal (* 0.05 rate))))))

(defun photon-update-next-context-from-report (model report learning-rate)
  (let ((chunk-states (cdr (assq 'chunk-states report)))
        (rate (* learning-rate (photon-model-param-get model 'next-context-weight))))
    (while (cdr chunk-states)
      (let ((signal (photon-vector-scale
                     rate
                     (photon-vector-add
                      (car (cdr chunk-states))
                      (photon-vector-scale -1.0 (car chunk-states))))))
        (photon-update-layer-gain model 'chunk-gain signal (* 0.25 rate))
        (photon-update-layer-gain model 'context-current-gain signal (* 0.25 rate))
        (photon-update-projection-matrix model 'chunk-proj signal (* 0.10 rate))
        (photon-update-projection-matrix model 'context-proj signal (* 0.10 rate)))
      (setq chunk-states (cdr chunk-states)))))

(defun photon-update-context-anchor-weight (model context target learning-rate)
  (when (and context (photon-model-context-anchor-enabled-p model))
    (let* ((anchor-logits (photon-model-context-anchor-logits model context))
           (anchor-prediction (photon-argmax-index anchor-logits))
           (anchor-top-3 (photon-top-k-logits anchor-logits 3)))
      (when (or (= anchor-prediction target)
                (photon-top-k-hit-p anchor-top-3 target)
                (photon-member-equal-p target context))
        (photon-model-param-add model 'context-anchor-weight
                                (* 5.0 learning-rate)
                                0.0
                                (photon-model-param-get
                                 model 'context-anchor-weight-max))))))

(defun photon-update-output-head-auxiliary (model state target learning-rate)
  (let* ((logits (photon-model-token-logits model state))
         (prediction (photon-argmax-index logits))
         (rival (photon-best-non-target-index logits target))
         (margin (photon-target-margin logits target rival)))
    (when (or (/= prediction target)
              (< margin (photon-model-param-get model 'next-token-margin)))
      (photon-update-output-head
       model state prediction target
       (* learning-rate
          (photon-model-param-get model 'output-head-aux-update-weight))))))

(defun photon-update-output-head-anchor-distill (model context target
                                                       learning-rate)
  (when (and context
             (> (photon-model-param-get
                 model 'output-head-anchor-distill-weight)
                0.0))
    (let* ((anchor-logits (photon-model-context-anchor-logits model context))
           (anchor-top-3 (photon-top-k-logits anchor-logits 3)))
      (when (or (photon-top-k-hit-p anchor-top-3 target)
                (photon-member-equal-p target context))
        (let* ((anchor-state (photon-context-anchor-state model context))
               (head-logits (photon-model-output-head-logits
                             model anchor-state))
               (prediction (photon-argmax-index head-logits)))
          (photon-update-output-head
           model anchor-state prediction target
           (* learning-rate
              (photon-model-param-get
               model 'output-head-anchor-distill-weight))))))))

(defun photon-update-ngram-readout-distill (model context state target
                                                  learning-rate)
  (when (and context
             (photon-context-anchor-ngram-enabled-p model)
             (or (> (photon-model-param-get
                     model 'output-head-ngram-distill-weight)
                    0.0)
                 (> (photon-model-param-get
                     model 'prototype-ngram-distill-weight)
                    0.0)))
    (let* ((ngram-logits (photon-context-anchor-ngram-logits model context))
           (ngram-top-3 (photon-top-k-logits ngram-logits 3)))
      (when (photon-top-k-hit-p ngram-top-3 target)
        (let* ((anchor-state (photon-context-anchor-state model context))
               (distill-state
                (photon-vector-squash
                 (photon-vector-add
                  (photon-vector-scale 0.70 state)
                  (photon-vector-scale 0.30 anchor-state))))
               (head-logits
                (photon-model-output-head-logits model distill-state))
               (head-prediction (photon-argmax-index head-logits))
               (head-rival
                (photon-best-non-target-index head-logits target))
               (head-margin
                (photon-target-margin head-logits target head-rival))
               (prototype-logits
                (photon-model-target-prototype-logits model distill-state))
               (prototype-prediction
                (photon-argmax-index prototype-logits))
               (prototype-rival
                (photon-best-non-target-index prototype-logits target))
               (prototype-margin
                (photon-target-margin
                 prototype-logits target prototype-rival)))
          (when (or (/= head-prediction target)
                    (< head-margin
                       (photon-model-param-get model 'next-token-margin)))
            (photon-update-output-head
             model distill-state head-prediction target
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-ngram-distill-weight))))
          (when (or (/= prototype-prediction target)
                    (< prototype-margin
                       (photon-model-param-get model 'next-token-margin)))
            (photon-update-target-prototype-signal
             model distill-state target
             (* learning-rate
                (photon-model-param-get
                 model 'prototype-ngram-distill-weight)))))))))

(defun photon-wordpiece-long-teacher-target-p (model target)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (and (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
         vocab
         (eq (photon-wordpiece-token-kind vocab target) 'long-piece))))

(defun photon-wordpiece-continuation-teacher-target-p (model target)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (and (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
         vocab
         (photon-wordpiece-token-continuation-p vocab target))))

(defun photon-wordpiece-start-long-teacher-target-p (model target)
  (let ((vocab (photon-model-metadata-get model 'vocab)))
    (and (eq (photon-model-metadata-get model 'tokenizer) 'wordpiece)
         vocab
         (eq (photon-wordpiece-token-shape vocab target)
             'start-long-piece))))

(defun photon-wordpiece-teacher-state (model context state target)
  (let* ((anchor-state (photon-context-anchor-state model context))
         (vocab (photon-model-metadata-get model 'vocab))
         (continuation-p
          (and vocab (photon-wordpiece-token-continuation-p vocab target)))
         (previous-token (car (last context)))
         (previous-embedding
          (and continuation-p
               previous-token
               (photon-model-token-embedding model previous-token))))
    (if previous-embedding
        (photon-vector-squash
         (photon-vector-add
          (photon-vector-add
           (photon-vector-scale 0.50 state)
           (photon-vector-scale 0.30 anchor-state))
          (photon-vector-scale 0.20 previous-embedding)))
      (photon-vector-squash
       (photon-vector-add
        (photon-vector-scale 0.60 state)
        (photon-vector-scale 0.40 anchor-state))))))

(defun photon-wordpiece-start-long-teacher-state (model context state target)
  (let ((result (photon-wordpiece-teacher-state model context state target))
        (token-mix
         (photon-model-param-get
          model 'wordpiece-start-long-teacher-token-mix))
        (context-mix
         (photon-model-param-get
          model 'wordpiece-start-long-teacher-context-mix))
        (anchor-mix
         (photon-model-param-get
          model 'wordpiece-start-long-teacher-anchor-mix)))
    (when (> token-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              token-mix
              (photon-model-token-embedding model target)))))
    (when (> context-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              context-mix
              (photon-model-context-average-embedding model context)))))
    (when (> anchor-mix 0.0)
      (setq result
            (photon-vector-add
             result
             (photon-vector-scale
              anchor-mix
              (photon-context-anchor-state model context)))))
    (photon-vector-squash result)))

(defun photon-update-wordpiece-long-teacher-state-path
    (model context target report learning-rate)
  (let ((weight
         (photon-model-param-get
          model 'wordpiece-long-teacher-state-path-weight)))
    (when (and (> weight 0.0)
               context
               (photon-context-anchor-ngram-enabled-p model)
               (photon-wordpiece-long-teacher-target-p model target))
      (let* ((state (cdr (assq 'state report)))
             (prediction (cdr (assq 'prediction report)))
             (rival (cdr (assq 'rival report)))
             (target-margin (cdr (assq 'target-margin report)))
             (ngram-logits (photon-context-anchor-ngram-logits model context))
             (ngram-top-3 (photon-top-k-logits ngram-logits 3)))
        (when (photon-top-k-hit-p ngram-top-3 target)
          (let* ((teacher-state
                  (photon-wordpiece-teacher-state
                   model context state target))
                 (rate (* learning-rate weight))
                 (head-logits
                  (photon-model-output-head-logits model teacher-state))
                 (teacher-prediction (photon-argmax-index head-logits)))
            (photon-update-target-prototype-signal
             model teacher-state target (* 0.75 rate))
            (photon-update-output-head-prototype
             model teacher-state target)
            (when (> rate 0.0)
              (photon-update-output-head
               model teacher-state teacher-prediction target rate))
            (when (and (/= prediction target)
                       (> rate 0.0))
              (photon-update-long-piece-state-path
               model context prediction target (* 0.50 rate)))
            (when (and (= prediction target)
                       rival
                       (< target-margin
                          (photon-model-param-get model 'next-token-margin))
                       (> rate 0.0))
              (photon-update-output-head
               model teacher-state rival target (* 0.50 rate)))
            t))))))

(defun photon-update-wordpiece-continuation-teacher-state-path
    (model context target report learning-rate)
  (let ((weight
         (photon-model-param-get
          model 'wordpiece-continuation-teacher-state-path-weight)))
    (when (and (> weight 0.0)
               context
               (photon-context-anchor-ngram-enabled-p model)
               (photon-wordpiece-continuation-teacher-target-p model target))
      (let* ((state (cdr (assq 'state report)))
             (prediction (cdr (assq 'prediction report)))
             (ngram-logits (photon-context-anchor-ngram-logits model context))
             (ngram-top-5 (photon-top-k-logits ngram-logits 5)))
        (when (photon-top-k-hit-p ngram-top-5 target)
          (let* ((teacher-state
                  (photon-wordpiece-teacher-state
                   model context state target))
                 (rate (* learning-rate weight))
                 (head-logits
                  (photon-model-output-head-logits model teacher-state))
                 (teacher-prediction (photon-argmax-index head-logits)))
            (photon-update-target-prototype-signal
             model teacher-state target rate)
            (photon-update-output-head-prototype
             model teacher-state target)
            (photon-update-output-head
             model teacher-state teacher-prediction target (* 0.75 rate))
            (when (/= prediction target)
              (photon-update-long-piece-state-path
               model context prediction target (* 0.75 rate)))
            t))))))

(defun photon-update-wordpiece-start-long-teacher-state-path
    (model context target report learning-rate)
  (let ((weight
         (photon-model-param-get
          model 'wordpiece-start-long-teacher-state-path-weight)))
    (when (and (> weight 0.0)
               context
               (photon-context-anchor-ngram-enabled-p model)
               (photon-wordpiece-start-long-teacher-target-p model target))
      (let* ((state (cdr (assq 'state report)))
             (prediction (cdr (assq 'prediction report)))
             (ngram-logits (photon-context-anchor-ngram-logits model context))
             (ngram-top-5 (photon-top-k-logits ngram-logits 5)))
        (when (photon-top-k-hit-p ngram-top-5 target)
          (let* ((teacher-state
                  (photon-wordpiece-start-long-teacher-state
                   model context state target))
                 (rate (* learning-rate weight))
                 (head-logits
                  (photon-model-output-head-logits model teacher-state))
                 (teacher-prediction (photon-argmax-index head-logits)))
            (photon-update-target-prototype-signal
             model teacher-state target (* 0.75 rate))
            (photon-update-output-head-prototype
             model teacher-state target)
            (photon-update-output-head
             model teacher-state teacher-prediction target (* 0.75 rate))
            (when (/= prediction target)
              (photon-update-long-piece-state-path
               model context prediction target (* 0.50 rate)))
            t))))))

(defun photon-update-next-token-from-report (model context target report learning-rate)
  (let ((prediction (cdr (assq 'prediction report)))
        (rival (cdr (assq 'rival report)))
        (target-margin (cdr (assq 'target-margin report)))
        (state (cdr (assq 'state report)))
        (next-token-rate (* learning-rate
                            (photon-model-param-get model 'next-token-weight))))
    (when (> next-token-rate 0.0)
      (photon-update-target-prototype model state target next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-target-state-cluster model state target))
    (when (> next-token-rate 0.0)
      (photon-update-char-class-bias model context target))
    (when (> next-token-rate 0.0)
      (photon-update-context-anchor-ngram model context target))
    (when (> next-token-rate 0.0)
      (photon-update-start-long-boundary-readout model context target))
    (when (> next-token-rate 0.0)
      (photon-update-context-anchor-ngram-contrastive
       model context target rival target-margin next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-context-anchor-weight
       model context target next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-output-head-auxiliary
       model state target next-token-rate))
    (when (> next-token-rate 0.0)
      (let* ((classifier-logits
              (photon-model-start-long-classifier-logits model state))
             (classifier-prediction
              (photon-argmax-index classifier-logits)))
        (photon-update-start-long-classifier
         model state classifier-prediction target next-token-rate)))
    (when (/= prediction target)
      (photon-update-output-head model state prediction target next-token-rate)
      (photon-update-embedding-table model context prediction target
                                     (* 0.25 next-token-rate))
      (photon-update-architecture-params model next-token-rate)
      (photon-update-layer-gains model prediction target next-token-rate)
      (photon-update-layer-projections model prediction target next-token-rate))
    (when (and (= prediction target)
               (< target-margin (photon-model-param-get model
                                                        'next-token-margin)))
      (photon-update-output-head
       model state rival target
       (* 0.50 next-token-rate
          (/ (- (photon-model-param-get model 'next-token-margin)
                target-margin)
             (max 0.001 (photon-model-param-get model 'next-token-margin))))))
    (when (> next-token-rate 0.0)
      (photon-update-output-head-anchor-distill
       model context target next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-ngram-readout-distill
       model context state target next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-wordpiece-long-teacher-state-path
       model context target report next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-wordpiece-continuation-teacher-state-path
       model context target report next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-update-wordpiece-start-long-teacher-state-path
       model context target report next-token-rate))
    (when (> next-token-rate 0.0)
      (photon-model-remember-next-token model context target))
    (if (= prediction target) 1 0)))

(defun photon-train-one (model context target learning-rate)
  (photon-train-one-from-report
   model context target learning-rate
   (photon-loss-report model context target)))

(defun photon-predict-one (model context)
  (let* ((cache-key (photon-prediction-cache-key context))
         (cached (photon-model-cache-get model cache-key)))
    (if cached
        cached
      (photon-model-cache-put
       model cache-key
       (let* ((states (photon-forward-model model context))
              (state (photon-prediction-state states context))
              (remembered (photon-model-next-token-memory-get model context))
              (prediction (or remembered
                              (photon-argmax-index
                               (photon-model-readout-logits
                                model state context)))))
         (list (cons 'state state)
               (cons 'prediction prediction)))))))

(defun photon-model-state-token (model state)
  (photon-argmax-index (photon-model-token-logits model state)))

(defun photon-model-reconstruction-token (model state)
  (photon-argmax-index (photon-model-reconstruction-logits model state)))

(defun photon-next-token-loss (prediction target)
  (if (= prediction target) 0 1))

(defun photon-reconstruction-loss (model context states)
  (let ((loss 0)
        (total 0))
    (while (and context states)
      (unless (= (photon-model-reconstruction-token model (car states))
                 (car context))
        (setq loss (1+ loss)))
      (setq total (1+ total))
      (setq context (cdr context))
      (setq states (cdr states)))
    (list (cons 'loss loss)
          (cons 'total total))))

(defun photon-context-state-distance (left right)
  (let ((left-items (photon-vector-to-list left))
        (right-items (photon-vector-to-list right))
        (sum 0.0)
        (total 0))
    (while (and left-items right-items)
      (setq sum (+ sum (abs (- (car left-items) (car right-items)))))
      (setq total (1+ total))
      (setq left-items (cdr left-items))
      (setq right-items (cdr right-items)))
    (if (= total 0)
        0.0
      (/ sum total))))

(defun photon-next-context-loss (model context)
  (let* ((chunk-states (photon-context-chunk-states model context))
         (loss 0)
         (total 0))
    (while (cdr chunk-states)
      (setq loss (+ loss
                    (truncate
                     (* 1000.0
                        (photon-context-state-distance
                         (car chunk-states) (car (cdr chunk-states)))))))
      (setq total (1+ total))
      (setq chunk-states (cdr chunk-states)))
    (list (cons 'loss loss)
          (cons 'total total))))

(defun photon-context-chunk-states (model context)
  (let* ((config (photon-model-config model))
         (chunk-size (photon-config-get config 'chunk-size))
         (chunks (photon-chunk-tokens context chunk-size))
         (chunk-states nil)
         (walk chunks))
    (while walk
      (setq chunk-states
            (cons (photon-model-chunker model (car walk)) chunk-states))
      (setq walk (cdr walk)))
    (nreverse chunk-states)))

(defun photon-loss-report (model context target)
  (let* ((states (photon-forward-model model context))
         (state (photon-prediction-state states context))
         (logits (photon-model-readout-logits model state context))
         (remembered (photon-model-next-token-memory-get model context))
         (memory-hit (photon-model-next-token-memory-hit-p model context))
         (fallback-prediction (photon-argmax-index logits))
         (top-3 (photon-top-k-logits logits 3))
         (top-5 (photon-top-k-logits logits 5))
         (prediction (or remembered fallback-prediction))
         (rival (photon-best-non-target-index logits target))
         (target-margin (photon-target-margin logits target rival))
         (reconstruction (photon-reconstruction-loss model context states))
         (next-context (photon-next-context-loss model context)))
    (list (cons 'prediction prediction)
          (cons 'fallback-prediction fallback-prediction)
          (cons 'fallback-top-3-hit (photon-top-k-hit-p top-3 target))
          (cons 'fallback-top-5-hit (photon-top-k-hit-p top-5 target))
          (cons 'memory-hit memory-hit)
          (cons 'rival rival)
          (cons 'target-margin target-margin)
          (cons 'state state)
          (cons 'states states)
          (cons 'chunk-states (photon-context-chunk-states model context))
          (cons 'next-token-loss (photon-next-token-loss prediction target))
          (cons 'reconstruction-loss (cdr (assq 'loss reconstruction)))
          (cons 'reconstruction-total (cdr (assq 'total reconstruction)))
          (cons 'next-context-loss (cdr (assq 'loss next-context)))
          (cons 'next-context-total (cdr (assq 'total next-context))))))

(defun photon-make-loss-totals ()
  (list (cons 'next-token-loss 0)
        (cons 'reconstruction-loss 0)
        (cons 'reconstruction-total 0)
        (cons 'next-context-loss 0)
        (cons 'next-context-total 0)))

(defun photon-loss-totals-add (totals report)
  (let ((fields '(next-token-loss reconstruction-loss reconstruction-total
                                  next-context-loss next-context-total)))
    (while fields
      (let* ((field (car fields))
             (cell (assq field totals))
             (value (cdr (assq field report))))
        (setcdr cell (+ (cdr cell) value)))
      (setq fields (cdr fields))))
  totals)

(defun photon-loss-totals-result (model totals)
  (let* ((reconstruction-total (cdr (assq 'reconstruction-total totals)))
         (next-context-total (cdr (assq 'next-context-total totals)))
         (reconstruction-loss (cdr (assq 'reconstruction-loss totals)))
         (next-context-loss (cdr (assq 'next-context-loss totals)))
         (weighted-next-token-loss
          (truncate (* 1000
                       (cdr (assq 'next-token-loss totals))
                       (photon-model-param-get model 'next-token-weight))))
         (weighted-reconstruction-loss
          (truncate (* 1000
                       reconstruction-loss
                       (photon-model-param-get model 'reconstruction-weight))))
         (weighted-next-context-loss
          (truncate (* next-context-loss
                       (photon-model-param-get model 'next-context-weight)))))
    (append
     totals
     (list (cons 'reconstruction-loss-permil
                 (if (= reconstruction-total 0)
                     0
                   (/ (* 1000 reconstruction-loss) reconstruction-total)))
           (cons 'next-context-loss-permil
                 (if (= next-context-total 0)
                     0
                   (/ next-context-loss next-context-total)))
           (cons 'weighted-next-token-loss weighted-next-token-loss)
           (cons 'weighted-reconstruction-loss weighted-reconstruction-loss)
           (cons 'weighted-next-context-loss weighted-next-context-loss)
           (cons 'total-loss
                 (+ weighted-next-token-loss
                    weighted-reconstruction-loss
                    weighted-next-context-loss))))))

(defun photon-train-one-from-report (model context target learning-rate report)
  (if (<= learning-rate 0.0)
      (if (= (cdr (assq 'prediction report)) target) 1 0)
    (let ((correct (photon-update-next-token-from-report
                    model context target report learning-rate)))
      (photon-update-reconstruction-from-report model context report learning-rate)
      (photon-update-next-context-from-report model report learning-rate)
      correct)))

(defun photon-train-epoch (model pairs learning-rate)
  (let ((correct 0)
        (total 0)
        (loss-totals (photon-make-loss-totals)))
    (while pairs
      (let ((report (photon-loss-report model
                                        (car (car pairs))
                                        (cdr (car pairs)))))
        (setq loss-totals (photon-loss-totals-add loss-totals report))
        (setq correct
              (+ correct
                 (photon-train-one-from-report
                  model (car (car pairs)) (cdr (car pairs))
                  learning-rate report))))
      (setq total (1+ total))
      (setq pairs (cdr pairs)))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'loss (photon-loss-totals-result model loss-totals)))))

(defun photon-train-epoch-stream (model tokens context-size learning-rate)
  (let ((correct 0)
        (total 0)
        (loss-totals (photon-make-loss-totals))
        (window nil)
        (remaining tokens)
        (filled 0))
    (while (and remaining (< filled context-size))
      (setq window (cons (car remaining) window))
      (setq remaining (cdr remaining))
      (setq filled (1+ filled)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while remaining
        (let ((report (photon-loss-report model window (car remaining))))
          (setq loss-totals (photon-loss-totals-add loss-totals report))
          (setq correct
                (+ correct
                   (photon-train-one-from-report
                    model window (car remaining) learning-rate report))))
        (setq total (1+ total))
        (setq window (photon-window-slide window (car remaining)))
        (setq remaining (cdr remaining))))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'loss (photon-loss-totals-result model loss-totals)))))

(defun photon-train-epoch-token-reader (model next-token context-size learning-rate)
  (let ((correct 0)
        (total 0)
        (loss-totals (photon-make-loss-totals))
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let ((report (photon-loss-report model window token)))
          (setq loss-totals (photon-loss-totals-add loss-totals report))
          (setq correct
                (+ correct
                   (photon-train-one-from-report
                    model window token learning-rate report))))
        (setq total (1+ total))
        (setq window (photon-window-slide window token))
        (setq token (funcall next-token))))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'loss (photon-loss-totals-result model loss-totals)))))

(defun photon-train-epoch-minibatch (model tokens context-size learning-rate batch-size)
  (let ((correct 0)
        (total 0)
        (loss-totals (photon-make-loss-totals))
        (batch-count 0)
        (grad nil)
        (window nil)
        (remaining tokens)
        (filled 0))
    (while (and remaining (< filled context-size))
      (setq window (cons (car remaining) window))
      (setq remaining (cdr remaining))
      (setq filled (1+ filled)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while remaining
        (let* ((target (car remaining))
               (report (photon-loss-report model window target))
               (prediction (cdr (assq 'prediction report)))
               (next-token-weight (photon-model-param-get model
                                                          'next-token-weight)))
          (setq loss-totals (photon-loss-totals-add loss-totals report))
          (when (> next-token-weight 0.0)
            (photon-update-target-prototype
             model (cdr (assq 'state report)) target
             (* learning-rate next-token-weight)))
          (when (> next-token-weight 0.0)
            (photon-model-remember-next-token model window target))
          (if (= prediction target)
              (setq correct (1+ correct))
            (setq grad
                  (photon-gradient-accumulate-miss
                   grad
                   (photon-vector-scale next-token-weight
                                        (cdr (assq 'state report)))
                   prediction target))
            (photon-update-embedding-table model window prediction target
                                           (* 0.25 learning-rate
                                              next-token-weight))
            (photon-update-architecture-params model
                                               (* learning-rate
                                                  next-token-weight))
            (photon-update-layer-gains model prediction target
                                       (* learning-rate next-token-weight))
            (photon-update-layer-projections model prediction target
                                             (* learning-rate
                                                next-token-weight)))
          (photon-update-reconstruction-from-report model window report
                                                    learning-rate)
          (photon-update-next-context-from-report model report learning-rate)
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-long-teacher-state-path
             model window target report (* learning-rate next-token-weight)))
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-continuation-teacher-state-path
             model window target report (* learning-rate next-token-weight)))
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-start-long-teacher-state-path
             model window target report (* learning-rate next-token-weight)))
          (setq total (1+ total))
          (setq batch-count (1+ batch-count))
          (when (>= batch-count batch-size)
            (photon-apply-output-gradient model grad learning-rate batch-count)
            (setq grad nil)
            (setq batch-count 0))
          (setq window (photon-window-slide window target))
          (setq remaining (cdr remaining)))))
    (when (> batch-count 0)
      (photon-apply-output-gradient model grad learning-rate batch-count))
    (list (cons 'correct correct)
          (cons 'total total)
          (cons 'accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'loss (photon-loss-totals-result model loss-totals))
          (cons 'batch-size batch-size))))

(defun photon-pair-shuffle-score (pair seed)
  (sxhash (list seed (car pair) (cdr pair))))

(defun photon-shuffle-pairs (pairs seed)
  (mapcar
   'cdr
   (sort
    (mapcar (lambda (pair)
              (cons (photon-pair-shuffle-score pair seed) pair))
            pairs)
    (lambda (left right) (< (car left) (car right))))))

(defun photon-split-list (items numerator denominator)
  (let* ((total (length items))
         (left-count (if (= denominator 0)
                         total
                       (/ (* total numerator) denominator)))
         (left nil)
         (right nil)
         (index 0))
    (while items
      (if (< index left-count)
          (setq left (cons (car items) left))
        (setq right (cons (car items) right)))
      (setq index (1+ index))
      (setq items (cdr items)))
    (list (cons 'train (nreverse left))
          (cons 'eval (nreverse right)))))

(defun photon-dataset-split (tokens context-size &optional train-numerator
                                    train-denominator)
  (photon-split-list
   (photon-training-pairs tokens context-size)
   (or train-numerator 4)
   (or train-denominator 5)))

(defun photon-make-batch-dataset (name tokens context-size &optional metadata)
  (let* ((pairs (photon-training-pairs tokens context-size))
         (token-count (length tokens)))
    (append
     (list (cons 'name name)
           (cons 'tokens tokens)
           (cons 'context-size context-size)
           (cons 'token-count token-count)
           (cons 'pair-count (length pairs)))
     metadata)))

(defun photon-token-frequency-table (tokens)
  (let ((counts nil))
    (while tokens
      (let ((cell (assq (car tokens) counts)))
        (if cell
            (setcdr cell (1+ (cdr cell)))
          (setq counts (cons (cons (car tokens) 1) counts))))
      (setq tokens (cdr tokens)))
    counts))

(defun photon-token-kind-counts-inc (counts kind)
  (let ((cell (assq kind counts)))
    (if cell
        (setcdr cell (1+ (cdr cell)))
      (setq counts (cons (cons kind 1) counts)))
    counts))

(defun photon-token-diagnostics (vocab tokens &optional tokenizer)
  (let ((total 0)
        (unknown 0)
        (rare 0)
        (long 0)
        (short 0)
        (special 0)
        (kind-counts nil)
        (frequencies (photon-token-frequency-table tokens))
        (walk tokens))
    (while walk
      (let* ((token (car walk))
             (count (or (cdr (assq token frequencies)) 0))
             (kind (if (eq tokenizer 'wordpiece)
                       (photon-wordpiece-token-kind vocab token)
                     'token)))
        (setq total (1+ total))
        (setq kind-counts (photon-token-kind-counts-inc kind-counts kind))
        (when (eq kind 'unknown)
          (setq unknown (1+ unknown)))
        (when (eq kind 'long-piece)
          (setq long (1+ long)))
        (when (eq kind 'short-piece)
          (setq short (1+ short)))
        (when (eq kind 'special)
          (setq special (1+ special)))
        (when (<= count 1)
          (setq rare (1+ rare))))
      (setq walk (cdr walk)))
    (list (cons 'vocab-size (length vocab))
          (cons 'token-count total)
          (cons 'unknown-token-count unknown)
          (cons 'unknown-token-permil
                (if (= total 0) 0 (/ (* 1000 unknown) total)))
          (cons 'rare-token-count rare)
          (cons 'rare-token-permil
                (if (= total 0) 0 (/ (* 1000 rare) total)))
          (cons 'long-piece-token-count long)
          (cons 'short-piece-token-count short)
          (cons 'special-token-count special)
          (cons 'token-kind-counts kind-counts))))

(defun photon-load-batch-dataset-text-file (path context-size &optional options)
  (let* ((tokenizer (or (cdr (assq 'tokenizer options)) 'char))
         (raw-text (photon-read-text-file path))
         (max-chars (cdr (assq 'max-chars options)))
         (text (if (and max-chars (> max-chars 0)
                        (> (length raw-text) max-chars))
                   (substring raw-text 0 max-chars)
                 raw-text))
         (piece-size (or (cdr (assq 'wordpiece-max-piece-size options)) 4))
         (vocab-path (cdr (assq 'vocab-path options)))
         (provided-vocab (cdr (assq 'vocab options)))
         vocab
         tokens
         token-diagnostics
         metadata)
    (if (eq tokenizer 'wordpiece)
        (progn
          (setq vocab
                (or provided-vocab
                    (and vocab-path (file-exists-p vocab-path)
                         (photon-load-vocab vocab-path))
                    (photon-build-wordpiece-vocab text piece-size)))
          (setq tokens (photon-encode-wordpiece-text vocab text piece-size))
          (setq token-diagnostics
                (photon-token-diagnostics vocab tokens tokenizer))
          (when (and vocab-path (not (file-exists-p vocab-path)))
            (photon-save-vocab
             vocab vocab-path
             (list (cons 'tokenizer 'wordpiece)
                   (cons 'wordpiece-max-piece-size piece-size)
                   (cons 'source-chars (length raw-text))
                   (cons 'used-chars (length text))
                   (cons 'token-diagnostics token-diagnostics)
                   (cons 'source-path path))))
          (setq metadata
                (list (cons 'tokenizer 'wordpiece)
                      (cons 'vocab vocab)
                      (cons 'vocab-path vocab-path)
                      (cons 'wordpiece-max-piece-size piece-size)
                      (cons 'source-chars (length raw-text))
                      (cons 'used-chars (length text))
                      (cons 'token-diagnostics token-diagnostics)
                      (cons 'source-path path))))
      (setq vocab (photon-build-char-vocab text))
      (setq tokens (photon-encode-text vocab text))
      (setq token-diagnostics
            (photon-token-diagnostics vocab tokens tokenizer))
      (setq metadata
            (list (cons 'tokenizer 'char)
                  (cons 'vocab vocab)
                  (cons 'source-chars (length raw-text))
                  (cons 'used-chars (length text))
                  (cons 'token-diagnostics token-diagnostics)
                  (cons 'source-path path))))
    (photon-make-batch-dataset path tokens context-size metadata)))

(defun photon-batch-dataset-apply-metadata (model dataset)
  (let ((tokenizer (cdr (assq 'tokenizer dataset)))
        (vocab (cdr (assq 'vocab dataset)))
        (piece-size (cdr (assq 'wordpiece-max-piece-size dataset)))
        (source-chars (cdr (assq 'source-chars dataset)))
        (used-chars (cdr (assq 'used-chars dataset)))
        (token-diagnostics (cdr (assq 'token-diagnostics dataset))))
    (when tokenizer
      (photon-model-metadata-put model 'tokenizer tokenizer))
    (when vocab
      (photon-model-metadata-put model 'vocab vocab))
    (when piece-size
      (photon-model-metadata-put model 'wordpiece-max-piece-size piece-size))
    (when source-chars
      (photon-model-metadata-put model 'source-chars source-chars))
    (when used-chars
      (photon-model-metadata-put model 'used-chars used-chars))
    (when token-diagnostics
      (photon-model-metadata-put model 'token-diagnostics token-diagnostics))
    (photon-model-metadata-put
     model 'context-size (cdr (assq 'context-size dataset)))
    (when (eq tokenizer 'wordpiece)
      (photon-model-apply-wordpiece-readout-defaults model)))
  model)

(defun photon-training-history-last-epoch (history)
  (let ((last -1))
    (while history
      (let ((epoch (car (car history))))
        (when (and (integerp epoch) (> epoch last))
          (setq last epoch)))
      (setq history (cdr history)))
    last))

(defun photon-eval-total-loss (eval)
  (cdr (assq 'total-loss (cdr (assq 'loss eval)))))

(defun photon-eval-better-p (candidate current-best)
  (if (null current-best)
      t
    (let ((candidate-loss (photon-eval-total-loss candidate))
          (best-loss (photon-eval-total-loss current-best))
          (candidate-fallback
           (or (cdr (assq 'fallback-accuracy-permil candidate)) 0))
          (best-fallback
           (or (cdr (assq 'fallback-accuracy-permil current-best)) 0))
          (candidate-accuracy (cdr (assq 'accuracy-permil candidate)))
          (best-accuracy (cdr (assq 'accuracy-permil current-best))))
      (or (< candidate-loss best-loss)
          (and (= candidate-loss best-loss)
               (> candidate-fallback best-fallback))
          (and (= candidate-loss best-loss)
               (= candidate-fallback best-fallback)
               (> candidate-accuracy best-accuracy))))))

(defun photon-training-history-entry (epoch train eval checkpoint-path
                                            &optional best-checkpoint-path
                                            best-p metadata)
  (cons epoch
        (list
         (cons 'train train)
         (cons 'eval eval)
         (cons 'snapshot
               (append
                (list (cons 'epoch epoch)
                      (cons 'train-accuracy-permil
                            (cdr (assq 'accuracy-permil train)))
                      (cons 'eval-accuracy-permil
                            (cdr (assq 'accuracy-permil eval)))
                      (cons 'eval-fallback-accuracy-permil
                            (cdr (assq 'fallback-accuracy-permil eval)))
                      (cons 'eval-loss (cdr (assq 'loss eval)))
                      (cons 'checkpoint-path checkpoint-path)
                      (cons 'best-checkpoint-path best-checkpoint-path)
                      (cons 'best-checkpoint-updated best-p))
                metadata)))))

(defun photon-training-history-csv-line (entry)
  (let* ((epoch (car entry))
         (body (cdr entry))
         (train (cdr (assq 'train body)))
         (eval (cdr (assq 'eval body)))
         (snapshot (cdr (assq 'snapshot body))))
    (format "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
            epoch
            (cdr (assq 'accuracy-permil train))
            (photon-eval-total-loss train)
            (cdr (assq 'accuracy-permil eval))
            (photon-eval-total-loss eval)
            (or (cdr (assq 'fallback-accuracy-permil eval)) "")
            (or (cdr (assq 'epoch-elapsed-seconds snapshot)) "")
            (or (cdr (assq 'total-elapsed-seconds snapshot)) "")
            (cdr (assq 'best-checkpoint-updated snapshot))
            (or (cdr (assq 'checkpoint-path snapshot)) ""))))

(defun photon-save-training-history-csv (history path)
  (with-temp-file path
    (insert "epoch,train_accuracy_permil,train_total_loss,eval_accuracy_permil,eval_total_loss,eval_fallback_accuracy_permil,epoch_elapsed_seconds,total_elapsed_seconds,best_updated,checkpoint_path\n")
    (while history
      (insert (photon-training-history-csv-line (car history)))
      (setq history (cdr history))))
  path)

(defun photon-save-training-history-sexp (history path &optional metadata)
  (photon-save-training-history history path metadata))

(defun photon-make-fallback-totals ()
  (list (cons 'total 0)
        (cons 'correct 0)
        (cons 'top-3-correct 0)
        (cons 'top-5-correct 0)
        (cons 'memory-hits 0)
        (cons 'margin-sum 0.0)))

(defun photon-fallback-totals-inc (totals key amount)
  (let ((cell (assq key totals)))
    (setcdr cell (+ (cdr cell) amount)))
  totals)

(defun photon-fallback-totals-add-report (totals report target)
  (photon-fallback-totals-inc totals 'total 1)
  (when (= (cdr (assq 'fallback-prediction report)) target)
    (photon-fallback-totals-inc totals 'correct 1))
  (when (cdr (assq 'fallback-top-3-hit report))
    (photon-fallback-totals-inc totals 'top-3-correct 1))
  (when (cdr (assq 'fallback-top-5-hit report))
    (photon-fallback-totals-inc totals 'top-5-correct 1))
  (when (cdr (assq 'memory-hit report))
    (photon-fallback-totals-inc totals 'memory-hits 1))
  (photon-fallback-totals-inc
   totals 'margin-sum (cdr (assq 'target-margin report)))
  totals)

(defun photon-fallback-totals-result (totals)
  (let ((total (cdr (assq 'total totals)))
        (correct (cdr (assq 'correct totals)))
        (top-3-correct (cdr (assq 'top-3-correct totals)))
        (top-5-correct (cdr (assq 'top-5-correct totals)))
        (memory-hits (cdr (assq 'memory-hits totals)))
        (margin-sum (cdr (assq 'margin-sum totals))))
    (list (cons 'fallback-correct correct)
          (cons 'fallback-total total)
          (cons 'fallback-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'fallback-top-3-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
          (cons 'fallback-top-5-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
          (cons 'memory-hit-permil
                (if (= total 0) 0 (/ (* 1000 memory-hits) total)))
          (cons 'average-target-margin
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-train-epoch-pairs-minibatch
    (model pairs learning-rate batch-size)
  (let ((correct 0)
        (total 0)
        (loss-totals (photon-make-loss-totals))
        (fallback-totals (photon-make-fallback-totals))
        (batch-count 0)
        (grad nil)
        (training (> learning-rate 0.0)))
    (while pairs
      (let* ((pair (car pairs))
             (context (car pair))
             (target (cdr pair))
             (report (photon-loss-report model context target))
             (prediction (cdr (assq 'prediction report)))
             (state (cdr (assq 'state report)))
             (next-token-weight
              (photon-model-param-get model 'next-token-weight)))
        (setq loss-totals (photon-loss-totals-add loss-totals report))
        (unless training
          (setq fallback-totals
                (photon-fallback-totals-add-report
                 fallback-totals report target)))
        (when (and training (> next-token-weight 0.0))
          (photon-update-target-prototype
           model state target (* learning-rate next-token-weight)))
        (when (and training (> next-token-weight 0.0))
          (photon-model-remember-next-token model context target))
        (if (= prediction target)
            (setq correct (1+ correct))
          (when training
            (setq grad
                  (photon-gradient-accumulate-miss
                   grad
                   (photon-vector-scale next-token-weight state)
                   prediction target))
            (photon-update-embedding-table model context prediction target
                                           (* 0.25 learning-rate
                                              next-token-weight))
            (photon-update-architecture-params
             model (* learning-rate next-token-weight))
            (photon-update-layer-gains
             model prediction target (* learning-rate next-token-weight))
            (photon-update-layer-projections
             model prediction target (* learning-rate next-token-weight))))
        (when training
          (photon-update-reconstruction-from-report
           model context report learning-rate)
          (photon-update-next-context-from-report model report learning-rate)
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-long-teacher-state-path
             model context target report (* learning-rate next-token-weight)))
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-continuation-teacher-state-path
             model context target report (* learning-rate next-token-weight)))
          (when (> next-token-weight 0.0)
            (photon-update-wordpiece-start-long-teacher-state-path
             model context target report (* learning-rate next-token-weight))))
        (setq total (1+ total))
        (setq batch-count (1+ batch-count))
        (when (and training (>= batch-count batch-size))
          (photon-apply-output-gradient model grad learning-rate batch-count)
          (setq grad nil)
          (setq batch-count 0)))
      (setq pairs (cdr pairs)))
    (when (and training (> batch-count 0))
      (photon-apply-output-gradient model grad learning-rate batch-count))
    (append
     (list (cons 'correct correct)
           (cons 'total total)
           (cons 'accuracy-permil
                 (if (= total 0) 0 (/ (* 1000 correct) total)))
           (cons 'loss (photon-loss-totals-result model loss-totals))
           (cons 'batch-size batch-size))
     (unless training
       (photon-fallback-totals-result fallback-totals)))))

(defun photon-evaluate-pairs (model pairs)
  (photon-train-epoch-pairs-minibatch model pairs 0.0 1))

(defun photon-evaluate-pairs-fallback (model pairs)
  (let ((total 0)
        (correct 0)
        (top-3-correct 0)
        (top-5-correct 0)
        (memory-hits 0)
        (margin-sum 0.0))
    (while pairs
      (let* ((pair (car pairs))
             (context (car pair))
             (target (cdr pair))
             (states (photon-forward-model model context))
             (state (photon-prediction-state states context))
             (logits (photon-model-readout-logits model state context))
             (prediction (photon-argmax-index logits))
             (top-3 (photon-top-k-logits logits 3))
             (top-5 (photon-top-k-logits logits 5))
             (rival (photon-best-non-target-index logits target)))
        (when (= prediction target)
          (setq correct (1+ correct)))
        (when (photon-top-k-hit-p top-3 target)
          (setq top-3-correct (1+ top-3-correct)))
        (when (photon-top-k-hit-p top-5 target)
          (setq top-5-correct (1+ top-5-correct)))
        (when (photon-model-next-token-memory-raw-hit-p model context)
          (setq memory-hits (1+ memory-hits)))
        (setq margin-sum
              (+ margin-sum (photon-target-margin logits target rival)))
        (setq total (1+ total)))
      (setq pairs (cdr pairs)))
    (list (cons 'fallback-correct correct)
          (cons 'fallback-total total)
          (cons 'fallback-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 correct) total)))
          (cons 'fallback-top-3-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
          (cons 'fallback-top-5-accuracy-permil
                (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
          (cons 'memory-hit-permil
                (if (= total 0) 0 (/ (* 1000 memory-hits) total)))
          (cons 'average-target-margin
                (if (= total 0) 0.0 (/ margin-sum total))))))

(defun photon-train-batch-options
    (model tokens context-size epochs learning-rate batch-size &optional options)
  (let* ((checkpoint-path (cdr (assq 'checkpoint-path options)))
         (best-checkpoint-path (cdr (assq 'best-checkpoint-path options)))
         (history-path (cdr (assq 'history-path options)))
         (history-csv-path (cdr (assq 'history-csv-path options)))
         (checkpoint-every (or (cdr (assq 'checkpoint-every options)) 1))
         (progress (cdr (assq 'progress options)))
         (resume (cdr (assq 'resume options)))
         (shuffle (cdr (assq 'shuffle options)))
         (restore-best-cell (assq 'restore-best options))
         (restore-best (if restore-best-cell (cdr restore-best-cell) t))
         (split (photon-dataset-split
                 tokens context-size
                 (or (cdr (assq 'train-split-numerator options)) 4)
                 (or (cdr (assq 'train-split-denominator options)) 5)))
         (train-pairs (cdr (assq 'train split)))
         (eval-pairs (cdr (assq 'eval split)))
         (history nil)
         (epoch 0)
         (end-epoch epochs)
         (best-eval nil)
         (best-epoch nil)
         (run-start (float-time)))
    (when (and resume checkpoint-path (file-exists-p checkpoint-path))
      (setq model (photon-load-model checkpoint-path)))
    (when (and resume history-path (file-exists-p history-path))
      (setq history (photon-load-training-history history-path))
      (setq epoch (1+ (photon-training-history-last-epoch history)))
      (setq end-epoch (+ epoch epochs)))
    (when history
      (let ((walk history))
        (while walk
          (let ((eval (cdr (assq 'eval (cdr (car walk))))))
            (when (photon-eval-better-p eval best-eval)
              (setq best-eval eval)
              (setq best-epoch (car (car walk)))))
          (setq walk (cdr walk)))))
    (photon-model-clear-cache model)
    (while (< epoch end-epoch)
      (let ((epoch-pairs train-pairs)
            (epoch-start (float-time)))
        (when shuffle
          (setq epoch-pairs (photon-shuffle-pairs epoch-pairs epoch)))
        (let* ((train-result
                (photon-train-epoch-pairs-minibatch
                 model epoch-pairs learning-rate batch-size))
               (eval-result (photon-evaluate-pairs model eval-pairs))
               (epoch-elapsed (- (float-time) epoch-start))
               (total-elapsed (- (float-time) run-start))
               (best-updated (photon-eval-better-p eval-result best-eval)))
          (when best-updated
            (setq best-eval eval-result)
            (setq best-epoch epoch)
            (when best-checkpoint-path
              (photon-save-model model best-checkpoint-path)))
          (when progress
            (message "photon epoch=%s train=%s eval=%s fallback=%s eval-loss=%s elapsed=%.3f total=%.3f best=%s"
                     epoch
                     (cdr (assq 'accuracy-permil train-result))
                     (cdr (assq 'accuracy-permil eval-result))
                     (cdr (assq 'fallback-accuracy-permil eval-result))
                     (photon-eval-total-loss eval-result)
                     epoch-elapsed
                     total-elapsed
                     best-epoch))
          (setq history
                (append
                 history
                 (list
                  (photon-training-history-entry
                   epoch train-result eval-result checkpoint-path
                   best-checkpoint-path best-updated
                   (list (cons 'epoch-elapsed-seconds epoch-elapsed)
                         (cons 'total-elapsed-seconds total-elapsed))))))))
      (when (and checkpoint-path
                 (> checkpoint-every 0)
                 (= (mod (1+ epoch) checkpoint-every) 0))
        (photon-save-model model checkpoint-path))
      (when history-path
        (photon-save-training-history history history-path
                                      (list (cons 'checkpoint-path
                                                  checkpoint-path)
                                            (cons 'best-checkpoint-path
                                                  best-checkpoint-path)
                                            (cons 'context-size context-size)
                                            (cons 'batch-size batch-size))))
      (when history-csv-path
        (photon-save-training-history-csv history history-csv-path))
      (setq epoch (1+ epoch)))
    (when (and restore-best best-checkpoint-path best-epoch
               (file-exists-p best-checkpoint-path))
      (setq model (photon-load-model best-checkpoint-path)))
    (when (> learning-rate 0.0)
      (photon-rebuild-target-state-clusters model tokens context-size)
      (when (and restore-best best-checkpoint-path best-epoch
                 (file-exists-p best-checkpoint-path))
        (photon-save-model model best-checkpoint-path)))
    (list (cons 'model model)
          (cons 'train-pairs (length train-pairs))
          (cons 'eval-pairs (length eval-pairs))
          (cons 'checkpoint-path checkpoint-path)
          (cons 'best-checkpoint-path best-checkpoint-path)
          (cons 'best-epoch best-epoch)
          (cons 'best-eval best-eval)
          (cons 'restored-best restore-best)
          (cons 'history-path history-path)
          (cons 'history-csv-path history-csv-path)
          (cons 'history history))))

(defun photon-train-batch-dataset
    (model dataset epochs learning-rate batch-size &optional options)
  (photon-batch-dataset-apply-metadata model dataset)
  (let ((result
         (photon-train-batch-options
          model
          (cdr (assq 'tokens dataset))
          (cdr (assq 'context-size dataset))
          epochs learning-rate batch-size options)))
    (append result
            (list (cons 'dataset-name (cdr (assq 'name dataset)))
                  (cons 'token-count (cdr (assq 'token-count dataset)))
                  (cons 'pair-count (cdr (assq 'pair-count dataset)))))))

(defun photon-train-minibatch (model tokens context-size epochs learning-rate batch-size)
  (let ((history nil)
        (epoch 0))
    (photon-model-clear-cache model)
    (while (< epoch epochs)
      (setq history
            (cons (cons epoch
                        (photon-train-epoch-minibatch
                         model tokens context-size learning-rate batch-size))
                  history))
      (setq epoch (1+ epoch)))
    (when (> learning-rate 0.0)
      (photon-rebuild-target-state-clusters model tokens context-size))
    (nreverse history)))

(defun photon-train (model tokens context-size epochs learning-rate)
  (let ((history nil)
        (epoch 0))
    (photon-model-clear-cache model)
    (while (< epoch epochs)
      (setq history
            (cons (cons epoch
                        (photon-train-epoch-stream model tokens context-size
                                                   learning-rate))
                  history))
      (setq epoch (1+ epoch)))
    (when (> learning-rate 0.0)
      (photon-rebuild-target-state-clusters model tokens context-size))
    (nreverse history)))

(defun photon-evaluate (model tokens context-size)
  (photon-train-epoch-stream model tokens context-size 0.0))

(defun photon-string-chars (text)
  (let ((chars nil)
        (i 0))
    (while (< i (length text))
      (setq chars (cons (substring text i (1+ i)) chars))
      (setq i (1+ i)))
    (nreverse chars)))

(defun photon-member-equal-p (value items)
  (let ((found nil))
    (while items
      (when (equal value (car items))
        (setq found t)
        (setq items nil))
      (when items
        (setq items (cdr items))))
    found))

(defun photon-sort-strings (items)
  (sort items 'string<))

(defun photon-build-char-vocab (text)
  (let ((chars (photon-string-chars text))
        (seen nil))
    (while chars
      (unless (photon-member-equal-p (car chars) seen)
        (setq seen (cons (car chars) seen)))
      (setq chars (cdr chars)))
    (photon-sort-strings seen)))

(defun photon-vocab-index (vocab char)
  (let ((index 0)
        (found nil))
    (while (and vocab (not found))
      (if (equal (car vocab) char)
          (setq found index)
        (setq vocab (cdr vocab))
        (setq index (1+ index))))
    found))

(defun photon-encode-text (vocab text)
  (let ((chars (photon-string-chars text))
        (tokens nil))
    (while chars
      (let ((index (photon-vocab-index vocab (car chars))))
        (when index
          (setq tokens (cons index tokens))))
      (setq chars (cdr chars)))
    (nreverse tokens)))

(defun photon-file-size (path)
  (nth 7 (file-attributes path)))

(defun photon-make-file-char-reader (path chunk-bytes)
  (let ((position 0)
        (size (photon-file-size path))
        (chunk-size (or chunk-bytes 4096))
        (buffer "")
        (index 0))
    (lambda ()
      (while (and (>= index (length buffer))
                  (< position size))
        (let ((end (min size (+ position chunk-size))))
          (with-temp-buffer
            (insert-file-contents path nil position end)
            (setq buffer (buffer-string))
            (setq index 0)
            (setq position end))))
      (if (< index (length buffer))
          (let ((char (substring buffer index (1+ index))))
            (setq index (1+ index))
            char)
        nil))))

(defun photon-build-char-vocab-file-stream (path &optional chunk-bytes)
  (let ((reader (photon-make-file-char-reader path chunk-bytes))
        (seen nil)
        (char nil))
    (setq char (funcall reader))
    (while char
      (unless (photon-member-equal-p char seen)
        (setq seen (cons char seen)))
      (setq char (funcall reader)))
    (photon-sort-strings seen)))

(defun photon-make-file-token-reader (path vocab &optional chunk-bytes)
  (let ((reader (photon-make-file-char-reader path chunk-bytes)))
    (lambda ()
      (let ((char (funcall reader))
            (token nil))
        (while (and char (null token))
          (setq token (photon-vocab-index vocab char))
          (when (null token)
            (setq char (funcall reader))))
        token))))

(defun photon-wordpiece-word-char-p (char)
  (or (string-match-p "\\`[A-Za-z0-9]\\'" char)
      (string= char "_")))

(defun photon-wordpiece-basic-pieces (text)
  (let ((chars (photon-string-chars text))
        (word "")
        (pieces nil))
    (while chars
      (let ((char (car chars)))
        (if (photon-wordpiece-word-char-p char)
            (setq word (concat word char))
          (progn
            (when (> (length word) 0)
              (setq pieces (cons word pieces))
              (setq word ""))
            (setq pieces (cons char pieces)))))
      (setq chars (cdr chars)))
    (when (> (length word) 0)
      (setq pieces (cons word pieces)))
    (nreverse pieces)))

(defun photon-wordpiece-add-seen (piece seen)
  (if (photon-member-equal-p piece seen)
      seen
    (cons piece seen)))

(defun photon-wordpiece-special-tokens ()
  '("[PAD]" "[UNK]" "[BOS]" "[EOS]"))

(defun photon-wordpiece-unknown-token-id (vocab)
  (or (photon-vocab-index vocab "[UNK]") 1))

(defun photon-wordpiece-add-word (word max-piece-size seen)
  (let ((index 0)
        (size (max 1 max-piece-size)))
    (while (< index (length word))
      (let* ((end (min (length word) (+ index size)))
             (piece (substring word index end)))
        (setq seen
              (photon-wordpiece-add-seen
               (if (= index 0) piece (concat "##" piece))
               seen))
        (let ((char-index index))
          (while (< char-index end)
            (let ((char-piece
                   (substring word char-index (1+ char-index))))
              (setq seen
                    (photon-wordpiece-add-seen
                     (if (= char-index 0)
                         char-piece
                       (concat "##" char-piece))
                     seen)))
            (setq char-index (1+ char-index)))))
      (setq index (+ index size)))
    seen))

(defun photon-build-wordpiece-vocab (text &optional max-piece-size)
  (let ((pieces (photon-wordpiece-basic-pieces text))
        (size (or max-piece-size 4))
        (seen nil))
    (while pieces
      (let ((piece (car pieces)))
        (if (and (> (length piece) 0)
                 (photon-wordpiece-word-char-p
                  (substring piece 0 1)))
            (setq seen (photon-wordpiece-add-word piece size seen))
          (setq seen (photon-wordpiece-add-seen piece seen))))
      (setq pieces (cdr pieces)))
    (append (photon-wordpiece-special-tokens)
            (photon-sort-strings seen))))

(defun photon-save-vocab (vocab path &optional metadata)
  (with-temp-file path
    (insert (prin1-to-string (list (cons 'format 'photon-vocab-v1)
                 (cons 'version photon-version)
                 (cons 'metadata metadata)
                 (cons 'vocab vocab))
           )))
  path)

(defun photon-load-vocab (path)
  (with-temp-buffer
    (insert-file-contents path)
    (let ((data (photon--read-sexp-string (buffer-string))))
      (if (and (consp data)
               (eq (cdr (assq 'format data)) 'photon-vocab-v1)
               (assq 'vocab data))
          (cdr (assq 'vocab data))
        data))))

(defun photon-wordpiece-greedy-piece (vocab word index max-piece-size)
  (let ((size (min (or max-piece-size 4) (- (length word) index)))
        (found nil))
    (while (and (> size 0) (null found))
      (let* ((piece (substring word index (+ index size)))
             (candidate (if (= index 0) piece (concat "##" piece))))
        (when (photon-vocab-index vocab candidate)
          (setq found candidate)))
      (setq size (1- size)))
    (or found
        (or (and (photon-vocab-index vocab "[UNK]") "[UNK]")
            (if (= index 0)
                (substring word index (1+ index))
              (concat "##" (substring word index (1+ index))))))))

(defun photon-encode-wordpiece-text (vocab text &optional max-piece-size
                                           add-special-tokens)
  (let ((pieces (photon-wordpiece-basic-pieces text))
        (tokens nil))
    (when add-special-tokens
      (let ((bos (photon-vocab-index vocab "[BOS]")))
        (when bos
          (setq tokens (cons bos tokens)))))
    (while pieces
      (let ((piece (car pieces)))
        (if (and (> (length piece) 0)
                 (photon-wordpiece-word-char-p
                  (substring piece 0 1)))
            (let ((index 0))
              (while (< index (length piece))
                (let* ((wp (photon-wordpiece-greedy-piece
                            vocab piece index max-piece-size))
                       (token (photon-vocab-index vocab wp))
                       (advance (if (string-prefix-p "##" wp)
                                    (- (length wp) 2)
                                  (length wp))))
                  (setq tokens
                        (cons (or token
                                  (photon-vocab-index vocab "[UNK]"))
                              tokens))
                  (setq index (+ index (max 1 advance))))))
          (let ((token (photon-vocab-index vocab piece)))
            (setq tokens
                  (cons (or token
                            (photon-vocab-index vocab "[UNK]"))
                        tokens)))))
      (setq pieces (cdr pieces)))
    (when add-special-tokens
      (let ((eos (photon-vocab-index vocab "[EOS]")))
        (when eos
          (setq tokens (cons eos tokens)))))
    (nreverse tokens)))

(defun photon-decode-wordpiece-tokens (vocab tokens)
  (let ((text ""))
    (while tokens
      (let ((piece (photon-token-to-char vocab (car tokens))))
        (cond ((or (string= piece "[PAD]")
                   (string= piece "[BOS]")
                   (string= piece "[EOS]")))
              ((string-prefix-p "##" piece)
               (setq text (concat text (substring piece 2))))
              (t
               (setq text (concat text piece)))))
      (setq tokens (cdr tokens)))
    text))

(defun photon-token-to-char (vocab token)
  (or (nth token vocab) "?"))

(defun photon-decode-tokens (vocab tokens)
  (let ((text ""))
    (while tokens
      (setq text (concat text (photon-token-to-char vocab (car tokens))))
      (setq tokens (cdr tokens)))
    text))

(defun photon-model-encode-text (model text)
  (let ((vocab (photon-model-metadata-get model 'vocab))
        (tokenizer (photon-model-metadata-get model 'tokenizer)))
    (if (eq tokenizer 'wordpiece)
        (photon-encode-wordpiece-text
         vocab text
         (or (photon-model-metadata-get model 'wordpiece-max-piece-size)
             4))
      (photon-encode-text vocab text))))

(defun photon-model-apply-wordpiece-readout-defaults (model)
  (photon-model-param-set
   model 'context-anchor-ngram-weight
   (photon-model-param-get model 'wordpiece-context-anchor-ngram-weight))
  (photon-model-param-set model 'readout-logit-temperature 1.0)
  (photon-model-param-set
   model 'output-head-prototype-projection-enabled 1.0)
  (photon-model-param-set model 'output-head-linear-readout-weight 0.0)
  model)

(defun photon-model-apply-wordpiece-final-readout-defaults (model)
  (photon-model-param-set
   model 'context-anchor-ngram-weight
   (photon-model-param-get model
                           'wordpiece-final-context-anchor-ngram-weight))
  (photon-model-param-set
   model 'readout-logit-temperature
   (photon-model-param-get model 'wordpiece-final-readout-temperature))
  (photon-model-param-set
   model 'output-head-prototype-projection-enabled
   (photon-model-param-get
    model 'wordpiece-final-output-head-prototype-projection-enabled))
  (photon-model-param-set
   model 'output-head-linear-readout-weight
   (photon-model-param-get
    model 'wordpiece-final-output-head-linear-readout-weight))
  model)

(defun photon-model-decode-tokens (model tokens)
  (let ((vocab (photon-model-metadata-get model 'vocab))
        (tokenizer (photon-model-metadata-get model 'tokenizer)))
    (if (eq tokenizer 'wordpiece)
        (photon-decode-wordpiece-tokens vocab tokens)
      (photon-decode-tokens vocab tokens))))

(defun photon-char-class (char)
  (cond ((null char) 'unknown)
        ((string= char "\n") 'newline)
        ((string= char " ") 'space)
        ((string-match-p "\\`[0-9]\\'" char) 'digit)
        ((string-match-p "\\`[A-Za-z]\\'" char) 'letter)
        ((string-match-p "\\`[[:punct:]]\\'" char) 'punctuation)
        (t 'other)))

(defun photon-diagnostics-empty-class (class)
  (list (cons 'class class)
        (cons 'total 0)
        (cons 'fallback-correct 0)
        (cons 'exact-correct 0)
        (cons 'top-3-correct 0)
        (cons 'top-5-correct 0)))

(defun photon-diagnostics-class-cell (classes class)
  (let ((cell (assq class classes)))
    (unless cell
      (setq cell (cons class (photon-diagnostics-empty-class class)))
      (setq classes (cons cell classes)))
    (cons cell classes)))

(defun photon-diagnostics-inc (alist key amount)
  (let ((cell (assq key alist)))
    (if cell
        (setcdr cell (+ (cdr cell) amount))
      (setq alist (cons (cons key amount) alist))))
  alist)

(defun photon-diagnostics-class-update (classes class fallback-correct
                                                exact-correct top-3-hit
                                                top-5-hit)
  (let* ((lookup (photon-diagnostics-class-cell classes class))
         (cell (car lookup))
         (updated (cdr lookup))
         (stats (cdr cell)))
    (setq stats (photon-diagnostics-inc stats 'total 1))
    (when fallback-correct
      (setq stats (photon-diagnostics-inc stats 'fallback-correct 1)))
    (when exact-correct
      (setq stats (photon-diagnostics-inc stats 'exact-correct 1)))
    (when top-3-hit
      (setq stats (photon-diagnostics-inc stats 'top-3-correct 1)))
    (when top-5-hit
      (setq stats (photon-diagnostics-inc stats 'top-5-correct 1)))
    (setcdr cell stats)
    updated))

(defun photon-diagnostics-class-finalize (classes)
  (let ((result nil))
    (while classes
      (let* ((stats (cdr (car classes)))
             (total (cdr (assq 'total stats))))
        (setq result
              (cons
               (append stats
                       (list
                        (cons 'fallback-accuracy-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'fallback-correct stats)))
                                   total)))
                        (cons 'exact-accuracy-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'exact-correct stats)))
                                   total)))
                        (cons 'top-3-accuracy-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'top-3-correct stats)))
                                   total)))
                        (cons 'top-5-accuracy-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'top-5-correct stats)))
                                   total)))))
               result)))
      (setq classes (cdr classes)))
    (nreverse result)))

(defun photon-top-k-logits-with-chars (top-k vocab)
  (let ((result nil))
    (while top-k
      (setq result
            (cons (list (cons 'token (car (car top-k)))
                        (cons 'char
                              (photon-token-to-char vocab (car (car top-k))))
                        (cons 'logit (cdr (car top-k))))
                  result))
      (setq top-k (cdr top-k)))
    (nreverse result)))

(defun photon-wordpiece-token-kind (vocab token)
  (let* ((piece (photon-token-to-char vocab token))
         (body (if (string-prefix-p "##" piece)
                   (substring piece 2)
                 piece))
         (length (length body)))
    (cond ((string= piece "[UNK]") 'unknown)
          ((photon-member-equal-p piece (photon-wordpiece-special-tokens))
           'special)
          ((> length 2) 'long-piece)
          (t 'short-piece))))

(defun photon-wordpiece-token-continuation-p (vocab token)
  (string-prefix-p "##" (photon-token-to-char vocab token)))

(defun photon-wordpiece-token-shape (vocab token)
  (let ((kind (photon-wordpiece-token-kind vocab token)))
    (cond
     ((or (eq kind 'special) (eq kind 'unknown)) kind)
     ((photon-wordpiece-token-continuation-p vocab token)
      (if (eq kind 'long-piece)
          'continuation-long-piece
        'continuation-short-piece))
     ((eq kind 'long-piece) 'start-long-piece)
     (t 'start-short-piece))))

(defun photon-wordpiece-token-length (vocab token)
  (let ((piece (photon-token-to-char vocab token)))
    (length (if (string-prefix-p "##" piece)
                (substring piece 2)
              piece))))

(defun photon-top-k-logits-with-pieces (top-k vocab)
  (let ((result nil))
    (while top-k
      (let ((token (car (car top-k))))
        (setq result
              (cons (list (cons 'token token)
                          (cons 'piece (photon-token-to-char vocab token))
                          (cons 'token-kind
                                (photon-wordpiece-token-kind vocab token))
                          (cons 'logit (cdr (car top-k))))
                    result)))
      (setq top-k (cdr top-k)))
    (nreverse result)))

(defun photon-readout-component-record (vocab component logits target fallback)
  (let* ((prediction (photon-argmax-index logits))
         (rival (photon-best-non-target-index logits target))
         (target-logit (nth target logits))
         (fallback-logit (nth fallback logits)))
    (list (cons 'component component)
          (cons 'prediction prediction)
          (cons 'prediction-char (photon-token-to-char vocab prediction))
          (cons 'target-logit target-logit)
          (cons 'fallback-logit fallback-logit)
          (cons 'target-minus-fallback (- target-logit fallback-logit))
          (cons 'target-margin
                (photon-target-margin logits target rival))
          (cons 'top-3
                (photon-top-k-logits-with-chars
                 (photon-top-k-logits logits 3)
                 vocab)))))

(defun photon-readout-component-wordpiece-record
    (vocab component logits target fallback)
  (let* ((prediction (photon-argmax-index logits))
         (rival (photon-best-non-target-index logits target))
         (target-logit (nth target logits))
         (fallback-logit (nth fallback logits)))
    (list (cons 'component component)
          (cons 'prediction prediction)
          (cons 'prediction-piece (photon-token-to-char vocab prediction))
          (cons 'prediction-kind
                (photon-wordpiece-token-kind vocab prediction))
          (cons 'target-logit target-logit)
          (cons 'fallback-logit fallback-logit)
          (cons 'target-minus-fallback (- target-logit fallback-logit))
          (cons 'target-margin
                (photon-target-margin logits target rival))
          (cons 'top-3
                (photon-top-k-logits-with-pieces
                 (photon-top-k-logits logits 3)
                 vocab)))))

(defun photon-readout-contribution-debug (vocab components target fallback)
  (let ((names '(linear-output-head output-head-prototype
                             raw-output-head output-head-cluster output-head
                             prototype state-cluster dense-prototype-readout
                             token-readout embedding-anchor
                             ngram-anchor start-long-boundary-readout
                             start-long-classifier start-long-classifier-readout
                             context-anchor structural-off
                             structural-tie-breaker base-readout
                             char-structural-bias full))
        (result nil))
    (while names
      (let* ((name (car names))
             (logits (cdr (assq name components))))
        (when logits
          (setq result
                (cons (photon-readout-component-record
                       vocab name logits target fallback)
                      result))))
      (setq names (cdr names)))
    (nreverse result)))

(defun photon-readout-wordpiece-contribution-debug
    (vocab components target fallback)
  (let ((names '(linear-output-head output-head-prototype raw-output-head
                                    output-head token-readout prototype
                                    dense-prototype-readout state-cluster
                                    ngram-anchor start-long-boundary-readout
                                    start-long-classifier
                                    start-long-classifier-readout
                                    full))
        (result nil))
    (while names
      (let* ((name (car names))
             (logits (cdr (assq name components))))
        (when logits
          (setq result
                (cons (photon-readout-component-wordpiece-record
                       vocab name logits target fallback)
                      result))))
      (setq names (cdr names)))
    (nreverse result)))

(defun photon-readout-ablation-cell (ablations mode)
  (let ((cell (assq mode ablations)))
    (unless cell
      (setq cell
            (cons mode
                  (list (cons 'mode mode)
                        (cons 'total 0)
                        (cons 'correct 0)
                        (cons 'margin-sum 0.0))))
      (setq ablations (cons cell ablations)))
    (cons cell ablations)))

(defun photon-readout-ablation-update-one (ablations mode logits target)
  (let* ((lookup (photon-readout-ablation-cell ablations mode))
         (cell (car lookup))
         (updated (cdr lookup))
         (stats (cdr cell))
         (prediction (photon-argmax-index logits))
         (rival (photon-best-non-target-index logits target))
         (margin (photon-target-margin logits target rival)))
    (setq stats (photon-diagnostics-inc stats 'total 1))
    (when (= prediction target)
      (setq stats (photon-diagnostics-inc stats 'correct 1)))
    (setq stats (photon-diagnostics-inc stats 'margin-sum margin))
    (setcdr cell stats)
    updated))

(defun photon-readout-ablation-update (ablations components target)
  (let ((modes '((full . full)
                 (linear-output-head-only . linear-output-head)
                 (output-head-prototype-only . output-head-prototype)
                 (raw-output-head-only . raw-output-head)
                 (output-head-only . output-head)
                 (raw-prototype-only . prototype)
                 (state-cluster-only . state-cluster)
                 (prototype-only . dense-prototype-readout)
                 (token-readout-only . token-readout)
                 (embedding-anchor-only . embedding-anchor)
                 (ngram-anchor-only . ngram-anchor)
                 (start-long-boundary-readout-only
                  . start-long-boundary-readout)
                 (start-long-classifier-only . start-long-classifier)
                 (start-long-classifier-readout-only
                  . start-long-classifier-readout)
                 (context-anchor-only . context-anchor)
                 (structural-off . structural-off)
                 (char-structural-bias-off . char-structural-bias-off))))
    (while modes
      (let* ((mode (car (car modes)))
             (component (cdr (car modes)))
             (logits (cdr (assq component components))))
        (when logits
          (setq ablations
                (photon-readout-ablation-update-one
                 ablations mode logits target))))
      (setq modes (cdr modes)))
    ablations))

(defun photon-readout-ablation-finalize (ablations)
  (let ((result nil))
    (while ablations
      (let* ((stats (cdr (car ablations)))
             (total (cdr (assq 'total stats))))
        (setq result
              (cons (append stats
                            (list
                             (cons 'accuracy-permil
                                   (if (= total 0)
                                       0
                                     (/ (* 1000
                                           (cdr (assq 'correct stats)))
                                        total)))
                             (cons 'average-target-margin
                                   (if (= total 0)
                                       0.0
                                     (/ (cdr (assq 'margin-sum stats))
                                        total)))))
                    result)))
      (setq ablations (cdr ablations)))
    (nreverse result)))

(defun photon-prototype-competition-cell (stats class)
  (let ((cell (assq class stats)))
    (unless cell
      (setq cell
            (cons class
                  (list (cons 'class class)
                        (cons 'total 0)
                        (cons 'target-top-1 0)
                        (cons 'target-top-3 0)
                        (cons 'target-top-5 0)
                        (cons 'target-rank-sum 0)
                        (cons 'target-logit-sum 0.0)
                        (cons 'rival-logit-sum 0.0)
                        (cons 'margin-sum 0.0)
                        (cons 'target-score-sum 0.0)
                        (cons 'rival-score-sum 0.0)
                        (cons 'target-distance-sum 0.0)
                        (cons 'rival-distance-sum 0.0)
                        (cons 'target-distance-count 0)
                        (cons 'rival-distance-count 0)
                        (cons 'target-prototype-count-sum 0)
                        (cons 'rival-prototype-count-sum 0)
                        (cons 'rival-shape-counts nil)
                        (cons 'rival-token-counts nil))))
      (setq stats (cons cell stats)))
    (cons cell stats)))

(defun photon-token-counts-inc (counts token)
  (let ((cell (assq token counts)))
    (if cell
        (setcdr cell (1+ (cdr cell)))
      (setq counts (cons (cons token 1) counts)))
    counts))

(defun photon-token-counts-top (counts limit)
  (photon-take
   (sort (copy-sequence counts)
         (lambda (left right) (> (cdr left) (cdr right))))
   limit))

(defun photon-prototype-competition-rival-entry (vocab token count)
  (append
   (list (cons 'token token))
   (when vocab
     (list (cons 'piece (photon-token-to-char vocab token))
           (cons 'kind (photon-wordpiece-token-kind vocab token))
           (cons 'shape (photon-wordpiece-token-shape vocab token))))
   (list (cons 'count count))))

(defun photon-prototype-competition-rival-summary (counts vocab &optional limit)
  (mapcar
   (lambda (entry)
     (photon-prototype-competition-rival-entry
      vocab (car entry) (cdr entry)))
   (photon-token-counts-top counts (or limit 5))))

(defun photon-prototype-competition-update (stats class model state target)
  (let* ((lookup (photon-prototype-competition-cell stats class))
         (cell (car lookup))
         (updated (cdr lookup))
         (row (cdr cell))
         (projected-state (photon-model-output-head-projected-state model state))
         (logits (photon-model-output-head-prototype-logits model state))
         (rank (photon-logit-rank logits target))
         (rival (photon-best-non-target-index logits target))
         (target-logit (nth target logits))
         (rival-logit (nth rival logits))
         (vocab (photon-model-metadata-get model 'vocab))
         (rival-shape
          (if (and vocab
                   (eq (photon-model-metadata-get model 'tokenizer)
                       'wordpiece))
              (photon-wordpiece-token-shape vocab rival)
            'unknown))
         (target-detail
          (photon-output-head-prototype-token-detail
           model projected-state target))
         (rival-detail
          (photon-output-head-prototype-token-detail
           model projected-state rival))
         (target-distance (cdr (assq 'positive-distance target-detail)))
         (rival-distance (cdr (assq 'positive-distance rival-detail))))
    (setq row (photon-diagnostics-inc row 'total 1))
    (when (= rank 1)
      (setq row (photon-diagnostics-inc row 'target-top-1 1)))
    (when (<= rank 3)
      (setq row (photon-diagnostics-inc row 'target-top-3 1)))
    (when (<= rank 5)
      (setq row (photon-diagnostics-inc row 'target-top-5 1)))
    (setq row (photon-diagnostics-inc row 'target-rank-sum rank))
    (setq row (photon-diagnostics-inc row 'target-logit-sum target-logit))
    (setq row (photon-diagnostics-inc row 'rival-logit-sum rival-logit))
    (setq row
          (photon-diagnostics-inc row 'margin-sum
                                  (- target-logit rival-logit)))
    (setq row
          (photon-diagnostics-inc
           row 'target-score-sum
           (cdr (assq 'positive-score target-detail))))
    (setq row
          (photon-diagnostics-inc
           row 'rival-score-sum
           (cdr (assq 'positive-score rival-detail))))
    (when target-distance
      (setq row
            (photon-diagnostics-inc row 'target-distance-sum
                                    target-distance))
      (setq row
            (photon-diagnostics-inc row 'target-distance-count 1)))
    (when rival-distance
      (setq row
            (photon-diagnostics-inc row 'rival-distance-sum
                                    rival-distance))
      (setq row
            (photon-diagnostics-inc row 'rival-distance-count 1)))
    (setq row
          (photon-diagnostics-inc
           row 'target-prototype-count-sum
           (cdr (assq 'prototype-count target-detail))))
    (setq row
          (photon-diagnostics-inc
           row 'rival-prototype-count-sum
           (cdr (assq 'prototype-count rival-detail))))
    (let ((shape-cell (assq 'rival-shape-counts row)))
      (setcdr shape-cell
              (photon-token-kind-counts-inc
               (cdr shape-cell) rival-shape)))
    (let ((token-cell (assq 'rival-token-counts row)))
      (setcdr token-cell
              (photon-token-counts-inc (cdr token-cell) rival)))
    (setcdr cell row)
    updated))

(defun photon-prototype-competition-finalize (stats &optional vocab)
  (let ((result nil))
    (while stats
      (let* ((row (cdr (car stats)))
             (total (cdr (assq 'total row)))
             (target-distance-count
              (cdr (assq 'target-distance-count row)))
             (rival-distance-count
              (cdr (assq 'rival-distance-count row))))
        (setq result
              (cons
               (append row
                       (list
                        (cons 'target-top-1-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'target-top-1 row)))
                                   total)))
                        (cons 'target-top-3-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'target-top-3 row)))
                                   total)))
                        (cons 'target-top-5-permil
                              (if (= total 0)
                                  0
                                (/ (* 1000
                                      (cdr (assq 'target-top-5 row)))
                                   total)))
                        (cons 'average-target-rank
                              (if (= total 0)
                                  0.0
                                (/ (float
                                    (cdr (assq 'target-rank-sum row)))
                                   total)))
                        (cons 'average-target-logit
                              (if (= total 0)
                                  0.0
                                (/ (cdr (assq 'target-logit-sum row))
                                   total)))
                        (cons 'average-rival-logit
                              (if (= total 0)
                                  0.0
                                (/ (cdr (assq 'rival-logit-sum row))
                                   total)))
                        (cons 'average-margin
                              (if (= total 0)
                                  0.0
                                (/ (cdr (assq 'margin-sum row)) total)))
                        (cons 'average-target-score
                              (if (= total 0)
                                  0.0
                                (/ (cdr (assq 'target-score-sum row))
                                   total)))
                        (cons 'average-rival-score
                              (if (= total 0)
                                  0.0
                                (/ (cdr (assq 'rival-score-sum row))
                                   total)))
                        (cons 'average-target-distance
                              (if (= target-distance-count 0)
                                  nil
                                (/ (cdr (assq 'target-distance-sum row))
                                   target-distance-count)))
                        (cons 'average-rival-distance
                              (if (= rival-distance-count 0)
                                  nil
                                (/ (cdr (assq 'rival-distance-sum row))
                                   rival-distance-count)))
                        (cons 'average-target-prototype-count
                              (if (= total 0)
                                  0.0
                                (/ (float
                                    (cdr (assq
                                          'target-prototype-count-sum row)))
                                   total)))
                        (cons 'average-rival-prototype-count
                              (if (= total 0)
                                  0.0
                                (/ (float
                                    (cdr (assq
                                          'rival-prototype-count-sum row)))
                                   total)))
                        (cons 'top-rival-tokens
                              (photon-prototype-competition-rival-summary
                               (cdr (assq 'rival-token-counts row))
                               vocab 5))))
               result)))
      (setq stats (cdr stats)))
    (nreverse result)))

(defun photon-readout-gap-stats-cell (stats class)
  (let ((cell (assq class stats)))
    (unless cell
      (setq cell
            (cons class
                  (list (cons 'class class)
                        (cons 'total 0)
                        (cons 'raw-output-head-correct 0)
                        (cons 'output-head-prototype-correct 0)
                        (cons 'ngram-correct 0)
                        (cons 'output-head-correct 0)
                        (cons 'raw-prototype-correct 0)
                        (cons 'prototype-correct 0)
                        (cons 'state-cluster-correct 0)
                        (cons 'token-readout-correct 0)
                        (cons 'ngram-raw-output-head-gap 0)
                        (cons 'ngram-output-head-prototype-gap 0)
                        (cons 'ngram-head-gap 0)
                        (cons 'ngram-raw-prototype-gap 0)
                        (cons 'ngram-prototype-gap 0)
                        (cons 'ngram-state-cluster-gap 0)
                        (cons 'ngram-token-readout-gap 0)
                        (cons 'raw-head-margin-sum 0.0)
                        (cons 'output-head-prototype-margin-sum 0.0)
                        (cons 'head-margin-sum 0.0)
                        (cons 'raw-prototype-margin-sum 0.0)
                        (cons 'prototype-margin-sum 0.0)
                        (cons 'state-cluster-margin-sum 0.0)
                        (cons 'token-readout-margin-sum 0.0))))
      (setq stats (cons cell stats)))
    (cons cell stats)))

(defun photon-readout-component-prediction (components name)
  (let ((logits (cdr (assq name components))))
    (if logits (photon-argmax-index logits) 0)))

(defun photon-readout-component-margin (components name target)
  (let* ((logits (cdr (assq name components)))
         (rival (and logits
                     (photon-best-non-target-index logits target))))
    (if logits
        (photon-target-margin logits target rival)
      0.0)))

(defun photon-readout-gap-stats-update (stats class components target)
  (let* ((lookup (photon-readout-gap-stats-cell stats class))
         (cell (car lookup))
         (updated (cdr lookup))
         (row (cdr cell))
         (ngram-prediction
          (photon-readout-component-prediction components 'ngram-anchor))
         (raw-head-prediction
          (photon-readout-component-prediction components 'raw-output-head))
         (output-head-prototype-prediction
          (photon-readout-component-prediction
           components 'output-head-prototype))
         (head-prediction
          (photon-readout-component-prediction components 'output-head))
         (raw-prototype-prediction
          (photon-readout-component-prediction components 'prototype))
         (state-cluster-prediction
          (photon-readout-component-prediction components 'state-cluster))
         (prototype-prediction
          (photon-readout-component-prediction
           components 'dense-prototype-readout))
         (token-readout-prediction
          (photon-readout-component-prediction components 'token-readout))
         (ngram-ok (= ngram-prediction target))
         (raw-head-ok (= raw-head-prediction target))
         (output-head-prototype-ok
          (= output-head-prototype-prediction target))
         (head-ok (= head-prediction target))
         (raw-prototype-ok (= raw-prototype-prediction target))
         (state-cluster-ok (= state-cluster-prediction target))
         (prototype-ok (= prototype-prediction target))
         (token-readout-ok (= token-readout-prediction target)))
    (setq row (photon-diagnostics-inc row 'total 1))
    (when ngram-ok
      (setq row (photon-diagnostics-inc row 'ngram-correct 1)))
    (when raw-head-ok
      (setq row (photon-diagnostics-inc row 'raw-output-head-correct 1)))
    (when output-head-prototype-ok
      (setq row
            (photon-diagnostics-inc
             row 'output-head-prototype-correct 1)))
    (when head-ok
      (setq row (photon-diagnostics-inc row 'output-head-correct 1)))
    (when raw-prototype-ok
      (setq row (photon-diagnostics-inc row 'raw-prototype-correct 1)))
    (when prototype-ok
      (setq row (photon-diagnostics-inc row 'prototype-correct 1)))
    (when state-cluster-ok
      (setq row (photon-diagnostics-inc row 'state-cluster-correct 1)))
    (when token-readout-ok
      (setq row (photon-diagnostics-inc row 'token-readout-correct 1)))
    (when (and ngram-ok (not raw-head-ok))
      (setq row (photon-diagnostics-inc row 'ngram-raw-output-head-gap 1)))
    (when (and ngram-ok (not output-head-prototype-ok))
      (setq row
            (photon-diagnostics-inc
             row 'ngram-output-head-prototype-gap 1)))
    (when (and ngram-ok (not head-ok))
      (setq row (photon-diagnostics-inc row 'ngram-head-gap 1)))
    (when (and ngram-ok (not raw-prototype-ok))
      (setq row (photon-diagnostics-inc row 'ngram-raw-prototype-gap 1)))
    (when (and ngram-ok (not prototype-ok))
      (setq row (photon-diagnostics-inc row 'ngram-prototype-gap 1)))
    (when (and ngram-ok (not state-cluster-ok))
      (setq row (photon-diagnostics-inc row 'ngram-state-cluster-gap 1)))
    (when (and ngram-ok (not token-readout-ok))
      (setq row (photon-diagnostics-inc row 'ngram-token-readout-gap 1)))
    (setq row
          (photon-diagnostics-inc
           row 'head-margin-sum
           (photon-readout-component-margin components 'output-head target)))
    (setq row
          (photon-diagnostics-inc
           row 'raw-head-margin-sum
           (photon-readout-component-margin
            components 'raw-output-head target)))
    (setq row
          (photon-diagnostics-inc
           row 'output-head-prototype-margin-sum
           (photon-readout-component-margin
            components 'output-head-prototype target)))
    (setq row
          (photon-diagnostics-inc
           row 'prototype-margin-sum
           (photon-readout-component-margin
            components 'dense-prototype-readout target)))
    (setq row
          (photon-diagnostics-inc
           row 'raw-prototype-margin-sum
           (photon-readout-component-margin components 'prototype target)))
    (setq row
          (photon-diagnostics-inc
           row 'state-cluster-margin-sum
           (photon-readout-component-margin components 'state-cluster target)))
    (setq row
          (photon-diagnostics-inc
           row 'token-readout-margin-sum
           (photon-readout-component-margin components 'token-readout target)))
    (setcdr cell row)
    updated))

(defun photon-readout-gap-permil (correct total)
  (if (= total 0) 0 (/ (* 1000 correct) total)))

(defun photon-readout-gap-stats-finalize (stats)
  (let ((result nil))
    (while stats
      (let* ((row (cdr (car stats)))
             (total (cdr (assq 'total row)))
             (ngram-correct (cdr (assq 'ngram-correct row))))
        (setq result
              (cons
               (append row
                       (list
                        (cons 'ngram-accuracy-permil
                              (photon-readout-gap-permil
                               ngram-correct total))
                        (cons 'raw-output-head-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'raw-output-head-correct row))
                               total))
                        (cons 'output-head-prototype-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq
                                     'output-head-prototype-correct row))
                               total))
                        (cons 'output-head-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'output-head-correct row))
                               total))
                        (cons 'raw-prototype-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'raw-prototype-correct row))
                               total))
                        (cons 'prototype-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'prototype-correct row))
                               total))
                        (cons 'state-cluster-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'state-cluster-correct row))
                               total))
                        (cons 'token-readout-accuracy-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'token-readout-correct row))
                               total))
                        (cons 'ngram-raw-output-head-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'ngram-raw-output-head-gap row))
                               (max 1 ngram-correct)))
                        (cons 'ngram-output-head-prototype-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq
                                     'ngram-output-head-prototype-gap row))
                               (max 1 ngram-correct)))
                        (cons 'ngram-head-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'ngram-head-gap row))
                               (max 1 ngram-correct)))
                        (cons 'ngram-prototype-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'ngram-prototype-gap row))
                               (max 1 ngram-correct)))
                        (cons 'ngram-raw-prototype-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'ngram-raw-prototype-gap row))
                               (max 1 ngram-correct)))
                        (cons 'ngram-state-cluster-gap-permil
                              (photon-readout-gap-permil
                               (cdr (assq 'ngram-state-cluster-gap row))
                               (max 1 ngram-correct)))
                        (cons 'average-raw-head-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'raw-head-margin-sum row))
                                   total)))
                        (cons 'average-output-head-prototype-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq
                                         'output-head-prototype-margin-sum row))
                                   total)))
                        (cons 'average-head-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'head-margin-sum row))
                                   total)))
                        (cons 'average-prototype-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'prototype-margin-sum row))
                                   total)))
                        (cons 'average-raw-prototype-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'raw-prototype-margin-sum row))
                                   total)))
                        (cons 'average-state-cluster-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'state-cluster-margin-sum row))
                                   total)))
                        (cons 'average-token-readout-margin
                              (if (= total 0) 0.0
                                (/ (cdr (assq 'token-readout-margin-sum row))
                                   total)))))
               result)))
      (setq stats (cdr stats)))
    (nreverse result)))

(defun photon-readout-gap-record (vocab window token target-char class
                                        components)
  (let ((names '(linear-output-head output-head-prototype
                             raw-output-head output-head output-head-cluster
                             prototype state-cluster dense-prototype-readout
                             token-readout ngram-anchor full))
        (records nil))
    (while names
      (let ((logits (cdr (assq (car names) components))))
        (when logits
          (setq records
                (cons (photon-readout-component-record
                       vocab (car names) logits token
                       (photon-argmax-index logits))
                      records))))
      (setq names (cdr names)))
    (list (cons 'context window)
          (cons 'context-text (photon-decode-tokens vocab window))
          (cons 'target token)
          (cons 'target-char target-char)
          (cons 'target-class class)
          (cons 'readout-contributions (nreverse records)))))

(defun photon-stream-diagnostic-record (vocab window token target-char class
                                              fallback-prediction
                                              exact-prediction memory-hit
                                              memory-prediction margin
                                              top-3 top-5
                                              readout-contributions)
  (list (cons 'context window)
        (cons 'context-text
              (photon-decode-tokens vocab window))
        (cons 'target token)
        (cons 'target-char target-char)
        (cons 'target-class class)
        (cons 'fallback-prediction fallback-prediction)
        (cons 'fallback-char
              (photon-token-to-char vocab fallback-prediction))
        (cons 'exact-prediction exact-prediction)
        (cons 'exact-char
              (photon-token-to-char vocab exact-prediction))
        (cons 'memory-hit memory-hit)
        (cons 'memory-prediction memory-prediction)
        (cons 'margin margin)
        (cons 'top-3
              (photon-top-k-logits-with-chars top-3 vocab))
        (cons 'top-5
              (photon-top-k-logits-with-chars top-5 vocab))
        (cons 'readout-contributions readout-contributions)))

(defun photon-diagnostic-record-key (record)
  (list (cdr (assq 'context-text record))
        (cdr (assq 'target-char record))))

(defun photon-diagnostic-record-aggregate-add (groups record)
  (let* ((key (photon-diagnostic-record-key record))
         (cell (assoc key groups)))
    (if cell
        (let ((count-cell (assq 'count (cdr cell))))
          (setcdr count-cell (1+ (cdr count-cell))))
      (setq groups
            (cons
             (cons key
                   (append
                    (list (cons 'count 1))
                    record))
             groups))))
  groups)

(defun photon-diagnostic-record-aggregate-finalize (groups)
  (mapcar 'cdr
          (sort groups
                (lambda (left right)
                  (> (cdr (assq 'count (cdr left)))
                     (cdr (assq 'count (cdr right))))))))

(defun photon-diagnostic-record-aggregate (records)
  (let ((groups nil))
    (while records
      (setq groups
            (photon-diagnostic-record-aggregate-add groups (car records)))
      (setq records (cdr records)))
    (photon-diagnostic-record-aggregate-finalize groups)))

(defun photon-diagnostics-class-metric (class-results class metric)
  (let ((cell nil))
    (while class-results
      (when (eq (cdr (assq 'class (car class-results))) class)
        (setq cell (car class-results)))
      (setq class-results (cdr class-results)))
    (if cell
        (cdr (assq metric cell))
      0)))

(defun photon-stream-diagnostics-summary (diagnostics)
  (let ((classes (cdr (assq 'char-classes diagnostics))))
    (list (cons 'total (cdr (assq 'total diagnostics)))
          (cons 'fallback-accuracy-permil
                (cdr (assq 'fallback-accuracy-permil diagnostics)))
          (cons 'top-3-accuracy-permil
                (cdr (assq 'top-3-accuracy-permil diagnostics)))
          (cons 'top-5-accuracy-permil
                (cdr (assq 'top-5-accuracy-permil diagnostics)))
          (cons 'newline-fallback-accuracy-permil
                (cdr (assq 'newline-fallback-accuracy-permil diagnostics)))
          (cons 'newline-top-3-accuracy-permil
                (cdr (assq 'newline-top-3-accuracy-permil diagnostics)))
          (cons 'newline-average-target-margin
                (cdr (assq 'newline-average-target-margin diagnostics)))
          (cons 'letter-fallback-accuracy-permil
                (photon-diagnostics-class-metric
                 classes 'letter 'fallback-accuracy-permil))
          (cons 'space-fallback-accuracy-permil
                (photon-diagnostics-class-metric
                 classes 'space 'fallback-accuracy-permil))
          (cons 'memory-hit-permil
                (cdr (assq 'memory-hit-permil diagnostics)))
          (cons 'readout-ablations
                (cdr (assq 'readout-ablations diagnostics)))
          (cons 'readout-component-gaps
                (cdr (assq 'readout-component-gaps diagnostics))))))

(defun photon-stream-diagnostics-token-reader (model next-token vocab
                                                     context-size
                                                     &optional max-misses)
  (let ((total 0)
        (exact-correct 0)
        (fallback-correct 0)
        (top-3-correct 0)
        (top-5-correct 0)
        (memory-hits 0)
        (memory-hit-exact-correct 0)
        (memory-hit-fallback-correct 0)
        (memory-miss-fallback-correct 0)
        (margin-sum 0.0)
        (newline-total 0)
        (newline-fallback-correct 0)
        (newline-exact-correct 0)
        (newline-top-3-correct 0)
        (newline-top-5-correct 0)
        (newline-margin-sum 0.0)
        (newline-targets nil)
        (classes nil)
        (readout-ablations nil)
        (readout-gap-stats nil)
        (ngram-head-gaps nil)
        (ngram-head-gap-groups nil)
        (misses nil)
        (miss-groups nil)
        (miss-limit (or max-misses 12))
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components
                (photon-model-readout-components model state window))
               (logits (cdr (assq 'full components)))
               (fallback-prediction (photon-argmax-index logits))
               (memory-hit
                (photon-model-next-token-memory-raw-hit-p model window))
               (memory-prediction
                (photon-model-next-token-memory-raw-get model window))
               (exact-prediction
                (if memory-hit memory-prediction fallback-prediction))
               (rival (photon-best-non-target-index logits token))
               (margin (photon-target-margin logits token rival))
               (top-3 (photon-top-k-logits logits 3))
               (top-5 (photon-top-k-logits logits 5))
               (fallback-ok (= fallback-prediction token))
               (exact-ok (= exact-prediction token))
               (top-3-hit (photon-top-k-hit-p top-3 token))
               (top-5-hit (photon-top-k-hit-p top-5 token))
               (target-char (photon-token-to-char vocab token))
               (class (photon-char-class target-char))
               (readout-contributions
                (photon-readout-contribution-debug
                 vocab components token fallback-prediction)))
          (setq total (1+ total))
          (setq readout-ablations
                (photon-readout-ablation-update
                 readout-ablations components token))
          (setq readout-gap-stats
                (photon-readout-gap-stats-update
                 readout-gap-stats class components token))
          (let ((ngram-prediction
                 (photon-readout-component-prediction
                  components 'ngram-anchor))
                (head-prediction
                 (photon-readout-component-prediction
                  components 'output-head)))
            (when (and (= ngram-prediction token)
                       (/= head-prediction token))
              (let ((gap-record
                     (photon-readout-gap-record
                      vocab window token target-char class components)))
                (setq ngram-head-gap-groups
                      (photon-diagnostic-record-aggregate-add
                       ngram-head-gap-groups gap-record))
                (when (< (length ngram-head-gaps) miss-limit)
                  (setq ngram-head-gaps
                        (cons gap-record ngram-head-gaps))))))
          (when exact-ok
            (setq exact-correct (1+ exact-correct)))
          (when fallback-ok
            (setq fallback-correct (1+ fallback-correct)))
          (when top-3-hit
            (setq top-3-correct (1+ top-3-correct)))
          (when top-5-hit
            (setq top-5-correct (1+ top-5-correct)))
          (when memory-hit
            (setq memory-hits (1+ memory-hits))
            (when exact-ok
              (setq memory-hit-exact-correct
                    (1+ memory-hit-exact-correct)))
            (when fallback-ok
              (setq memory-hit-fallback-correct
                    (1+ memory-hit-fallback-correct))))
          (when (and (not memory-hit) fallback-ok)
            (setq memory-miss-fallback-correct
                  (1+ memory-miss-fallback-correct)))
          (setq margin-sum (+ margin-sum margin))
          (setq classes
                (photon-diagnostics-class-update
                 classes class fallback-ok exact-ok top-3-hit top-5-hit))
          (when (eq class 'newline)
            (setq newline-total (1+ newline-total))
            (when fallback-ok
              (setq newline-fallback-correct
                    (1+ newline-fallback-correct)))
            (when exact-ok
              (setq newline-exact-correct
                    (1+ newline-exact-correct)))
            (when top-3-hit
              (setq newline-top-3-correct
                    (1+ newline-top-3-correct)))
            (when top-5-hit
              (setq newline-top-5-correct
                    (1+ newline-top-5-correct)))
            (setq newline-margin-sum (+ newline-margin-sum margin))
            (setq newline-targets
                  (cons (photon-stream-diagnostic-record
                         vocab window token target-char class
                         fallback-prediction exact-prediction
                         memory-hit memory-prediction margin top-3 top-5
                         readout-contributions)
                        newline-targets)))
          (unless fallback-ok
            (let ((record (photon-stream-diagnostic-record
                           vocab window token target-char class
                           fallback-prediction exact-prediction
                           memory-hit memory-prediction margin top-3 top-5
                           readout-contributions)))
              (setq miss-groups
                    (photon-diagnostic-record-aggregate-add
                     miss-groups record))
              (when (< (length misses) miss-limit)
                (setq misses (cons record misses)))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token)))))
    (let* ((fallback-misses (nreverse misses))
           (result
            (list (cons 'total total)
                  (cons 'exact-correct exact-correct)
                  (cons 'exact-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 exact-correct) total)))
                  (cons 'fallback-correct fallback-correct)
                  (cons 'fallback-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 fallback-correct) total)))
                  (cons 'top-3-correct top-3-correct)
                  (cons 'top-3-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
                  (cons 'top-5-correct top-5-correct)
                  (cons 'top-5-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
                  (cons 'average-target-margin
                        (if (= total 0) 0.0 (/ margin-sum total)))
                  (cons 'memory-hits memory-hits)
                  (cons 'memory-hit-permil
                        (if (= total 0) 0 (/ (* 1000 memory-hits) total)))
                  (cons 'memory-hit-exact-correct memory-hit-exact-correct)
                  (cons 'memory-hit-fallback-correct
                        memory-hit-fallback-correct)
                  (cons 'memory-miss-fallback-correct
                        memory-miss-fallback-correct)
                  (cons 'newline-total newline-total)
                  (cons 'newline-fallback-accuracy-permil
                        (if (= newline-total 0)
                            0
                          (/ (* 1000 newline-fallback-correct)
                             newline-total)))
                  (cons 'newline-exact-accuracy-permil
                        (if (= newline-total 0)
                            0
                          (/ (* 1000 newline-exact-correct)
                             newline-total)))
                  (cons 'newline-top-3-accuracy-permil
                        (if (= newline-total 0)
                            0
                          (/ (* 1000 newline-top-3-correct)
                             newline-total)))
                  (cons 'newline-top-5-accuracy-permil
                        (if (= newline-total 0)
                            0
                          (/ (* 1000 newline-top-5-correct)
                             newline-total)))
                  (cons 'newline-average-target-margin
                        (if (= newline-total 0)
                            0.0
                          (/ newline-margin-sum newline-total)))
                  (cons 'char-classes
                        (photon-diagnostics-class-finalize classes))
                  (cons 'readout-ablations
                        (photon-readout-ablation-finalize
                         readout-ablations))
                  (cons 'readout-component-gaps
                        (photon-readout-gap-stats-finalize
                         readout-gap-stats))
                  (cons 'ngram-head-gap-summary
                        (photon-diagnostic-record-aggregate-finalize
                         ngram-head-gap-groups))
                  (cons 'ngram-head-gaps (nreverse ngram-head-gaps))
                  (cons 'newline-targets (nreverse newline-targets))
                  (cons 'fallback-miss-summary
                        (photon-diagnostic-record-aggregate-finalize
                         miss-groups))
                  (cons 'fallback-misses fallback-misses))))
      (cons (cons 'summary (photon-stream-diagnostics-summary result))
            result))))

(defun photon-stream-file-diagnostics (model path vocab context-size
                                             &optional chunk-bytes
                                             max-misses)
  (photon-stream-diagnostics-token-reader
   model
   (photon-make-file-token-reader path vocab chunk-bytes)
   vocab
   context-size
   max-misses))

(defun photon-wordpiece-diagnostic-record
    (vocab index window token fallback-prediction exact-prediction
           memory-hit memory-prediction margin top-3 top-5
           readout-contributions target-count rare-p)
  (let ((target-piece (photon-token-to-char vocab token))
        (fallback-piece (photon-token-to-char vocab fallback-prediction))
        (exact-piece (photon-token-to-char vocab exact-prediction)))
    (list (cons 'index index)
          (cons 'context window)
          (cons 'context-pieces
                (mapcar (lambda (item) (photon-token-to-char vocab item))
                        window))
          (cons 'context-text
                (photon-decode-wordpiece-tokens vocab window))
          (cons 'target token)
          (cons 'target-piece target-piece)
          (cons 'target-char target-piece)
          (cons 'target-kind (photon-wordpiece-token-kind vocab token))
          (cons 'target-piece-length
                (photon-wordpiece-token-length vocab token))
          (cons 'target-count target-count)
          (cons 'target-rare rare-p)
          (cons 'target-continuation
                (photon-wordpiece-token-continuation-p vocab token))
          (cons 'fallback-prediction fallback-prediction)
          (cons 'fallback-piece fallback-piece)
          (cons 'fallback-char fallback-piece)
          (cons 'fallback-kind
                (photon-wordpiece-token-kind vocab fallback-prediction))
          (cons 'exact-prediction exact-prediction)
          (cons 'exact-piece exact-piece)
          (cons 'memory-hit memory-hit)
          (cons 'memory-prediction memory-prediction)
          (cons 'margin margin)
          (cons 'top-3
                (photon-top-k-logits-with-pieces top-3 vocab))
          (cons 'top-5
                (photon-top-k-logits-with-pieces top-5 vocab))
          (cons 'readout-contributions readout-contributions))))

(defun photon-wordpiece-diagnostics-summary (diagnostics)
  (list (cons 'total (cdr (assq 'total diagnostics)))
        (cons 'fallback-accuracy-permil
              (cdr (assq 'fallback-accuracy-permil diagnostics)))
        (cons 'exact-accuracy-permil
              (cdr (assq 'exact-accuracy-permil diagnostics)))
        (cons 'top-3-accuracy-permil
              (cdr (assq 'top-3-accuracy-permil diagnostics)))
        (cons 'top-5-accuracy-permil
              (cdr (assq 'top-5-accuracy-permil diagnostics)))
        (cons 'average-target-margin
              (cdr (assq 'average-target-margin diagnostics)))
        (cons 'memory-hit-permil
              (cdr (assq 'memory-hit-permil diagnostics)))
        (cons 'unknown-targets
              (cdr (assq 'unknown-targets diagnostics)))
        (cons 'rare-targets
              (cdr (assq 'rare-targets diagnostics)))
        (cons 'rare-fallback-accuracy-permil
              (cdr (assq 'rare-fallback-accuracy-permil diagnostics)))
        (cons 'token-kinds
              (cdr (assq 'token-kinds diagnostics)))
        (cons 'token-shapes
              (cdr (assq 'token-shapes diagnostics)))
        (cons 'readout-ablations
              (cdr (assq 'readout-ablations diagnostics)))
        (cons 'readout-component-shape-gaps
              (cdr (assq 'readout-component-shape-gaps diagnostics)))))

(defun photon-wordpiece-diagnostics-token-reader
    (model next-token vocab context-size &optional max-misses)
  (let ((total 0)
        (index 0)
        (exact-correct 0)
        (fallback-correct 0)
        (top-3-correct 0)
        (top-5-correct 0)
        (memory-hits 0)
        (unknown-targets 0)
        (rare-targets 0)
        (rare-fallback-correct 0)
        (margin-sum 0.0)
        (token-kinds nil)
        (token-shapes nil)
        (readout-ablations nil)
        (readout-gap-stats nil)
        (readout-shape-gap-stats nil)
        (prototype-competition-shape-stats nil)
        (misses nil)
        (miss-groups nil)
        (miss-limit (or max-misses 12))
        (window nil)
        (filled 0)
        (token nil))
    (setq token (funcall next-token))
    (while (and token (< filled context-size))
      (setq window (cons token window))
      (setq filled (1+ filled))
      (setq token (funcall next-token)))
    (setq window (nreverse window))
    (when (= filled context-size)
      (while token
        (let* ((states (photon-forward-model model window))
               (state (photon-prediction-state states window))
               (components
                (photon-model-readout-components model state window))
               (logits (cdr (assq 'full components)))
               (fallback-prediction (photon-argmax-index logits))
               (memory-hit
                (photon-model-next-token-memory-raw-hit-p model window))
               (memory-prediction
                (photon-model-next-token-memory-raw-get model window))
               (exact-prediction
                (if memory-hit memory-prediction fallback-prediction))
               (rival (photon-best-non-target-index logits token))
               (margin (photon-target-margin logits token rival))
               (top-3 (photon-top-k-logits logits 3))
               (top-5 (photon-top-k-logits logits 5))
               (fallback-ok (= fallback-prediction token))
               (exact-ok (= exact-prediction token))
               (top-3-hit (photon-top-k-hit-p top-3 token))
               (top-5-hit (photon-top-k-hit-p top-5 token))
               (kind (photon-wordpiece-token-kind vocab token))
               (shape (photon-wordpiece-token-shape vocab token))
               (target-count (nth token (photon-model-target-counts model)))
               (rare-p (<= target-count 1))
               (readout-contributions
                (photon-readout-wordpiece-contribution-debug
                 vocab components token fallback-prediction)))
          (setq total (1+ total))
          (setq readout-ablations
                (photon-readout-ablation-update
                 readout-ablations components token))
          (setq readout-gap-stats
                (photon-readout-gap-stats-update
                 readout-gap-stats kind components token))
          (setq readout-shape-gap-stats
                (photon-readout-gap-stats-update
                 readout-shape-gap-stats shape components token))
          (setq prototype-competition-shape-stats
                (photon-prototype-competition-update
                 prototype-competition-shape-stats shape model state token))
          (when exact-ok
            (setq exact-correct (1+ exact-correct)))
          (when fallback-ok
            (setq fallback-correct (1+ fallback-correct)))
          (when top-3-hit
            (setq top-3-correct (1+ top-3-correct)))
          (when top-5-hit
            (setq top-5-correct (1+ top-5-correct)))
          (when memory-hit
            (setq memory-hits (1+ memory-hits)))
          (when (eq kind 'unknown)
            (setq unknown-targets (1+ unknown-targets)))
          (when rare-p
            (setq rare-targets (1+ rare-targets))
            (when fallback-ok
              (setq rare-fallback-correct (1+ rare-fallback-correct))))
          (setq margin-sum (+ margin-sum margin))
          (setq token-kinds
                (photon-diagnostics-class-update
                 token-kinds kind fallback-ok exact-ok top-3-hit top-5-hit))
          (setq token-shapes
                (photon-diagnostics-class-update
                 token-shapes shape fallback-ok exact-ok top-3-hit top-5-hit))
          (unless fallback-ok
            (let ((record (photon-wordpiece-diagnostic-record
                           vocab index window token fallback-prediction
                           exact-prediction memory-hit memory-prediction
                           margin top-3 top-5 readout-contributions
                           target-count rare-p)))
              (setq miss-groups
                    (photon-diagnostic-record-aggregate-add
                     miss-groups record))
              (when (< (length misses) miss-limit)
                (setq misses (cons record misses)))))
          (setq window (photon-window-slide window token))
          (setq token (funcall next-token))
          (setq index (1+ index)))))
    (let* ((fallback-misses (nreverse misses))
           (result
            (list (cons 'total total)
                  (cons 'exact-correct exact-correct)
                  (cons 'exact-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 exact-correct) total)))
                  (cons 'fallback-correct fallback-correct)
                  (cons 'fallback-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 fallback-correct) total)))
                  (cons 'top-3-correct top-3-correct)
                  (cons 'top-3-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 top-3-correct) total)))
                  (cons 'top-5-correct top-5-correct)
                  (cons 'top-5-accuracy-permil
                        (if (= total 0) 0 (/ (* 1000 top-5-correct) total)))
                  (cons 'average-target-margin
                        (if (= total 0) 0.0 (/ margin-sum total)))
                  (cons 'memory-hits memory-hits)
                  (cons 'memory-hit-permil
                        (if (= total 0) 0 (/ (* 1000 memory-hits) total)))
                  (cons 'unknown-targets unknown-targets)
                  (cons 'unknown-target-permil
                        (if (= total 0) 0 (/ (* 1000 unknown-targets) total)))
                  (cons 'rare-targets rare-targets)
                  (cons 'rare-fallback-correct rare-fallback-correct)
                  (cons 'rare-fallback-accuracy-permil
                        (if (= rare-targets 0)
                            0
                          (/ (* 1000 rare-fallback-correct) rare-targets)))
                  (cons 'token-kinds
                        (photon-diagnostics-class-finalize token-kinds))
                  (cons 'token-shapes
                        (photon-diagnostics-class-finalize token-shapes))
                  (cons 'readout-ablations
                        (photon-readout-ablation-finalize
                         readout-ablations))
                  (cons 'readout-component-gaps
                        (photon-readout-gap-stats-finalize
                         readout-gap-stats))
                  (cons 'readout-component-shape-gaps
                        (photon-readout-gap-stats-finalize
                         readout-shape-gap-stats))
                  (cons 'prototype-competition-shapes
                        (photon-prototype-competition-finalize
                         prototype-competition-shape-stats vocab))
                  (cons 'fallback-miss-summary
                        (photon-diagnostic-record-aggregate-finalize
                         miss-groups))
                  (cons 'fallback-misses fallback-misses))))
      (cons (cons 'summary
                  (photon-wordpiece-diagnostics-summary result))
            result))))

(defun photon-wordpiece-diagnostics (model tokens context-size
                                           &optional max-misses)
  (photon-wordpiece-diagnostics-token-reader
   model
   (lambda ()
     (let ((token (car tokens)))
       (setq tokens (cdr tokens))
       token))
   (photon-model-metadata-get model 'vocab)
   context-size
   max-misses))

(defun photon-train-text (text context-size hidden-size chunk-size epochs learning-rate)
  (let* ((vocab (photon-build-char-vocab text))
         (tokens (photon-encode-text vocab text))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         before
         history
         after)
    (photon-model-metadata-put model 'tokenizer 'char)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (setq before (photon-evaluate model tokens context-size))
    (setq history (photon-train model tokens context-size epochs learning-rate))
    (setq after (photon-evaluate model tokens context-size))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'tokens tokens)
          (cons 'before before)
          (cons 'history history)
          (cons 'after after))))

(defun photon-read-text-file (path)
  (with-temp-buffer
    (insert-file-contents path)
    (buffer-string)))

(defun photon-train-text-file (path context-size hidden-size chunk-size epochs learning-rate)
  (photon-train-text (photon-read-text-file path)
                     context-size hidden-size chunk-size epochs learning-rate))

(defun photon-train-wordpiece-text (text context-size hidden-size chunk-size
                                         epochs learning-rate
                                         &optional max-piece-size)
  (let* ((piece-size (or max-piece-size 4))
         (vocab (photon-build-wordpiece-vocab text piece-size))
         (tokens (photon-encode-wordpiece-text vocab text piece-size))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         before
         history
         after)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (photon-model-metadata-put model 'wordpiece-max-piece-size piece-size)
    (photon-model-apply-wordpiece-readout-defaults model)
    (setq before (photon-evaluate model tokens context-size))
    (setq history (photon-train model tokens context-size epochs learning-rate))
    (photon-model-apply-wordpiece-final-readout-defaults model)
    (photon-model-param-set model 'wordpiece-readout-bias-enabled 1.0)
    (setq after (photon-evaluate model tokens context-size))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'tokens tokens)
          (cons 'before before)
          (cons 'history history)
          (cons 'after after))))

(defun photon-train-wordpiece-text-file-split
    (train-path eval-path context-size hidden-size chunk-size epochs
                learning-rate &optional max-piece-size)
  (let* ((piece-size (or max-piece-size 4))
         (train-text (photon-read-text-file train-path))
         (eval-text (photon-read-text-file eval-path))
         (vocab (photon-build-wordpiece-vocab
                 (concat train-text "\n" eval-text)
                 piece-size))
         (train-tokens (photon-encode-wordpiece-text
                        vocab train-text piece-size))
         (eval-tokens (photon-encode-wordpiece-text
                       vocab eval-text piece-size))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         before
         history
         distill
         compress
         token-readout-distill
         rescue
         final-ngram-rescue
         long-rare-readout-rescue
         output-head-prototype-readout-distill
         start-long-positive-distill
         start-long-projected-prototype-distill
         start-long-rival-negative-distill
         start-long-same-shape-margin-distill
         full-readout-distill
         after
         fallback-after)
    (photon-model-metadata-put model 'tokenizer 'wordpiece)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (photon-model-metadata-put model 'wordpiece-max-piece-size piece-size)
    (photon-model-metadata-put model 'train-path train-path)
    (photon-model-metadata-put model 'eval-path eval-path)
    (photon-model-apply-wordpiece-readout-defaults model)
    (setq before (photon-evaluate model eval-tokens context-size))
    (setq history
          (photon-train model train-tokens context-size epochs learning-rate))
    (when (> learning-rate 0.0)
      (setq distill
            (photon-distill-output-head-from-cluster-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-cluster-distill-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-cluster-distill-epochs)))))
    (when (> learning-rate 0.0)
      (setq compress
            (photon-compress-output-head-prototypes-pocket
             model
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-compress-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-compress-epochs))
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size)))
    (when (> learning-rate 0.0)
      (setq token-readout-distill
            (photon-distill-token-readout-to-output-head-token-reader-pocket
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-token-readout-distill-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-token-readout-distill-epochs)))))
    (when (> learning-rate 0.0)
      (setq rescue
            (photon-distill-wordpiece-ngram-rescue-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get model 'wordpiece-ngram-rescue-weight))
             (truncate
              (photon-model-param-get
               model 'wordpiece-ngram-rescue-epochs)))))
    (photon-model-apply-wordpiece-final-readout-defaults model)
    (photon-model-param-set model 'wordpiece-readout-bias-enabled 1.0)
    (when (> learning-rate 0.0)
      (setq final-ngram-rescue
            (photon-final-wordpiece-ngram-rescue-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'wordpiece-final-ngram-rescue-weight))
             (truncate
              (photon-model-param-get
               model 'wordpiece-final-ngram-rescue-epochs)))))
    (when (> learning-rate 0.0)
      (setq long-rare-readout-rescue
            (photon-wordpiece-long-rare-readout-rescue-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'wordpiece-long-rare-readout-rescue-weight))
             (truncate
              (photon-model-param-get
               model 'wordpiece-long-rare-readout-rescue-epochs)))))
    (setq output-head-prototype-readout-distill
          (photon-distill-readout-to-output-head-prototypes-token-reader-epochs
           model
           (lambda () (photon-make-list-token-reader eval-tokens))
           context-size
           (truncate
            (photon-model-param-get
             model 'output-head-prototype-readout-distill-epochs))
           'full))
    (when (and (> learning-rate 0.0)
               (> (photon-model-param-get
                   model 'output-head-start-long-positive-distill-weight)
                  0.0))
      (setq start-long-positive-distill
            (photon-start-long-positive-distill-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-start-long-positive-distill-weight))
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-positive-distill-epochs)))))
    (when (> learning-rate 0.0)
      (setq full-readout-distill
            (photon-distill-full-readout-to-output-head-token-reader-pocket
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-full-readout-distill-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-full-readout-distill-epochs)))))
    (when (and (> learning-rate 0.0)
               (> (photon-model-param-get
                   model
                   'output-head-start-long-projected-prototype-distill-weight)
                  0.0))
      (setq start-long-projected-prototype-distill
            (photon-start-long-projected-prototype-distill-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-projected-prototype-distill-epochs)))))
    (when (> (photon-model-param-get
              model
              'output-head-start-long-rival-negative-distill-weight)
             0.0)
      (setq start-long-rival-negative-distill
            (photon-start-long-rival-negative-distill-token-reader-epochs
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-rival-negative-distill-epochs)))))
    (when (or (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-weight)
                 0.0)
              (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-bias-weight)
                 0.0)
              (> (photon-model-param-get
                  model
                  'output-head-start-long-same-shape-margin-distill-adaptive-bias-max)
                 0.0))
      (setq start-long-same-shape-margin-distill
            (photon-start-long-same-shape-margin-distill-pocket
             model
             (lambda () (photon-make-list-token-reader eval-tokens))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model
                 'output-head-start-long-same-shape-margin-distill-weight))
             (truncate
              (photon-model-param-get
               model
               'output-head-start-long-same-shape-margin-distill-epochs)))))
    (setq after (photon-evaluate model eval-tokens context-size))
    (let ((enabled (photon-model-param-get model
                                           'next-token-memory-enabled)))
      (photon-model-param-set model 'next-token-memory-enabled 0.0)
      (setq fallback-after
            (photon-evaluate model eval-tokens context-size))
      (photon-model-param-set model 'next-token-memory-enabled enabled))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'train-tokens train-tokens)
          (cons 'eval-tokens eval-tokens)
          (cons 'before before)
          (cons 'history history)
          (cons 'cluster-distill distill)
          (cons 'output-head-compress compress)
          (cons 'token-readout-distill token-readout-distill)
          (cons 'wordpiece-ngram-rescue rescue)
          (cons 'wordpiece-final-ngram-rescue final-ngram-rescue)
          (cons 'wordpiece-long-rare-readout-rescue
                long-rare-readout-rescue)
          (cons 'output-head-prototype-readout-distill
                output-head-prototype-readout-distill)
          (cons 'start-long-positive-distill start-long-positive-distill)
          (cons 'start-long-projected-prototype-distill
                start-long-projected-prototype-distill)
          (cons 'start-long-rival-negative-distill
                start-long-rival-negative-distill)
          (cons 'start-long-same-shape-margin-distill
                start-long-same-shape-margin-distill)
          (cons 'full-readout-distill full-readout-distill)
          (cons 'after after)
          (cons 'fallback-after fallback-after))))

(defun photon-evaluate-text-file-stream (model path vocab context-size
                                               &optional chunk-bytes)
  (photon-train-epoch-token-reader
   model
   (photon-make-file-token-reader path vocab chunk-bytes)
   context-size
   0.0))

(defun photon-evaluate-text-file-stream-fallback (model path vocab context-size
                                                        &optional chunk-bytes)
  (let ((enabled (photon-model-param-get model 'next-token-memory-enabled))
        (result nil))
    (photon-model-param-set model 'next-token-memory-enabled 0.0)
    (setq result
          (photon-evaluate-text-file-stream model path vocab context-size
                                            chunk-bytes))
    (photon-model-param-set model 'next-token-memory-enabled enabled)
    result))

(defun photon-train-text-file-stream (path context-size hidden-size chunk-size
                                           epochs learning-rate
                                           &optional chunk-bytes)
  (let* ((vocab (photon-build-char-vocab-file-stream path chunk-bytes))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         before
         (history nil)
         (distill nil)
         (compress nil)
         (epoch 0)
         (read-chunk-bytes (or chunk-bytes 4096))
         (after nil))
    (photon-model-metadata-put model 'tokenizer 'char-stream)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (photon-model-metadata-put model 'stream-chunk-bytes read-chunk-bytes)
    (setq before (photon-evaluate-text-file-stream
                  model path vocab context-size chunk-bytes))
    (photon-model-clear-cache model)
    (while (< epoch epochs)
      (setq history
            (cons (cons epoch
                        (photon-train-epoch-token-reader
                         model
                         (photon-make-file-token-reader
                          path vocab read-chunk-bytes)
                         context-size
                         learning-rate))
                  history))
      (setq epoch (1+ epoch)))
    (setq history (nreverse history))
    (when (> learning-rate 0.0)
      (photon-rebuild-target-state-clusters-token-reader
       model
       (photon-make-file-token-reader path vocab read-chunk-bytes)
       context-size))
    (setq after (photon-evaluate-text-file-stream
                 model path vocab context-size read-chunk-bytes))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'before before)
          (cons 'history history)
          (cons 'after after))))

(defun photon-train-text-file-stream-split (train-path eval-path
                                                       context-size hidden-size
                                                       chunk-size epochs
                                                       learning-rate
                                                       &optional chunk-bytes)
  (let* ((vocab (photon-build-char-vocab-file-stream train-path chunk-bytes))
         (config (photon-make-config (length vocab) hidden-size chunk-size 2))
         (model (photon-make-model config))
         (read-chunk-bytes (or chunk-bytes 4096))
         before
         (history nil)
         (epoch 0)
         (after nil)
         (fallback-after nil))
    (photon-model-metadata-put model 'tokenizer 'char-stream)
    (photon-model-metadata-put model 'vocab vocab)
    (photon-model-metadata-put model 'context-size context-size)
    (photon-model-metadata-put model 'stream-chunk-bytes read-chunk-bytes)
    (photon-model-metadata-put model 'train-path train-path)
    (photon-model-metadata-put model 'eval-path eval-path)
    (setq before (photon-evaluate-text-file-stream
                  model eval-path vocab context-size read-chunk-bytes))
    (photon-model-clear-cache model)
    (while (< epoch epochs)
      (setq history
            (cons (cons epoch
                        (photon-train-epoch-token-reader
                         model
                         (photon-make-file-token-reader
                          train-path vocab read-chunk-bytes)
                         context-size
                         learning-rate))
                  history))
      (setq epoch (1+ epoch)))
    (setq history (nreverse history))
    (when (> learning-rate 0.0)
      (photon-rebuild-target-state-clusters-token-reader
       model
       (photon-make-file-token-reader train-path vocab read-chunk-bytes)
       context-size))
    (when (> learning-rate 0.0)
      (setq distill
            (photon-distill-output-head-from-cluster-token-reader-epochs
             model
             (lambda ()
               (photon-make-file-token-reader
                eval-path vocab read-chunk-bytes))
             context-size
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-cluster-distill-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-cluster-distill-epochs)))))
    (when (> learning-rate 0.0)
      (setq compress
            (photon-compress-output-head-prototypes-pocket
             model
             (* learning-rate
                (photon-model-param-get
                 model 'output-head-compress-weight))
             (truncate
              (photon-model-param-get
               model 'output-head-compress-epochs))
             (lambda ()
               (photon-make-file-token-reader
                eval-path vocab read-chunk-bytes))
             context-size)))
    (setq after (photon-evaluate-text-file-stream
                 model eval-path vocab context-size read-chunk-bytes))
    (setq fallback-after
          (photon-evaluate-text-file-stream-fallback
           model eval-path vocab context-size read-chunk-bytes))
    (list (cons 'model model)
          (cons 'vocab vocab)
          (cons 'before before)
          (cons 'history history)
          (cons 'cluster-distill distill)
          (cons 'output-head-compress compress)
          (cons 'after after)
          (cons 'fallback-after fallback-after))))

(defun photon-model-generate-text (model prompt steps &optional options)
  (let* ((vocab (photon-model-metadata-get model 'vocab))
         (context-size (or (photon-model-metadata-get model 'context-size)
                           (photon-config-get (photon-model-config model) 'chunk-size)))
         (tokens (photon-model-encode-text model prompt))
         (window tokens)
         (i 0))
    (while (< i steps)
      (when (> (length window) context-size)
        (setq window (last window context-size)))
      (let* ((states (photon-forward-model model window))
             (state (photon-prediction-state states window))
             (logits (photon-model-readout-logits model state window))
             (adjusted-logits
              (photon-generation-adjust-logits
               model window logits
               (photon-generation-options-with-history options tokens)))
             (next-token (photon-sample-index
                          adjusted-logits options window)))
        (setq tokens (append tokens (list next-token)))
        (setq window (photon-window-append window next-token context-size)))
      (setq i (1+ i)))
    (photon-model-decode-tokens model tokens)))

(defun photon-model-generate-text-with-trace (model prompt steps &optional options)
  (let* ((vocab (photon-model-metadata-get model 'vocab))
         (context-size (or (photon-model-metadata-get model 'context-size)
                           (photon-config-get
                            (photon-model-config model) 'chunk-size)))
         (tokens (photon-model-encode-text model prompt))
         (window tokens)
         (trace nil)
         (i 0))
    (while (< i steps)
      (when (> (length window) context-size)
        (setq window (last window context-size)))
      (let* ((record (photon-model-next-token-trace-record
                      model vocab window i options tokens))
             (next-token (cdr (assq 'prediction record))))
        (setq trace (cons record trace))
        (setq tokens (append tokens (list next-token)))
        (setq window (photon-window-append window next-token context-size)))
      (setq i (1+ i)))
    (list (cons 'prompt prompt)
          (cons 'options options)
          (cons 'tokens tokens)
          (cons 'generated (photon-model-decode-tokens model tokens))
          (cons 'trace (nreverse trace)))))

(defun photon-save-model (model path)
  (with-temp-file path
    (insert (prin1-to-string (list (cons 'format 'photon-model-v1)
                 (cons 'version photon-version)
                 (cons 'vector-backend (photon-vector-backend-name))
                 (cons 'tokenizer
                       (photon-model-metadata-get model 'tokenizer))
                 (cons 'vocab
                       (photon-model-metadata-get model 'vocab))
                 (cons 'wordpiece-max-piece-size
                       (photon-model-metadata-get
                        model 'wordpiece-max-piece-size))
                 (cons 'model model))
           )))
  path)

(defun photon-save-training-history (history path &optional metadata)
  (with-temp-file path
    (insert (prin1-to-string (list (cons 'format 'photon-training-history-v1)
                 (cons 'version photon-version)
                 (cons 'metadata metadata)
                 (cons 'history history))
           )))
  path)

(defun photon-load-training-history (path)
  (with-temp-buffer
    (insert-file-contents path)
    (let ((data (photon--read-sexp-string (buffer-string))))
      (if (and (consp data)
               (eq (cdr (assq 'format data)) 'photon-training-history-v1)
               (assq 'history data))
          (cdr (assq 'history data))
        data))))

(defun photon-model-vector-storage-p (model)
  (let ((table (photon-model-embedding-table model)))
    (and (consp table) (vectorp (car table)))))

(defun photon-restore-model-vector-backend (backend model)
  (cond
   ((or (eq backend 'elisp-vector)
        (and (null backend) (photon-model-vector-storage-p model)))
    (photon-use-elisp-vector-backend))
   ((eq backend 'elisp-tensor)
    (photon-use-elisp-tensor-backend))
   ((eq backend 'list-vector)
    (photon-use-list-vector-backend)))
  model)

(defun photon-load-model (path)
  (with-temp-buffer
    (insert-file-contents path)
    (let ((data (photon--read-sexp-string (buffer-string))))
      (if (and (consp data)
               (eq (cdr (assq 'format data)) 'photon-model-v1)
               (assq 'model data))
          (photon-restore-model-vector-backend
           (cdr (assq 'vector-backend data))
           (cdr (assq 'model data)))
        data))))

(defun photon-vote-add (counts token)
  (let ((cell (assq token counts)))
    (if cell
        (setcdr cell (1+ (cdr cell)))
      (setq counts (cons (cons token 1) counts)))
    counts))

(defun photon-vote-winner (counts)
  (let ((winner (car counts)))
    (while counts
      (when (> (cdr (car counts)) (cdr winner))
        (setq winner (car counts)))
      (setq counts (cdr counts)))
    (car winner)))

(defun photon-multi-query-next-token (config prompt query-count)
  (let ((counts nil)
        (i 0))
    (while (< i query-count)
      (let ((variant (append prompt (list (mod i (photon-config-get config 'vocab-size))))))
        (setq counts (photon-vote-add counts (photon-next-token config variant))))
      (setq i (1+ i)))
    (photon-vote-winner counts)))

(defun photon-model-multi-query-next-token (model prompt query-count)
  (let ((counts nil)
        (config (photon-model-config model))
        (i 0))
    (while (< i query-count)
      (let ((variant (append prompt (list (mod i (photon-config-get config 'vocab-size))))))
        (setq counts (photon-vote-add counts (photon-model-next-token model variant))))
      (setq i (1+ i)))
    (photon-vote-winner counts)))

(defun photon-demo ()
  (let* ((config (photon-make-config 8 4 2 2))
         (model (photon-make-model config))
         (prompt '(1 3 5))
         (corpus '(1 2 1 2 1 2))
         (before (photon-evaluate model corpus 2))
         (history (photon-train model corpus 2 1 0.20))
         (after (photon-evaluate model corpus 2))
         (generated (photon-model-generate model prompt 2))
         (voted (photon-model-multi-query-next-token model prompt 3)))
    (list (cons 'version photon-version)
          (cons 'prompt prompt)
          (cons 'train-before before)
          (cons 'train-history history)
          (cons 'train-after after)
          (cons 'generated generated)
          (cons 'multi-query-next voted))))

(provide 'photon)

;;; photon.el ends here
