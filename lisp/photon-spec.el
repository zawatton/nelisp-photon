;;; photon-spec.el --- hardware-aware model-config autosizer  -*- lexical-binding: t; -*-

;; Picks a GPT-style transformer configuration that fits the host's RAM
;; and CPU core count, so photon scales from a laptop to a workstation
;; without hand-tuning.  The chosen plist feeds `photon-transformer-create'.

;;; Code:

(require 'cl-lib)

(defconst photon-spec--ladder
  ;; Ascending size tiers; each is a config plist understood by
  ;; `photon-transformer-create'.  Head dim is ~64 (heads = dim/64) and
  ;; dim is always divisible by heads.  Estimated parameter count is
  ;; monotonically non-decreasing across the ladder, which the autosizer
  ;; relies on for a clean "largest tier that fits" search.
  '((:dim 64   :heads 1  :context 128  :layers 2  :ff 256)
    (:dim 128  :heads 2  :context 256  :layers 4  :ff 512)
    (:dim 256  :heads 4  :context 512  :layers 6  :ff 1024)
    (:dim 384  :heads 6  :context 768  :layers 8  :ff 1536)
    (:dim 512  :heads 8  :context 1024 :layers 12 :ff 2048)
    (:dim 768  :heads 12 :context 1024 :layers 12 :ff 3072)
    (:dim 1024 :heads 16 :context 2048 :layers 16 :ff 4096)
    (:dim 1536 :heads 24 :context 2048 :layers 24 :ff 6144)
    (:dim 2048 :heads 32 :context 2048 :layers 24 :ff 8192))
  "Model-size tiers, smallest first.  See `photon-spec-autosize'.")

(defconst photon-spec-default-vocab 32000
  "Vocabulary size assumed when sizing the model (BPE scale).")

;;;###autoload
(defun photon-spec-cpu-cores ()
  "Return the number of logical CPU cores on this host (>= 1)."
  (cond
   ((fboundp 'num-processors) (max 1 (num-processors)))
   ((file-readable-p "/proc/cpuinfo")
    (with-temp-buffer
      (insert-file-contents "/proc/cpuinfo")
      (max 1 (count-matches "^processor[ \t]*:" (point-min) (point-max)))))
   (t 1)))

;;;###autoload
(defun photon-spec-ram-bytes ()
  "Return total system RAM in bytes (best effort, 4 GiB fallback)."
  (let ((mi (and (fboundp 'memory-info) (memory-info))))
    (cond
     ((and mi (integerp (nth 0 mi)) (> (nth 0 mi) 0))
      (* 1024 (nth 0 mi)))                 ; `memory-info' reports KiB
     ((file-readable-p "/proc/meminfo")
      (with-temp-buffer
        (insert-file-contents "/proc/meminfo")
        (goto-char (point-min))
        (if (re-search-forward "^MemTotal:[ \t]*\\([0-9]+\\)[ \t]*kB" nil t)
            (* 1024 (string-to-number (match-string 1)))
          (* 4 1024 1024 1024))))
     (t (* 4 1024 1024 1024)))))

;;;###autoload
(defun photon-spec-detect ()
  "Return a plist (:ram-bytes N :cores N) describing this host."
  (list :ram-bytes (photon-spec-ram-bytes) :cores (photon-spec-cpu-cores)))

(defun photon-spec--param-bytes (cfg vocab)
  "Estimate fp32 weight bytes for tier CFG at VOCAB.
Counts token + positional embeddings, per-layer attention (qkvo) and
MLP weights, and the output head."
  (let* ((dim (plist-get cfg :dim))
         (ff (plist-get cfg :ff))
         (ctx (plist-get cfg :context))
         (layers (plist-get cfg :layers))
         (embed (+ (* vocab dim) (* ctx dim)))
         (per-layer (+ (* 4 dim dim) (* 2 dim ff)))
         (params (+ embed (* layers per-layer) (* vocab dim))))
    (* 4 params)))

;;;###autoload
(cl-defun photon-spec-autosize (&key ram-bytes cores vocab fraction mode)
  "Return a transformer config plist sized to the host.

RAM-BYTES and CORES default to detection via `photon-spec-detect'.
VOCAB defaults to `photon-spec-default-vocab'.  FRACTION is the share
of RAM the weights may occupy (default 0.25).  MODE `train' reserves
4x for gradients and optimizer state; `infer' (default) reserves 1.2x.

Both RAM and core count cap the tier, so the result never exceeds what
the machine can hold or feasibly run.  The returned plist carries
:vocab/:dim/:heads/:context/:layers/:ff (for `photon-transformer-create')
plus :threads and :tier for reporting."
  (let* ((ram (or ram-bytes (photon-spec-ram-bytes)))
         (nc (or cores (photon-spec-cpu-cores)))
         (vocab (or vocab photon-spec-default-vocab))
         (frac (or fraction 0.25))
         (factor (pcase mode ('train 4.0) (_ 1.2)))
         (budget (/ (* ram frac) factor))
         (n (length photon-spec--ladder))
         (ram-tier 0) (i 0))
    ;; Largest tier whose estimated footprint fits the budget (floor at 0).
    (while (< i n)
      (when (<= (photon-spec--param-bytes (nth i photon-spec--ladder) vocab) budget)
        (setq ram-tier i))
      (setq i (1+ i)))
    (let* ((core-tier (1- (max 1 (min n nc))))
           (idx (min ram-tier core-tier))
           (cfg (nth idx photon-spec--ladder)))
      (append (list :vocab vocab :threads nc :tier (1+ idx))
              (copy-sequence cfg)))))

;;;###autoload
(defun photon-spec-describe ()
  "Return a human-readable summary of the autosized config on this host."
  (let* ((hw (photon-spec-detect))
         (cfg (photon-spec-autosize)))
    (format "RAM=%.1fGiB cores=%d -> tier %d: dim=%d layers=%d heads=%d ctx=%d ff=%d"
            (/ (plist-get hw :ram-bytes) (float (* 1024 1024 1024)))
            (plist-get hw :cores) (plist-get cfg :tier)
            (plist-get cfg :dim) (plist-get cfg :layers)
            (plist-get cfg :heads) (plist-get cfg :context) (plist-get cfg :ff))))

(provide 'photon-spec)
;;; photon-spec.el ends here
