;;; photon-transformer.el --- GPT-style transformer forward on photon-tensor  -*- lexical-binding: t; -*-

;; A minimal decoder-only (GPT-style) transformer forward pass built on
;; the photon-tensor core.  Pre-norm blocks: token+positional embedding
;; -> N x (LayerNorm -> causal multi-head self-attention -> residual ->
;; LayerNorm -> MLP(GELU) -> residual) -> final LayerNorm -> output head.
;;
;; Backend-agnostic: it only calls photon-tensor ops, so once those ops
;; gain a GPU (nelisp-gpu) backend the same forward runs on the GPU.
;;
;; A model is a plist:
;;   :config (:vocab :dim :heads :context :layers :ff)
;;   :wte  (vocab dim)   :wpe (context dim)
;;   :layers (list of layer plists)
;;   :lnfg (dim) :lnfb (dim)   :head (vocab dim)
;; Each layer plist:
;;   :ln1g :ln1b (dim)  :wq :wk :wv :wo (dim dim)
;;   :ln2g :ln2b (dim)  :w1 (ff dim) :b1 (ff) :w2 (dim ff) :b2 (dim)

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)

(defun photon-transformer--slice-cols (a c0 ncols)
  "Return columns [C0, C0+NCOLS) of 2D tensor A as a fresh tensor."
  (let* ((sh (photon-tensor-shape a)) (rows (car sh)) (cols (nth 1 sh))
         (d (photon-tensor-data a)) (out (make-vector (* rows ncols) 0.0)) (i 0))
    (while (< i rows)
      (let ((j 0))
        (while (< j ncols)
          (aset out (+ (* i ncols) j) (aref d (+ (* i cols) c0 j)))
          (setq j (1+ j))))
      (setq i (1+ i)))
    (photon-tensor (list rows ncols) out)))

(defun photon-transformer--set-cols (dst src c0)
  "Write 2D tensor SRC into columns starting at C0 of 2D tensor DST."
  (let* ((ds (photon-tensor-shape dst)) (rows (car ds)) (cols (nth 1 ds))
         (ncols (nth 1 (photon-tensor-shape src)))
         (dd (photon-tensor-data dst)) (sd (photon-tensor-data src)) (i 0))
    (while (< i rows)
      (let ((j 0))
        (while (< j ncols)
          (aset dd (+ (* i cols) c0 j) (aref sd (+ (* i ncols) j)))
          (setq j (1+ j))))
      (setq i (1+ i)))
    dst))

(defun photon-transformer--causal-mask (s)
  "Mask the upper triangle (j>i) of square 2D tensor S to -1e30 in place."
  (let* ((n (car (photon-tensor-shape s))) (d (photon-tensor-data s)) (i 0))
    (while (< i n)
      (let ((j (1+ i)))
        (while (< j n) (aset d (+ (* i n) j) -1.0e30) (setq j (1+ j))))
      (setq i (1+ i)))
    s))

(defun photon-transformer--embed (model tokens)
  "Token + positional embedding of TOKENS -> (seq x dim) tensor."
  (let* ((dim (plist-get (plist-get model :config) :dim))
         (seq (length tokens))
         (e (photon-tensor-embedding (plist-get model :wte) tokens dim))
         (wd (photon-tensor-data (plist-get model :wpe)))
         (ed (photon-tensor-data e)) (i 0))
    (while (< i seq)
      (let ((j 0))
        (while (< j dim)
          (aset ed (+ (* i dim) j) (+ (aref ed (+ (* i dim) j)) (aref wd (+ (* i dim) j))))
          (setq j (1+ j))))
      (setq i (1+ i)))
    e))

(defun photon-transformer--attention (x layer heads)
  "Causal multi-head self-attention of X (seq x dim) for one LAYER."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (hd (/ dim heads)) (scale (/ 1.0 (sqrt (float hd))))
         (q (photon-tensor-linear x (plist-get layer :wq)))
         (k (photon-tensor-linear x (plist-get layer :wk)))
         (v (photon-tensor-linear x (plist-get layer :wv)))
         (ctx (photon-tensor-create (list seq dim) 0.0)) (h 0))
    (while (< h heads)
      (let* ((c0 (* h hd))
             (qh (photon-transformer--slice-cols q c0 hd))
             (kh (photon-transformer--slice-cols k c0 hd))
             (vh (photon-transformer--slice-cols v c0 hd))
             (scores (photon-tensor-scale
                      (photon-tensor-matmul qh (photon-tensor-transpose kh)) scale)))
        (photon-transformer--causal-mask scores)
        (photon-transformer--set-cols
         ctx (photon-tensor-matmul (photon-tensor-softmax-rows scores) vh) c0))
      (setq h (1+ h)))
    (photon-tensor-linear ctx (plist-get layer :wo))))

(defun photon-transformer--ffn (x layer)
  "Position-wise feed-forward block on X (seq x dim): linear -> gelu -> linear.
Extracted so a backend can fuse the three ops into one GPU submission."
  (photon-tensor-linear
   (photon-tensor-gelu (photon-tensor-linear x (plist-get layer :w1)
                                             (plist-get layer :b1)))
   (plist-get layer :w2) (plist-get layer :b2)))

(defun photon-transformer--layer (x layer heads)
  "Run one pre-norm transformer block on X (seq x dim)."
  (let* ((a (photon-tensor-layernorm-rows x (plist-get layer :ln1g) (plist-get layer :ln1b)))
         (x1 (photon-tensor-add x (photon-transformer--attention a layer heads)))
         (b (photon-tensor-layernorm-rows x1 (plist-get layer :ln2g) (plist-get layer :ln2b)))
         (m (photon-transformer--ffn b layer)))
    (photon-tensor-add x1 m)))

(defun photon-transformer-forward (model tokens)
  "Run the transformer forward over TOKENS; return (seq x vocab) logits."
  (let* ((cfg (plist-get model :config))
         (heads (plist-get cfg :heads))
         (x (photon-transformer--embed model tokens)))
    (dolist (ly (plist-get model :layers))
      (setq x (photon-transformer--layer x ly heads)))
    (setq x (photon-tensor-layernorm-rows x (plist-get model :lnfg) (plist-get model :lnfb)))
    (photon-tensor-linear x (plist-get model :head))))

(defun photon-transformer-next-token-logits (model tokens)
  "Return the logits row (length vocab) for the position after TOKENS."
  (let* ((logits (photon-transformer-forward model tokens))
         (seq (car (photon-tensor-shape logits))))
    (photon-tensor-row logits (1- seq))))

(defun photon-transformer--argmax (vec)
  "Return the index of the maximum element of float vector VEC."
  (let ((best 0) (bv (aref vec 0)) (i 1) (n (length vec)))
    (while (< i n)
      (when (> (aref vec i) bv) (setq bv (aref vec i) best i))
      (setq i (1+ i)))
    best))

(defun photon-transformer-generate (model tokens steps)
  "Greedy-generate STEPS tokens after TOKENS; return the full token list.
Each step runs a forward pass (on whatever photon-tensor backend is
active, CPU or GPU) over the last :context tokens and appends the argmax."
  (let* ((context (plist-get (plist-get model :config) :context))
         (out (append tokens nil)) (i 0))
    (while (< i steps)
      (let* ((window (last out (min (length out) context)))
             (logits (photon-transformer-next-token-logits model window))
             (next (photon-transformer--argmax (photon-tensor-data logits))))
        (setq out (append out (list next))))
      (setq i (1+ i)))
    out))

;; --- deterministic construction (for smoke tests / reproducible runs) ---
(defun photon-transformer--init (shape scale seed)
  "Deterministic pseudo-weights of SHAPE in [-SCALE, SCALE), varied by SEED."
  (let* ((size 1) (i 0) v)
    (dolist (d shape) (setq size (* size d)))
    (setq v (make-vector size 0.0))
    (while (< i size)
      (let ((r (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536))
                  65536.0)))
        (aset v i (* (- r 0.5) 2.0 scale)))
      (setq i (1+ i)))
    (photon-tensor shape v)))

(defun photon-transformer-create (&rest cfg)
  "Build a transformer with deterministic weights.
Keywords: :vocab :dim :heads :context :layers :ff."
  (let* ((vocab (or (plist-get cfg :vocab) 16))
         (dim (or (plist-get cfg :dim) 8))
         (heads (or (plist-get cfg :heads) 2))
         (context (or (plist-get cfg :context) 8))
         (nlayers (or (plist-get cfg :layers) 2))
         (ff (or (plist-get cfg :ff) (* 4 dim)))
         (sc (/ 1.0 (sqrt (float dim))))
         (seed 0) layers)
    (cl-flet ((nx () (setq seed (1+ seed))))
      (dotimes (_ nlayers)
        (push (list :ln1g (photon-tensor-create (list dim) 1.0)
                    :ln1b (photon-tensor-create (list dim) 0.0)
                    :wq (photon-transformer--init (list dim dim) sc (nx))
                    :wk (photon-transformer--init (list dim dim) sc (nx))
                    :wv (photon-transformer--init (list dim dim) sc (nx))
                    :wo (photon-transformer--init (list dim dim) sc (nx))
                    :ln2g (photon-tensor-create (list dim) 1.0)
                    :ln2b (photon-tensor-create (list dim) 0.0)
                    :w1 (photon-transformer--init (list ff dim) sc (nx))
                    :b1 (photon-tensor-create (list ff) 0.0)
                    :w2 (photon-transformer--init (list dim ff) sc (nx))
                    :b2 (photon-tensor-create (list dim) 0.0))
              layers))
      (list :config (list :vocab vocab :dim dim :heads heads
                          :context context :layers nlayers :ff ff)
            :wte (photon-transformer--init (list vocab dim) sc (nx))
            :wpe (photon-transformer--init (list context dim) sc (nx))
            :layers (nreverse layers)
            :lnfg (photon-tensor-create (list dim) 1.0)
            :lnfb (photon-tensor-create (list dim) 0.0)
            :head (photon-transformer--init (list vocab dim) sc (nx))))))

(provide 'photon-transformer)
;;; photon-transformer.el ends here
