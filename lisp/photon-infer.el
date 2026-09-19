;;; photon-infer.el --- end-to-end text generation pipeline  -*- lexical-binding: t; -*-

;; Glue that turns a string prompt into generated text: tokenize with a
;; photon-bpe tokenizer, run a photon-transformer on whatever
;; photon-tensor backend is active (CPU or the nelisp-gpu GPU backend),
;; then detokenize.  The transformer and tensor ops are unchanged; this
;; only wires the tokenizer to the model.

;;; Code:

(require 'photon-tensor)
(require 'photon-transformer)
(require 'photon-bpe)

;;;###autoload
(defun photon-infer-generate (model bpe prompt steps)
  "Generate STEPS tokens after PROMPT (a string) and return decoded text.
Encodes PROMPT with tokenizer BPE, greedily generates with MODEL on the
active photon-tensor backend, and decodes the full token sequence.  Token
ids are clamped into the model's vocab as a safety net for tokenizer/model
mismatch (a no-op when the model was built with the tokenizer's vocab)."
  (let* ((vocab (plist-get (plist-get model :config) :vocab))
         (ids (mapcar (lambda (x) (if (>= x vocab) (mod x vocab) x))
                      (photon-bpe-encode bpe prompt)))
         (out (photon-transformer-generate model ids steps)))
    (photon-bpe-decode bpe out)))

;;;###autoload
(defun photon-infer-token-count (bpe prompt)
  "Return how many tokens tokenizer BPE produces for PROMPT."
  (length (photon-bpe-encode bpe prompt)))

(provide 'photon-infer)
;;; photon-infer.el ends here
