;;; photon-weights.el --- save/load transformer weights to a file  -*- lexical-binding: t; -*-

;; Serialize a photon-transformer's config and every weight tensor to a
;; self-describing sexp file, and load it back into a fresh transformer
;; scaffold.  Floats round-trip exactly through `prin1'/`read' (Emacs
;; prints the shortest representation that reads back identically), so a
;; loaded model is bit-for-bit identical and its forward pass is
;; deterministic against the original.

;;; Code:

(require 'photon-tensor)
(require 'photon-transformer)

(defconst photon-weights-format "photon-weights-v1")

(defconst photon-weights--top-keys '(:wte :wpe :lnfg :lnfb :head)
  "Top-level weight tensor keys in a transformer model.")

(defconst photon-weights--layer-keys
  '(:ln1g :ln1b :wq :wk :wv :wo :ln2g :ln2b :w1 :b1 :w2 :b2)
  "Per-layer weight tensor keys in a transformer model.")

(defun photon-weights--copy (tn)
  "Deep-copy tensor TN (a [SHAPE DATA] vector)."
  (photon-tensor (copy-sequence (photon-tensor-shape tn))
                 (copy-sequence (photon-tensor-data tn))))

(defun photon-weights--collect (model)
  "Return an alist (NAME . TENSOR) covering every weight in MODEL."
  (let (acc)
    (dolist (k photon-weights--top-keys)
      (push (cons (substring (symbol-name k) 1) (plist-get model k)) acc))
    (let ((i 0))
      (dolist (ly (plist-get model :layers))
        (dolist (k photon-weights--layer-keys)
          (push (cons (format "layer%d.%s" i (substring (symbol-name k) 1))
                      (plist-get ly k))
                acc))
        (setq i (1+ i))))
    (nreverse acc)))

;;;###autoload
(defun photon-weights-save (model path)
  "Write MODEL's config and all weight tensors to PATH."
  (let ((form (list :format photon-weights-format
                    :config (plist-get model :config)
                    :weights (photon-weights--collect model))))
    (with-temp-buffer
      (let ((print-length nil) (print-level nil))
        (prin1 form (current-buffer)))
      (let ((coding-system-for-write 'utf-8))
        (write-region (point-min) (point-max) path nil 'silent)))
    path))

(defun photon-weights--read-file (path)
  "Read the sexp stored at PATH."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8)) (insert-file-contents path))
    (goto-char (point-min))
    (read (current-buffer))))

;;;###autoload
(defun photon-weights-load (path)
  "Load a photon-weights file at PATH into a fresh transformer; return it.
Builds the scaffold with `photon-transformer-create' from the stored
config, then overwrites every weight tensor with a copy from the file."
  (let ((form (photon-weights--read-file path)))
    (unless (equal (plist-get form :format) photon-weights-format)
      (error "photon-weights: bad format %S" (plist-get form :format)))
    (let* ((cfg (plist-get form :config))
           (weights (plist-get form :weights))
           (model (apply #'photon-transformer-create cfg)))
      (dolist (k photon-weights--top-keys)
        (let ((tn (cdr (assoc (substring (symbol-name k) 1) weights))))
          (when tn (setq model (plist-put model k (photon-weights--copy tn))))))
      (let ((i 0))
        (dolist (ly (plist-get model :layers))
          (dolist (k photon-weights--layer-keys)
            (let ((tn (cdr (assoc (format "layer%d.%s" i (substring (symbol-name k) 1))
                                  weights))))
              (when tn (plist-put ly k (photon-weights--copy tn)))))
          (setq i (1+ i))))
      model)))

(provide 'photon-weights)
;;; photon-weights.el ends here
