;;; photon-bpe.el --- byte-level BPE tokenizer for nelisp-photon  -*- lexical-binding: t; -*-

;; A byte-level Byte-Pair-Encoding tokenizer.  The base alphabet is the
;; 256 byte values, so ANY UTF-8 text round-trips exactly regardless of
;; the learned merges -- decode(encode(s)) == s by construction, because
;; merges only group contiguous bytes and decoding concatenates the
;; bytes back in order.

;;; Code:

(require 'cl-lib)

(cl-defstruct (photon-bpe (:constructor photon-bpe--make))
  ranks      ; hash "A,B" -> merge rank (priority; lower = applied first)
  merged     ; hash "A,B" -> new token id
  id->bytes  ; hash id -> list of byte values (0..255)
  size)      ; next free id == current vocab size

(defun photon-bpe--bytes (text)
  "Return TEXT as a list of byte values (0..255) via UTF-8."
  (append (encode-coding-string text 'utf-8) nil))

(defun photon-bpe-new ()
  "Return a fresh tokenizer holding only the 256 base byte tokens."
  (let ((id->bytes (make-hash-table :test 'eq)) (i 0))
    (while (< i 256) (puthash i (list i) id->bytes) (setq i (1+ i)))
    (photon-bpe--make :ranks (make-hash-table :test 'equal)
                      :merged (make-hash-table :test 'equal)
                      :id->bytes id->bytes :size 256)))

(defun photon-bpe--merge-seq (s a b newid)
  "Return id-list S with every adjacent (A B) replaced by NEWID."
  (let (out)
    (while s
      (if (and (cdr s) (= (car s) a) (= (cadr s) b))
          (progn (push newid out) (setq s (cddr s)))
        (push (car s) out)
        (setq s (cdr s))))
    (nreverse out)))

(defun photon-bpe--pairs-count (seqs)
  "Count adjacent pairs across all id-lists in SEQS -> hash key->count."
  (let ((h (make-hash-table :test 'equal)))
    (dolist (s seqs h)
      (let ((p s))
        (while (and p (cdr p))
          (let ((k (format "%d,%d" (car p) (cadr p))))
            (puthash k (1+ (gethash k h 0)) h))
          (setq p (cdr p)))))))

;;;###autoload
(defun photon-bpe-train (texts num-merges)
  "Learn up to NUM-MERGES merges from TEXTS (list of strings); return a bpe."
  (let* ((bpe (photon-bpe-new))
         (seqs (mapcar #'photon-bpe--bytes texts))
         (rank 0))
    (while (< rank num-merges)
      (let ((counts (photon-bpe--pairs-count seqs)) (best nil) (bestc 1))
        (maphash (lambda (k c) (when (> c bestc) (setq bestc c best k))) counts)
        (if (null best)
            (setq rank num-merges)        ; no pair occurs more than once
          (let* ((parts (split-string best ","))
                 (a (string-to-number (car parts)))
                 (b (string-to-number (cadr parts)))
                 (newid (photon-bpe-size bpe)))
            (puthash best rank (photon-bpe-ranks bpe))
            (puthash best newid (photon-bpe-merged bpe))
            (puthash newid (append (gethash a (photon-bpe-id->bytes bpe))
                                   (gethash b (photon-bpe-id->bytes bpe)))
                     (photon-bpe-id->bytes bpe))
            (setf (photon-bpe-size bpe) (1+ newid))
            (setq seqs (mapcar (lambda (s) (photon-bpe--merge-seq s a b newid)) seqs))
            (setq rank (1+ rank))))))
    bpe))

(defun photon-bpe--encode-ids (bpe ids)
  "Greedily apply learned merges to base IDS, lowest rank first."
  (let ((ranks (photon-bpe-ranks bpe)) (merged (photon-bpe-merged bpe))
        (go t))
    (while go
      (let ((p ids) (br nil) (ba nil) (bb nil))
        (while (and p (cdr p))
          (let* ((k (format "%d,%d" (car p) (cadr p))) (r (gethash k ranks)))
            (when (and r (or (null br) (< r br)))
              (setq br r ba (car p) bb (cadr p))))
          (setq p (cdr p)))
        (if (null br)
            (setq go nil)
          (setq ids (photon-bpe--merge-seq
                     ids ba bb (gethash (format "%d,%d" ba bb) merged))))))
    ids))

;;;###autoload
(defun photon-bpe-encode (bpe text)
  "Encode TEXT to a list of token ids using tokenizer BPE."
  (photon-bpe--encode-ids bpe (photon-bpe--bytes text)))

;;;###autoload
(defun photon-bpe-decode (bpe ids)
  "Decode token IDS back to the original string using tokenizer BPE."
  (let (bytes)
    (dolist (id ids)
      (setq bytes (nconc bytes (copy-sequence (gethash id (photon-bpe-id->bytes bpe))))))
    (decode-coding-string (apply #'unibyte-string bytes) 'utf-8)))

;; --- persistence (a real tokenizer loads its merges from a file) ------

(defconst photon-bpe-format "photon-bpe-v1")

(defun photon-bpe--merge-list (bpe)
  "Return the merges of BPE as a list of (A B NEWID) ordered by rank."
  (let (acc)
    (maphash (lambda (k newid)
               (let* ((parts (split-string k ","))
                      (a (string-to-number (car parts)))
                      (b (string-to-number (cadr parts))))
                 (push (list (gethash k (photon-bpe-ranks bpe)) a b newid) acc)))
             (photon-bpe-merged bpe))
    (mapcar #'cdr (sort acc (lambda (x y) (< (car x) (car y)))))))

;;;###autoload
(defun photon-bpe-save (bpe path)
  "Write tokenizer BPE merges to PATH."
  (with-temp-buffer
    (let ((print-length nil) (print-level nil))
      (prin1 (list :format photon-bpe-format
                   :merges (photon-bpe--merge-list bpe))
             (current-buffer)))
    (let ((coding-system-for-write 'utf-8))
      (write-region (point-min) (point-max) path nil 'silent)))
  path)

;;;###autoload
(defun photon-bpe-load (path)
  "Load a tokenizer from PATH by replaying its merges over the base bytes."
  (let (form)
    (with-temp-buffer
      (let ((coding-system-for-read 'utf-8)) (insert-file-contents path))
      (goto-char (point-min))
      (setq form (read (current-buffer))))
    (unless (equal (plist-get form :format) photon-bpe-format)
      (error "photon-bpe: bad format %S" (plist-get form :format)))
    (let ((bpe (photon-bpe-new)) (rank 0))
      (dolist (m (plist-get form :merges))
        (let* ((a (nth 0 m)) (b (nth 1 m)) (newid (nth 2 m))
               (k (format "%d,%d" a b)))
          (puthash k rank (photon-bpe-ranks bpe))
          (puthash k newid (photon-bpe-merged bpe))
          (puthash newid (append (gethash a (photon-bpe-id->bytes bpe))
                                 (gethash b (photon-bpe-id->bytes bpe)))
                   (photon-bpe-id->bytes bpe))
          (setf (photon-bpe-size bpe) (max (photon-bpe-size bpe) (1+ newid)))
          (setq rank (1+ rank))))
      bpe)))

(provide 'photon-bpe)
;;; photon-bpe.el ends here
