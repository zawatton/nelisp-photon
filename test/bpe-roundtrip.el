;;; bpe-roundtrip.el --- tests for the photon-bpe tokenizer  -*- lexical-binding: t; -*-
;; Run from the nelisp-photon root:
;;   emacs -Q --batch -L lisp -l test/bpe-roundtrip.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-bpe)

(defvar bpe--fail 0)
(defun bpe--ck (name ok)
  (princ (format "%-46s %s\n" name (if ok "PASS"
                                     (progn (setq bpe--fail (1+ bpe--fail)) "FAIL")))))

(let* ((corpus '("the quick brown fox jumps over the lazy dog"
                 "the cat sat on the mat and the dog ran"
                 "to be or not to be that is the question"))
       (bpe (photon-bpe-train corpus 80))
       (samples (append corpus
                        '("" "a" "the the the"
                          "unseen words with PUNCT!? and 123"
                          "日本語のテキストもラウンドトリップする"
                          "mixed 日本語 and english 42%"))))
  ;; 1. round-trip on trained + unseen + multibyte text
  (let ((ok t))
    (dolist (s samples)
      (unless (string= s (photon-bpe-decode bpe (photon-bpe-encode bpe s)))
        (setq ok nil)))
    (bpe--ck "round-trip: decode(encode(s)) == s" ok))

  ;; 2. all emitted ids are within the learned vocab
  (let ((ok t) (vsz (photon-bpe-size bpe)))
    (dolist (s samples)
      (dolist (id (photon-bpe-encode bpe s))
        (when (or (< id 0) (>= id vsz)) (setq ok nil))))
    (bpe--ck "ids within vocab size" ok))

  ;; 3. BPE actually compresses trained text (fewer tokens than raw bytes)
  (let* ((s (car corpus))
         (nbytes (length (encode-coding-string s 'utf-8)))
         (ntok (length (photon-bpe-encode bpe s))))
    (bpe--ck "compression: tokens < raw bytes" (< ntok nbytes)))

  ;; 4. save -> load reproduces identical encoding and round-trip
  (let* ((tmp (make-temp-file "photon-bpe" nil ".el"))
         (_ (photon-bpe-save bpe tmp))
         (bpe2 (photon-bpe-load tmp))
         (ok-enc t) (ok-rt t))
    (dolist (s samples)
      (unless (equal (photon-bpe-encode bpe s) (photon-bpe-encode bpe2 s))
        (setq ok-enc nil))
      (unless (string= s (photon-bpe-decode bpe2 (photon-bpe-encode bpe2 s)))
        (setq ok-rt nil)))
    (delete-file tmp)
    (bpe--ck "persist: loaded encode == trained encode" ok-enc)
    (bpe--ck "persist: loaded tokenizer round-trips" ok-rt)))

(princ (format "BPE-ROUNDTRIP %s (%d failures)\n"
               (if (= bpe--fail 0) "ALL-PASS" "HAS-FAILURES") bpe--fail))
(kill-emacs (if (= bpe--fail 0) 0 1))
;;; bpe-roundtrip.el ends here
