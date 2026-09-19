;;; transformer-gpu-generate.el --- GPU inference demo (GOAL #13)  -*- lexical-binding: t; -*-
;; A small Transformer generates tokens on the GPU backend (nelisp-gpu
;; SPIR-V/Vulkan kernels).  Run from nelisp-photon root with
;; ../nelisp-gpu/host/vkrun built:
;;   emacs -Q --batch -L lisp -l test/transformer-gpu-generate.el
(add-to-list 'load-path (expand-file-name "lisp"))
(require 'photon-transformer)
(require 'photon-tensor-gpu)

(let* ((vocab 16)
       (m (photon-transformer-create :vocab vocab :dim 8 :heads 2
                                     :context 8 :layers 2 :ff 16))
       (prompt '(1 2 3))
       (steps 5)
       (cpu-gen (photon-transformer-generate m prompt steps)))
  (photon-tensor-use-gpu-backend)
  (let ((gpu-gen (photon-transformer-generate m prompt steps)))
    (photon-tensor-use-cpu-backend)
    (let ((count-ok (= (length gpu-gen) (+ (length prompt) steps)))
          (range-ok t) (i 0) (n (length gpu-gen)))
      (while (< i n)
        (let ((tk (nth i gpu-gen)))
          (unless (and (integerp tk) (>= tk 0) (< tk vocab)) (setq range-ok nil)))
        (setq i (1+ i)))
      (princ (format "prompt=%S\ncpu-gen=%S\ngpu-gen=%S\ncpu==gpu=%S\n"
                     prompt cpu-gen gpu-gen (equal cpu-gen gpu-gen)))
      ;; GOAL: a small Transformer runs an inference (generation) step on
      ;; the GPU, producing the right number of in-vocab tokens.
      (princ (format "TRANSFORMER-GPU-GENERATE=%s\n"
                     (if (and count-ok range-ok) "PASS" "FAIL"))))))
;;; transformer-gpu-generate.el ends here
