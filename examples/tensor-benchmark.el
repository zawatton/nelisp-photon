;;; tensor-benchmark.el --- Compare photon vector/tensor backends -*- lexical-binding: t; -*-

(load-file (expand-file-name "../lisp/photon.el" (file-name-directory load-file-name)))

(defun photon-tensor-benchmark-env-int (name default)
  (let ((value (getenv name)))
    (if (and value (> (length value) 0))
        (string-to-number value)
      default)))

(defun photon-tensor-benchmark-vector-row (row size)
  (let ((values nil)
        (col 0))
    (while (< col size)
      (setq values
            (cons (/ (float (1+ (mod (+ (* row 17) (* col 13)) 29)))
                     29.0)
                  values))
      (setq col (1+ col)))
    (photon-vector-from-list (nreverse values))))

(defun photon-tensor-benchmark-matrix (size)
  (let ((rows nil)
        (row 0))
    (while (< row size)
      (setq rows (cons (photon-tensor-benchmark-vector-row row size) rows))
      (setq row (1+ row)))
    (photon-matrix-from-rows (nreverse rows))))

(defun photon-tensor-benchmark-vector (size)
  (let ((values nil)
        (i 0))
    (while (< i size)
      (setq values (cons (/ (float (1+ (mod (* i 7) 31))) 31.0)
                         values))
      (setq i (1+ i)))
    (photon-vector-from-list (nreverse values))))

(defun photon-tensor-benchmark-run-loop (iters fn)
  (let ((i 0)
        (value nil)
        (start (float-time)))
    (while (< i iters)
      (setq value (funcall fn))
      (setq i (1+ i)))
    (list (cons 'elapsed-seconds (- (float-time) start))
          (cons 'last value))))

(defun photon-tensor-benchmark-row (backend size iters)
  (photon-use-vector-backend-name backend)
  (let* ((matrix (photon-tensor-benchmark-matrix size))
         (vec (photon-tensor-benchmark-vector size))
         (tensor-a (photon-tensor-zeros (list size size)))
         (tensor-b (photon-tensor-map (lambda (_value) 1.0) tensor-a))
         (matvec
          (photon-tensor-benchmark-run-loop
           iters
           (lambda ()
             (photon-matrix-vector-dot matrix vec))))
         (tensor
          (photon-tensor-benchmark-run-loop
           iters
           (lambda ()
             (photon-tensor-map
              (lambda (value) (+ value 0.5))
              (photon-tensor-scale
               0.5
               (photon-tensor-add tensor-a tensor-b)))))))
    (list (cons 'backend (photon-vector-backend-name))
          (cons 'features (photon-vector-backend-features))
          (cons 'size size)
          (cons 'iterations iters)
          (cons 'matrix-shape (photon-matrix-shape matrix))
          (cons 'tensor-shape (photon-tensor-shape tensor-a))
          (cons 'matvec-elapsed-seconds
                (cdr (assq 'elapsed-seconds matvec)))
          (cons 'tensor-elapsed-seconds
                (cdr (assq 'elapsed-seconds tensor)))
          (cons 'matvec-output-shape
                (photon-tensor-shape (cdr (assq 'last matvec))))
          (cons 'tensor-output-shape
                (photon-tensor-shape (cdr (assq 'last tensor)))))))

(let* ((size (photon-tensor-benchmark-env-int "PHOTON_TENSOR_BENCH_SIZE" 24))
       (iters (photon-tensor-benchmark-env-int "PHOTON_TENSOR_BENCH_ITERS" 200))
       (backends '("list-vector" "elisp-vector" "elisp-tensor"))
       (rows nil))
  (while backends
    (setq rows
          (cons (photon-tensor-benchmark-row (car backends) size iters)
                rows))
    (setq backends (cdr backends)))
  (prin1 (list (cons 'benchmark 'tensor-backends)
               (cons 'rows (nreverse rows))))
  (princ "\n"))

;;; tensor-benchmark.el ends here
