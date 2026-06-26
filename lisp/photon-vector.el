;;; photon-vector.el --- Vector backend protocol for photon -*- lexical-binding: t; -*-

(defun photon-repeat (value count)
  (let ((result nil)
        (i 0))
    (while (< i count)
      (setq result (cons value result))
      (setq i (1+ i)))
    (nreverse result)))

(defun photon-list-vector-map2 (fn left right)
  (let ((result nil))
    (while (and left right)
      (setq result (cons (funcall fn (car left) (car right)) result))
      (setq left (cdr left))
      (setq right (cdr right)))
    (nreverse result)))

(defun photon-list-vector-add (left right)
  (photon-list-vector-map2 '+ left right))

(defun photon-list-vector-mul (left right)
  (photon-list-vector-map2 '* left right))

(defun photon-list-vector-scale (scale vec)
  (let ((result nil))
    (while vec
      (setq result (cons (* scale (car vec)) result))
      (setq vec (cdr vec)))
    (nreverse result)))

(defun photon-list-vector-dot (left right)
  (let ((sum 0.0))
    (while (and left right)
      (setq sum (+ sum (* (car left) (car right))))
      (setq left (cdr left))
      (setq right (cdr right)))
    sum))

(defun photon-list-matrix-vector-dot (matrix vec)
  (let ((result nil))
    (while matrix
      (setq result (cons (photon-list-vector-dot (car matrix) vec) result))
      (setq matrix (cdr matrix)))
    (nreverse result)))

(defun photon-list-matrix-from-rows (rows)
  rows)

(defun photon-list-matrix-shape (matrix)
  (list (length matrix)
        (if matrix
            (length (car matrix))
          0)))

(defun photon-list-tensor-shape (tensor)
  (cond
   ((vectorp tensor)
    (let ((size (length tensor)))
      (if (and (> size 0)
               (or (vectorp (aref tensor 0)) (consp (aref tensor 0))))
          (cons size (photon-list-tensor-shape (aref tensor 0)))
        (list size))))
   ((consp tensor)
    (let ((size (length tensor)))
      (if (and (> size 0)
               (or (vectorp (car tensor)) (consp (car tensor))))
          (cons size (photon-list-tensor-shape (car tensor)))
        (list size))))
   (t nil)))

(defun photon-list-tensor-map (fn tensor)
  (cond
   ((vectorp tensor)
    (apply 'vector
           (mapcar (lambda (item)
                     (photon-list-tensor-map fn item))
                   (append tensor nil))))
   ((consp tensor)
    (mapcar (lambda (item)
              (photon-list-tensor-map fn item))
            tensor))
   (t (funcall fn tensor))))

(defun photon-list-tensor-map2 (fn left right)
  (cond
   ((and (vectorp left) (vectorp right))
    (let* ((size (length left))
           (result (make-vector size nil))
           (i 0))
      (while (< i size)
        (aset result i
              (photon-list-tensor-map2 fn (aref left i) (aref right i)))
        (setq i (1+ i)))
      result))
   ((and (consp left) (consp right))
    (let ((result nil))
      (while (and left right)
        (setq result
              (cons (photon-list-tensor-map2 fn (car left) (car right))
                    result))
        (setq left (cdr left))
        (setq right (cdr right)))
      (nreverse result)))
   (t (funcall fn left right))))

(defun photon-list-tensor-add (left right)
  (photon-list-tensor-map2 '+ left right))

(defun photon-list-tensor-scale (scale tensor)
  (photon-list-tensor-map (lambda (value) (* scale value)) tensor))

(defun photon-list-tensor-zeros (shape)
  (if (null shape)
      0.0
    (let ((count (car shape))
          (rest (cdr shape))
          (result nil)
          (i 0))
      (while (< i count)
        (setq result (cons (photon-list-tensor-zeros rest) result))
        (setq i (1+ i)))
      (nreverse result))))

(defun photon-squash (x)
  (/ x (+ 1.0 (abs x))))

(defun photon-list-vector-squash (vec)
  (let ((result nil))
    (while vec
      (setq result (cons (photon-squash (car vec)) result))
      (setq vec (cdr vec)))
    (nreverse result)))

(defun photon-list-vector-from-list (items)
  items)

(defun photon-list-vector-to-list (vec)
  vec)

(defun photon-list-vector-zeros (size)
  (photon-repeat 0.0 size))

(defconst photon-vector-fingerprint-mod 104729)

(defun photon-vector-fingerprint-value (value)
  (mod (truncate (* 1000003.0 value)) photon-vector-fingerprint-mod))

(defun photon-list-vector-fingerprint (vec)
  (let ((length 0)
        (h1 17)
        (h2 29)
        (first 0)
        (last 0)
        (has-first nil))
    (while vec
      (let ((q (photon-vector-fingerprint-value (car vec))))
        (unless has-first
          (setq first q)
          (setq has-first t))
        (setq last q)
        (setq h1 (mod (+ (* h1 257) q length)
                      photon-vector-fingerprint-mod))
        (setq h2 (mod (+ (* h2 263) (* q (1+ length)))
                      photon-vector-fingerprint-mod))
        (setq length (1+ length)))
      (setq vec (cdr vec)))
    (list 'fp length h1 h2 first last)))

(defun photon-make-list-vector-backend ()
  (list (cons 'name 'list-vector)
        (cons 'features '(vector matrix tensor elementwise fingerprint))
        (cons 'add 'photon-list-vector-add)
        (cons 'mul 'photon-list-vector-mul)
        (cons 'matvec 'photon-list-matrix-vector-dot)
        (cons 'matrix-from-rows 'photon-list-matrix-from-rows)
        (cons 'matrix-shape 'photon-list-matrix-shape)
        (cons 'tensor-shape 'photon-list-tensor-shape)
        (cons 'tensor-add 'photon-list-tensor-add)
        (cons 'tensor-scale 'photon-list-tensor-scale)
        (cons 'tensor-map 'photon-list-tensor-map)
        (cons 'tensor-zeros 'photon-list-tensor-zeros)
        (cons 'scale 'photon-list-vector-scale)
        (cons 'dot 'photon-list-vector-dot)
        (cons 'squash 'photon-list-vector-squash)
        (cons 'from-list 'photon-list-vector-from-list)
        (cons 'to-list 'photon-list-vector-to-list)
        (cons 'fingerprint 'photon-list-vector-fingerprint)
        (cons 'zeros 'photon-list-vector-zeros)))

(defun photon-elisp-vector-add (left right)
  (let* ((size (length left))
         (result (make-vector size 0.0))
         (i 0))
    (while (< i size)
      (aset result i (+ (aref left i) (aref right i)))
      (setq i (1+ i)))
    result))

(defun photon-elisp-vector-scale (scale vec)
  (let* ((size (length vec))
         (result (make-vector size 0.0))
         (i 0))
    (while (< i size)
      (aset result i (* scale (aref vec i)))
      (setq i (1+ i)))
    result))

(defun photon-elisp-vector-mul (left right)
  (let* ((size (length left))
         (result (make-vector size 0.0))
         (i 0))
    (while (< i size)
      (aset result i (* (aref left i) (aref right i)))
      (setq i (1+ i)))
    result))

(defun photon-elisp-vector-dot (left right)
  (let ((sum 0.0)
        (size (length left))
        (i 0))
    (while (< i size)
      (setq sum (+ sum (* (aref left i) (aref right i))))
      (setq i (1+ i)))
    sum))

(defun photon-elisp-matrix-vector-dot (matrix vec)
  (let* ((rows (length matrix))
         (result (make-vector rows 0.0))
         (i 0))
    (while matrix
      (aset result i (photon-elisp-vector-dot (car matrix) vec))
      (setq matrix (cdr matrix))
      (setq i (1+ i)))
    result))

(defun photon-elisp-matrix-from-rows (rows)
  rows)

(defun photon-elisp-matrix-shape (matrix)
  (photon-list-matrix-shape matrix))

(defun photon-elisp-tensor-shape (tensor)
  (photon-list-tensor-shape tensor))

(defun photon-elisp-tensor-map (fn tensor)
  (photon-list-tensor-map fn tensor))

(defun photon-elisp-tensor-add (left right)
  (photon-list-tensor-add left right))

(defun photon-elisp-tensor-scale (scale tensor)
  (photon-list-tensor-scale scale tensor))

(defun photon-elisp-tensor-zeros (shape)
  (photon-list-tensor-map 'identity (photon-list-tensor-zeros shape)))

(defun photon-elisp-vector-squash (vec)
  (let* ((size (length vec))
         (result (make-vector size 0.0))
         (i 0))
    (while (< i size)
      (aset result i (photon-squash (aref vec i)))
      (setq i (1+ i)))
    result))

(defun photon-elisp-vector-from-list (items)
  (apply 'vector items))

(defun photon-elisp-vector-to-list (vec)
  (append vec nil))

(defun photon-elisp-vector-zeros (size)
  (make-vector size 0.0))

(defun photon-elisp-vector-fingerprint (vec)
  (let ((length (length vec))
        (h1 17)
        (h2 29)
        (first 0)
        (last 0)
        (i 0))
    (while (< i length)
      (let ((q (photon-vector-fingerprint-value (aref vec i))))
        (when (= i 0)
          (setq first q))
        (setq last q)
        (setq h1 (mod (+ (* h1 257) q i)
                      photon-vector-fingerprint-mod))
        (setq h2 (mod (+ (* h2 263) (* q (1+ i)))
                      photon-vector-fingerprint-mod))
        (setq i (1+ i))))
    (list 'fp length h1 h2 first last)))

(defun photon-make-elisp-vector-backend ()
  (list (cons 'name 'elisp-vector)
        (cons 'features '(vector matrix tensor elementwise fingerprint))
        (cons 'add 'photon-elisp-vector-add)
        (cons 'mul 'photon-elisp-vector-mul)
        (cons 'matvec 'photon-elisp-matrix-vector-dot)
        (cons 'matrix-from-rows 'photon-elisp-matrix-from-rows)
        (cons 'matrix-shape 'photon-elisp-matrix-shape)
        (cons 'tensor-shape 'photon-elisp-tensor-shape)
        (cons 'tensor-add 'photon-elisp-tensor-add)
        (cons 'tensor-scale 'photon-elisp-tensor-scale)
        (cons 'tensor-map 'photon-elisp-tensor-map)
        (cons 'tensor-zeros 'photon-elisp-tensor-zeros)
        (cons 'scale 'photon-elisp-vector-scale)
        (cons 'dot 'photon-elisp-vector-dot)
        (cons 'squash 'photon-elisp-vector-squash)
        (cons 'from-list 'photon-elisp-vector-from-list)
        (cons 'to-list 'photon-elisp-vector-to-list)
        (cons 'fingerprint 'photon-elisp-vector-fingerprint)
        (cons 'zeros 'photon-elisp-vector-zeros)))

(defun photon-tensor-vectorize-rows (rows)
  (apply 'vector rows))

(defun photon-tensor-matrix-from-rows (rows)
  (photon-tensor-vectorize-rows rows))

(defun photon-tensor-matrix-shape (matrix)
  (let ((rows (length matrix)))
    (list rows
          (if (> rows 0)
              (length (aref matrix 0))
            0))))

(defun photon-tensor-matrix-vector-dot (matrix vec)
  (let* ((rows (length matrix))
         (result (make-vector rows 0.0))
         (i 0))
    (while (< i rows)
      (aset result i (photon-elisp-vector-dot (aref matrix i) vec))
      (setq i (1+ i)))
    result))

(defun photon-tensor-zeros-vector (shape)
  (if (null shape)
      0.0
    (let* ((count (car shape))
           (rest (cdr shape))
           (result (make-vector count nil))
           (i 0))
      (while (< i count)
        (aset result i (photon-tensor-zeros-vector rest))
        (setq i (1+ i)))
      result)))

(defun photon-tensor-map-vector (fn tensor)
  (if (vectorp tensor)
      (let* ((size (length tensor))
             (result (make-vector size nil))
             (i 0))
        (while (< i size)
          (aset result i (photon-tensor-map-vector fn (aref tensor i)))
          (setq i (1+ i)))
        result)
    (funcall fn tensor)))

(defun photon-tensor-map2-vector (fn left right)
  (if (and (vectorp left) (vectorp right))
      (let* ((size (length left))
             (result (make-vector size nil))
             (i 0))
        (while (< i size)
          (aset result i
                (photon-tensor-map2-vector fn (aref left i) (aref right i)))
          (setq i (1+ i)))
        result)
    (funcall fn left right)))

(defun photon-tensor-add-vector (left right)
  (photon-tensor-map2-vector '+ left right))

(defun photon-tensor-scale-vector (scale tensor)
  (photon-tensor-map-vector (lambda (value) (* scale value)) tensor))

(defun photon-make-elisp-tensor-backend ()
  (list (cons 'name 'elisp-tensor)
        (cons 'features '(vector matrix tensor elementwise fingerprint))
        (cons 'add 'photon-elisp-vector-add)
        (cons 'mul 'photon-elisp-vector-mul)
        (cons 'matvec 'photon-tensor-matrix-vector-dot)
        (cons 'matrix-from-rows 'photon-tensor-matrix-from-rows)
        (cons 'matrix-shape 'photon-tensor-matrix-shape)
        (cons 'tensor-shape 'photon-list-tensor-shape)
        (cons 'tensor-add 'photon-tensor-add-vector)
        (cons 'tensor-scale 'photon-tensor-scale-vector)
        (cons 'tensor-map 'photon-tensor-map-vector)
        (cons 'tensor-zeros 'photon-tensor-zeros-vector)
        (cons 'scale 'photon-elisp-vector-scale)
        (cons 'dot 'photon-elisp-vector-dot)
        (cons 'squash 'photon-elisp-vector-squash)
        (cons 'from-list 'photon-elisp-vector-from-list)
        (cons 'to-list 'photon-elisp-vector-to-list)
        (cons 'fingerprint 'photon-elisp-vector-fingerprint)
        (cons 'zeros 'photon-elisp-vector-zeros)))

(defvar photon-vector-backend (photon-make-list-vector-backend))

(defun photon-make-external-vector-backend (name features operations)
  (append (list (cons 'name name)
                (cons 'features features))
          operations))

(defun photon-set-vector-backend (backend)
  (setq photon-vector-backend backend))

(defun photon-use-list-vector-backend ()
  (photon-set-vector-backend (photon-make-list-vector-backend)))

(defun photon-use-elisp-vector-backend ()
  (photon-set-vector-backend (photon-make-elisp-vector-backend)))

(defun photon-use-elisp-tensor-backend ()
  (photon-set-vector-backend (photon-make-elisp-tensor-backend)))

(defun photon-use-vector-backend-name (name)
  (cond
   ((or (eq name 'list-vector) (equal name "list-vector"))
    (photon-use-list-vector-backend))
   ((or (eq name 'elisp-vector) (equal name "elisp-vector"))
    (photon-use-elisp-vector-backend))
   ((or (eq name 'elisp-tensor) (equal name "elisp-tensor"))
    (photon-use-elisp-tensor-backend))
   (t (error "Unknown photon vector backend: %S" name))))

(defun photon-vector-backend-name ()
  (cdr (assq 'name photon-vector-backend)))

(defun photon-vector-backend-features ()
  (let ((cell (assq 'features photon-vector-backend)))
    (if cell
        (cdr cell)
      nil)))

(defun photon-vector-backend-supports-p (feature)
  (if (memq feature (photon-vector-backend-features))
      t
    nil))

(defun photon-vector-backend-op (op)
  (cdr (assq op photon-vector-backend)))

(defun photon-vector-backend-op-or (op fallback)
  (let ((fn (photon-vector-backend-op op)))
    (if fn fn fallback)))

(defun photon-vector-add (left right)
  (funcall (photon-vector-backend-op 'add) left right))

(defun photon-vector-mul (left right)
  (funcall (photon-vector-backend-op 'mul) left right))

(defun photon-vector-scale (scale vec)
  (funcall (photon-vector-backend-op 'scale) scale vec))

(defun photon-vector-dot (left right)
  (funcall (photon-vector-backend-op 'dot) left right))

(defun photon-vector-squash (vec)
  (funcall (photon-vector-backend-op 'squash) vec))

(defun photon-vector-from-list (items)
  (funcall (photon-vector-backend-op 'from-list) items))

(defun photon-vector-to-list (vec)
  (funcall (photon-vector-backend-op 'to-list) vec))

(defun photon-vector-fingerprint (vec)
  (let ((op (photon-vector-backend-op 'fingerprint)))
    (if op
        (funcall op vec)
      (photon-list-vector-fingerprint (photon-vector-to-list vec)))))

(defun photon-vector-zeros (size)
  (funcall (photon-vector-backend-op 'zeros) size))

(defun photon-matrix-vector-dot (matrix vec)
  (funcall (photon-vector-backend-op 'matvec) matrix vec))

(defun photon-vector-length (vec)
  (length vec))

(defun photon-matrix-rows (matrix)
  (length matrix))

(defun photon-matrix-cols (matrix)
  (if matrix
      (length (car matrix))
    0))

(defun photon-matrix-shape (matrix)
  (funcall (photon-vector-backend-op-or
            'matrix-shape 'photon-list-matrix-shape)
           matrix))

(defun photon-matrix-from-rows (rows)
  (funcall (photon-vector-backend-op-or
            'matrix-from-rows 'photon-list-matrix-from-rows)
           rows))

(defun photon-tensor-shape (tensor)
  (funcall (photon-vector-backend-op-or
            'tensor-shape 'photon-list-tensor-shape)
           tensor))

(defun photon-tensor-add (left right)
  (funcall (photon-vector-backend-op-or
            'tensor-add 'photon-list-tensor-add)
           left right))

(defun photon-tensor-scale (scale tensor)
  (funcall (photon-vector-backend-op-or
            'tensor-scale 'photon-list-tensor-scale)
           scale tensor))

(defun photon-tensor-map (fn tensor)
  (funcall (photon-vector-backend-op-or
            'tensor-map 'photon-list-tensor-map)
           fn tensor))

(defun photon-tensor-zeros (shape)
  (funcall (photon-vector-backend-op-or
            'tensor-zeros 'photon-list-tensor-zeros)
           shape))

(defun photon-matrix-zeros (rows cols)
  (let ((matrix nil)
        (i 0))
    (while (< i rows)
      (setq matrix (cons (photon-vector-zeros cols) matrix))
      (setq i (1+ i)))
    (nreverse matrix)))

(provide 'photon-vector)

;;; photon-vector.el ends here
