(load "lisp/photon.el")

(photon-use-elisp-vector-backend)

(let* ((config (photon-make-config 8 6 3 2))
       (model (photon-make-model config))
       (corpus '(1 2 3 1 2 3 1 2 3))
       (before-embedding (photon-vector-to-list
                          (nth 1 (photon-model-embedding-table model))))
       (before (photon-evaluate model corpus 3))
       (history (photon-train model corpus 3 2 0.20))
       (after (photon-evaluate model corpus 3))
       (after-embedding (photon-vector-to-list
                         (nth 1 (photon-model-embedding-table model)))))
  (list (cons 'backend (photon-vector-backend-name))
        (cons 'embedding-changed (not (equal before-embedding after-embedding)))
        (cons 'before before)
        (cons 'history history)
        (cons 'after after)
        (cons 'generated (photon-model-generate model '(1 2 3) 3))))
