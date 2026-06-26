(load "lisp/photon.el")

(photon-use-elisp-vector-backend)

(let* ((config (photon-make-config 8 6 3 2))
       (model (photon-make-model config))
       (corpus '(1 2 3 1 2 3 1 2 3))
       (before-gain (photon-vector-to-list
                     (photon-model-gain-get model 'chunk-gain)))
       (before (photon-evaluate model corpus 3))
       (history (photon-train model corpus 3 2 0.20))
       (after (photon-evaluate model corpus 3))
       (after-gain (photon-vector-to-list
                    (photon-model-gain-get model 'chunk-gain))))
  (list (cons 'backend (photon-vector-backend-name))
        (cons 'gain-changed (not (equal before-gain after-gain)))
        (cons 'before before)
        (cons 'history history)
        (cons 'after after)
        (cons 'gain-count (length after-gain))
        (cons 'generated (photon-model-generate model '(1 2 3) 3))))
