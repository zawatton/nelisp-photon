NELISP ?= ../nelisp/target/nelisp
EMACS ?= emacs

.PHONY: test smoke nelisp-smoke train text-train file-train stream-train stream-diagnostics wordpiece-train vector-train minibatch-train batch-train serious-train embedding-train params-train gain-train projection-train cache-train benchmark generation-benchmark tensor-benchmark save-load

test:
	$(EMACS) -Q --batch -L lisp -l test/photon-test.el -f ert-run-tests-batch-and-exit

smoke: nelisp-smoke

nelisp-smoke:
	$(NELISP) --load examples/run.el

train:
	$(NELISP) --load examples/train.el

text-train:
	$(NELISP) --load examples/text-train.el

file-train:
	$(NELISP) --load examples/file-train.el

stream-train:
	$(EMACS) -Q --batch -L lisp --script examples/stream-train.el

stream-diagnostics:
	$(EMACS) -Q --batch -L lisp --script examples/stream-diagnostics.el

wordpiece-train:
	$(EMACS) -Q --batch -L lisp --script examples/wordpiece-train.el

vector-train:
	$(NELISP) --load examples/vector-train.el

minibatch-train:
	$(NELISP) --load examples/minibatch-train.el

batch-train:
	$(EMACS) -Q --batch -L lisp --script examples/batch-train.el

serious-train:
	$(EMACS) -Q --batch -L lisp --script examples/serious-train.el

embedding-train:
	$(NELISP) --load examples/embedding-train.el

params-train:
	$(NELISP) --load examples/params-train.el

gain-train:
	$(NELISP) --load examples/gain-train.el

projection-train:
	$(NELISP) --load examples/projection-train.el

cache-train:
	$(NELISP) --load examples/cache-train.el

benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/benchmark.el

generation-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/generation-benchmark.el

tensor-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/tensor-benchmark.el

save-load:
	$(EMACS) -Q --batch -L lisp --script examples/save-load.el
