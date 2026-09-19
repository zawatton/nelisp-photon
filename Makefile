NELISP ?= ../nelisp/target/nelisp
EMACS ?= emacs

.PHONY: test smoke nelisp-smoke train text-train file-train stream-train stream-diagnostics wordpiece-train vector-train minibatch-train batch-train serious-train embedding-train params-train gain-train projection-train cache-train benchmark generation-benchmark large-readout-smoke-benchmark large-readout-mid-benchmark large-readout-mid-ablation large-readout-benchmark large-generation-benchmark large-streaming-eval photon-cli-train photon-cli-generate tensor-benchmark save-load

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

large-readout-smoke-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/large-readout-smoke-benchmark.el

large-readout-mid-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/large-readout-mid-benchmark.el

large-readout-mid-ablation:
	$(EMACS) -Q --batch -L lisp --script examples/large-readout-mid-ablation.el

large-readout-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/large-readout-benchmark.el

large-generation-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/large-generation-benchmark.el

large-streaming-eval:
	$(NELISP) --load examples/large-streaming-eval.el

photon-cli-train:
	$(NELISP) --eval '(progn (setq photon-cli-command "train") (load "examples/photon-cli.el"))'

photon-cli-generate:
	$(NELISP) --eval '(progn (setq photon-cli-command "generate" photon-cli-model-path "target/photon-large-model.el" photon-cli-prompt "photon stream" photon-cli-steps "32") (load "examples/photon-cli.el"))'

tensor-benchmark:
	$(EMACS) -Q --batch -L lisp --script examples/tensor-benchmark.el

save-load:
	$(EMACS) -Q --batch -L lisp --script examples/save-load.el
