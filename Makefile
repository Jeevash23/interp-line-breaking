# Line-breaking in Pythia-410M: code and data release.
# `make help` lists the targets. PYTHON defaults to the environment that `make setup` (uv sync) creates.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
export OPENBLAS_NUM_THREADS ?= 2
export OMP_NUM_THREADS ?= 2
export MKL_NUM_THREADS ?= 2
.NOTPARALLEL:

PAPER := paper/v6-2026-11-10

.PHONY: help setup setup-experiment python-check unpack attachments check-freezes check tables figures pdf models

help:
	@echo "make setup             create .venv with the pinned environment (uv sync; see ENVIRONMENT.md)"
	@echo "make check             CPU only, about 8 minutes, no model and no network: unpack the saved outputs,"
	@echo "                       check the freeze records, and run all 712 checks of the paper's numbers against the outputs"
	@echo "make check-freezes     only the freeze records (FROZEN.md, xmodel/FROZEN.md)"
	@echo "make tables            rewrite the paper's tables from the saved outputs"
	@echo "make attachments       unpack the two release attachments (put them in attachments/ first)"
	@echo "make figures           draw the paper's four figures (Figure 4 needs exploration-activations.tar.xz)"
	@echo "make pdf               build the paper with latexmk (after make figures)"
	@echo "make setup-experiment  the environment for rerunning the experiments (adds transformers)"
	@echo "make models            download Pythia-410M at the pinned revision into the Hugging Face cache (0.91 GB);"
	@echo "                       python tools/cache_models.py --list shows every checkpoint the study used"

setup:
	uv sync --frozen

setup-experiment:
	uv sync --frozen --extra experiment

python-check:
	@$(PY) -c "import numpy, torch" 2>/dev/null || { echo "No usable Python at $(PYTHON): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py

attachments: python-check
	$(PY) tools/unpack_attachments.py

check-freezes: unpack
	$(PY) tools/check_frozen.py

check: unpack
	$(PY) tools/check_frozen.py
	cd $(PAPER) && $(PY) check_paper.py

tables: unpack
	cd $(PAPER) && $(PY) make_tables.py

figures: unpack attachments
	@[ -f outputs/exp03_pythia-410m_resid_L0-12.npy ] || { echo "Figure 4 needs exploration-activations.tar.xz in attachments/ (see README.md)"; exit 1; }
	cd $(PAPER) && $(PY) make_figures.py

pdf:
	@[ -f $(PAPER)/figures/fig-round5.pdf ] || { echo "run make figures first"; exit 1; }
	cd $(PAPER) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

models: python-check
	$(PY) tools/cache_models.py pythia-410m
