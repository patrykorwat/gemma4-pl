# Convenience targets. Most real work runs via SLURM on Helios,
# these are for local dev, data prep, and quick interactive checks.

PY ?= python
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYBIN := $(VENV)/bin/python

.PHONY: help
help:
	@echo "Targets:"
	@echo "  setup          create local venv and install in editable mode with dev extras"
	@echo "  fmt            run ruff format and lint fix"
	@echo "  lint           run ruff check without fixing"
	@echo "  test           run pytest"
	@echo "  download-model dry run the base model download (local)"
	@echo "  prep-sft       prepare SFT data from raw sources (local dry run)"
	@echo "  clean          remove caches and build artifacts"

.PHONY: setup
setup:
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip wheel
	$(PIP) install -e ".[dev]"

.PHONY: fmt
fmt:
	$(PYBIN) -m ruff format src tests scripts
	$(PYBIN) -m ruff check --fix src tests scripts

.PHONY: lint
lint:
	$(PYBIN) -m ruff check src tests scripts

.PHONY: test
test:
	$(PYBIN) -m pytest -q

.PHONY: download-model
download-model:
	$(PYBIN) scripts/download_base_model.py --dry-run

.PHONY: prep-sft
prep-sft:
	$(PYBIN) scripts/prepare_sft_data.py --dry-run

.PHONY: clean
clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
