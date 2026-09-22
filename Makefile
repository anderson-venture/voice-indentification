# Development entry points. PY38 lets you point at a specific interpreter.
PY   ?= .venv/bin/python
PY38 ?= python3.8

.PHONY: help venv install install-dev install-export fixtures test lint fmt typecheck check models clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

venv:  ## Create the Python 3.8 virtualenv
	$(PY38) -m venv .venv
	$(PY) -m pip install -U pip

install:  ## Runtime dependencies
	$(PY) -m pip install -r requirements.txt

install-dev:  ## Runtime + test/lint/type tools
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

install-export:  ## PyTorch etc, ONLY for re-exporting or fine-tuning a model (~2 GB)
	$(PY) -m pip install -r requirements-export.txt --extra-index-url https://download.pytorch.org/whl/cpu

fixtures:  ## Regenerate the synthetic test audio
	PYTHONPATH=src:. $(PY) -m scripts.make_fixtures

test:  ## Run the suite (needs no network)
	PYTHONPATH=src $(PY) -m pytest

lint:  ## Lint, including the Python 3.8 syntax constraint
	$(PY) -m ruff check src scripts tests

fmt:  ## Format
	$(PY) -m black src scripts tests

typecheck:  ## mypy --strict on src/
	$(PY) -m mypy

check: lint typecheck test  ## What CI runs

models:  ## Checksum models into models/MANIFEST.json
	PYTHONPATH=src:. $(PY) -m scripts.fetch_models

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache
