# Agentic AI Support & Knowledge Research System
#
# The venv lives OUTSIDE this project directory by default. This project sits
# on an external/exFAT-style drive, where macOS AppleDouble ("._") sidecar
# files corrupt wheel installs (`make setup` would silently produce a venv with
# no packages in it - python present, uvicorn/pytest/etc missing). Override
# with VENV=... if you want it elsewhere, e.g. VENV=.venv to use this folder.
VENV ?= $(HOME)/.cache/agentic-support-system/venv
PY   := $(VENV)/bin/python
UV   ?= uv

# Copy instead of hardlink: the cache and an external project drive are usually
# on different filesystems.
export UV_LINK_MODE := copy
export COPYFILE_DISABLE := 1

.DEFAULT_GOAL := help
.PHONY: help setup install test test-unit test-integration lint format typecheck \
        check demo eval run migrate ingest clean

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install all dependencies
	$(UV) venv --python 3.12 "$(VENV)" || $(UV) venv "$(VENV)"
	$(UV) pip install --python "$(PY)" -e ".[dev,web]"
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")
	@"$(PY)" -c "import uvicorn, pytest" && echo "Setup OK: $(VENV)"

install: ## Reinstall dependencies into an existing virtualenv
	$(UV) pip install --python "$(PY)" -e ".[dev,web]"

install-postgres: ## Add the PostgreSQL + pgvector extras
	$(UV) pip install --python "$(PY)" -e ".[dev,web,postgres]"

test: ## Run the full test suite
	$(PY) -m pytest -q

test-unit: ## Run unit tests only
	$(PY) -m pytest tests/unit -q

test-integration: ## Run integration tests only
	$(PY) -m pytest tests/integration -q -m integration

lint: ## Lint with ruff
	$(PY) -m ruff check app tests

format: ## Auto-format and auto-fix
	$(PY) -m ruff format app tests
	$(PY) -m ruff check --fix app tests

typecheck: ## Static type check with mypy
	$(PY) -m mypy app

check: lint typecheck test ## Lint, typecheck and test

demo: ## Run the five specification test cases end to end
	$(PY) -m scripts.demo

eval: ## Run the evaluation suite and print a scorecard
	$(PY) -m app.evaluation.runner

ingest: ## Ingest the seed documents in data/documents
	$(PY) -m scripts.ingest data/documents

migrate: ## Apply database migrations
	$(PY) -m scripts.migrate

run: ## Start the API server with reload
	$(PY) -m uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache *.egg-info
