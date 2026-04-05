PYTHON ?= $(shell if command -v mise >/dev/null 2>&1; then mise which python 2>/dev/null || command -v python3; else command -v python3; fi)
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python
TEST_ENV_STAMP := $(VENV)/.test-deps.installed
NOTEBOOK_ENV_STAMP := $(VENV)/.notebook-deps.installed
CLEAN_DIRS := artifacts data/cache data/processed runtime mlruns htmlcov build dist pip-wheel-metadata .pytest_cache .mypy_cache .ruff_cache .hypothesis .jupyter
.DEFAULT_GOAL := help

.PHONY: help ensure-venv venv-bootstrap venv install-notebook install-test test test-agent-llm test-report-template test-report-llm demo-list demo-scenario clean clean-all clean-venv

SCENARIO ?= promote_no_champion

help: ## Show available Make targets
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-22s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

ensure-venv:
	@DESIRED_PYTHON="$(PYTHON)"; \
	DESIRED_VERSION="$$( "$$DESIRED_PYTHON" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' )"; \
	CURRENT_VERSION="$$( [ -x "$(VENV_PYTHON)" ] && "$(VENV_PYTHON)" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo missing )"; \
	if [ "$$CURRENT_VERSION" != "$$DESIRED_VERSION" ]; then \
		rm -rf "$(VENV)"; \
		"$$DESIRED_PYTHON" -m venv "$(VENV)"; \
	fi

venv-bootstrap: ensure-venv
	@$(VENV_PYTHON) -c "import pip, setuptools, wheel" >/dev/null 2>&1 || \
		$(VENV_PYTHON) -m pip install --upgrade pip setuptools wheel

venv: venv-bootstrap ## Create an empty local virtual environment using the pinned Python when available

$(NOTEBOOK_ENV_STAMP): pyproject.toml Makefile | venv-bootstrap
	$(VENV_PYTHON) -m pip install --no-build-isolation -e '.[agent,notebook]'
	touch $(NOTEBOOK_ENV_STAMP)

install-notebook: $(NOTEBOOK_ENV_STAMP) ## Install package + notebook dependencies into .venv

$(TEST_ENV_STAMP): pyproject.toml Makefile | venv-bootstrap
	$(VENV_PYTHON) -m pip install --no-build-isolation -e '.[agent,dev]'
	touch $(TEST_ENV_STAMP)

install-test: $(TEST_ENV_STAMP) ## Install package + test dependencies into .venv

test: $(TEST_ENV_STAMP) ## Run the core local suite (force-skips opt-in LLM suites)
	RUN_LLM_EXPLAINER_TESTS= RUN_LLM_REPORT_QUALITY_TESTS= $(VENV_PYTHON) -m unittest discover -s tests -p 'test_*.py'

test-agent-llm: $(TEST_ENV_STAMP) ## Run the opt-in live explainer-agent smoke test
	RUN_LLM_EXPLAINER_TESTS=1 $(VENV_PYTHON) -m pytest tests/test_explainer_live.py -v

test-report-template: $(TEST_ENV_STAMP) ## Run report template structure checks only
	$(VENV_PYTHON) -m pytest tests/test_report_quality.py::TestTemplateReportStructure -v

test-report-llm: $(TEST_ENV_STAMP) ## Run opt-in LLM-backed report-quality checks
	RUN_LLM_REPORT_QUALITY_TESTS=1 $(VENV_PYTHON) -m pytest tests/test_report_quality.py -v -m requires_llm

demo-list: $(TEST_ENV_STAMP) ## List replayable demo scenarios
	$(VENV_PYTHON) -m hdb_resale_mlops.demo --list

demo-scenario: $(TEST_ENV_STAMP) ## Generate a demo review packet for SCENARIO
	$(VENV_PYTHON) -m hdb_resale_mlops.demo --scenario $(SCENARIO) --output-dir artifacts/demo/$(SCENARIO)

clean: ## Remove generated runtime state, local MLflow data, caches, and build/test artifacts
	rm -rf $(CLEAN_DIRS)
	find . \
		-path './.git' -prune -o \
		-path './$(VENV)' -prune -o \
		-type d \( -name __pycache__ -o -name .ipynb_checkpoints -o -name '*.egg-info' \) \
		-exec rm -rf {} +
	find . \
		-path './.git' -prune -o \
		-path './$(VENV)' -prune -o \
		-type f \( \
			-name '.coverage' -o \
			-name '.coverage.*' -o \
			-name '*.db' -o \
			-name '*.db-*' -o \
			-name '*.sqlite' -o \
			-name '*.sqlite-*' -o \
			-name '*.sqlite3' -o \
			-name '*.sqlite3-*' -o \
			-name '*.pyc' -o \
			-name '*.pyo' \
		) -delete
	-rmdir data artifacts runtime 2>/dev/null || true

clean-all: clean clean-venv ## Remove all generated state, including the local virtual environment

clean-venv: ## Remove the local virtual environment
	rm -rf $(VENV)
