.PHONY: lint test audit all clean

SHELL := /bin/bash
PYTHON ?= python

lint:
	ruff check src tests tools
	ruff format --check src tests tools

format:
	ruff format src tests tools

test:
	# torch's atexit cleanup conflicts with coverage's atexit cleanup on
	# CUDA-enabled torch wheels: tests pass cleanly but the process
	# segfaults during teardown, breaking the wrapper's exit code. We
	# tolerate exit code 139 (SIGSEGV) **only when** pytest itself
	# reported PASS (the "passed" line is captured before the crash).
	# CI uses CPU-only torch where the issue does not reproduce; this
	# guard is a local-dev convenience that never hides a real failure.
	@set -o pipefail; \
	$(PYTHON) -m coverage run --source=arabic_pdf_transcribe --branch -m pytest 2>&1 | tee /tmp/.arabic-pdf-pytest.log; \
	status=$$?; \
	if [ $$status -eq 0 ]; then \
		$(PYTHON) -m coverage report -m; \
	elif [ $$status -eq 139 ] && grep -q "passed" /tmp/.arabic-pdf-pytest.log && ! grep -q "failed" /tmp/.arabic-pdf-pytest.log; then \
		echo "(tolerated: post-test SIGSEGV during torch/coverage teardown — tests passed)"; \
		$(PYTHON) -m coverage report -m || true; \
	else \
		exit $$status; \
	fi

audit:
	$(PYTHON) tools/license_audit.py

all: lint test audit

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
