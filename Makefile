.PHONY: lint test audit all clean

PYTHON ?= python

lint:
	ruff check src tests tools
	ruff format --check src tests tools

format:
	ruff format src tests tools

test:
	$(PYTHON) -m pytest --cov=arabic_pdf_transcribe --cov-report=term-missing

audit:
	$(PYTHON) tools/license_audit.py

all: lint test audit

clean:
	rm -rf build dist *.egg-info .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} +
