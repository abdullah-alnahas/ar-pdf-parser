"""Phase-1 skeleton smoke tests.

These tests must hold for every later phase as well — in particular the
lazy-import assertion: importing :mod:`arabic_pdf_transcribe` must not pull in
``transformers``, ``torch``, ``Pillow``, or ``huggingface_hub``. Architect
emphasised this guarantee at plan-approval time, so the test lives here from
phase 1 onward.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

import arabic_pdf_transcribe


def test_package_imports() -> None:
    assert arabic_pdf_transcribe.__version__
    assert isinstance(arabic_pdf_transcribe.__version__, str)


def test_version_module_consistent() -> None:
    from arabic_pdf_transcribe import _version

    assert _version.__version__ == arabic_pdf_transcribe.__version__


def test_regions_module_exposes_schema_version() -> None:
    from arabic_pdf_transcribe import regions

    assert regions.REGION_SCHEMA_VERSION == "1"


def test_errors_module_has_base_and_phase4_5_exceptions() -> None:
    from arabic_pdf_transcribe import errors

    assert issubclass(errors.ModelDownloadError, errors.ArabicPdfTranscribeError)
    assert issubclass(errors.OCRTranscriptionError, errors.ArabicPdfTranscribeError)


def test_cli_main_is_callable() -> None:
    """Phase 8 replaced the phase-1 placeholder. The entry point now
    drives the full pipeline; ``--help`` short-circuits with exit 0
    via argparse's ``SystemExit``."""
    import pytest

    from arabic_pdf_transcribe import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Lazy-import guarantee.
# ---------------------------------------------------------------------------
# A cold subprocess imports only the top-level package and reports its
# ``sys.modules`` keyset. None of the heavy ML libraries should appear.
HEAVY_MODULES = ("transformers", "torch", "huggingface_hub", "PIL")


def test_top_level_import_does_not_load_heavy_deps() -> None:
    code = (
        "import sys, json, arabic_pdf_transcribe\n" "print(json.dumps([m for m in sys.modules]))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    loaded = set(json.loads(proc.stdout))
    leaked = [m for m in HEAVY_MODULES if any(k == m or k.startswith(m + ".") for k in loaded)]
    assert leaked == [], f"top-level import leaked heavy modules: {leaked}"


def test_package_reload_is_safe() -> None:
    importlib.reload(arabic_pdf_transcribe)
    assert arabic_pdf_transcribe.__version__
