"""Phase-8 end-to-end CLI integration tests.

Exercises ``arabic_pdf_transcribe.cli.main`` against the in-tree
fixtures with the validator forced into ``accept`` (native branch
only, no ML model load). The tests assert exit codes, output file
shape, and progress-stream content.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from arabic_pdf_transcribe.cli import (
    EXIT_CORRUPTED_OR_FORMAT,
    EXIT_ENCRYPTED,
    EXIT_OK,
    main,
)
from arabic_pdf_transcribe.errors import EncryptedPDFError
from arabic_pdf_transcribe.validate.native_validator import (
    ValidationResult,
    ValidatorConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
CLEAN = FIXTURES / "digital-clean" / "lorem-ar-real.pdf"


def _force_native_accept() -> object:
    def _accept(page: object, config: ValidatorConfig) -> ValidationResult:
        return ValidationResult(accept=True, signals={}, reasons=())

    return _accept


def _patched_main(argv: list[str]) -> int:
    """Run ``main`` with the validator forced to accept every page.

    Patching at the pipeline boundary keeps the CLI surface unchanged
    — the test models a clean digital input.
    """
    accept = _force_native_accept()
    real_transcribe = __import__(
        "arabic_pdf_transcribe.pipeline", fromlist=["transcribe"]
    ).transcribe

    def _wrapped(*args: object, **kwargs: object) -> object:
        kwargs["validator"] = accept  # type: ignore[index]
        return real_transcribe(*args, **kwargs)

    with patch("arabic_pdf_transcribe.cli.transcribe", new=_wrapped):
        return main(argv)


# ---------------------------------------------------------------------------
# Exit code 0 — happy paths
# ---------------------------------------------------------------------------


def test_md_output_to_file_returns_zero(tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    rc = _patched_main([str(CLEAN), "-o", str(out), "--quiet"])
    assert rc == EXIT_OK
    assert out.exists()
    assert out.stat().st_size > 0


def test_md_output_to_stdout_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _patched_main([str(CLEAN), "--quiet"])
    captured = capsys.readouterr()
    assert rc == EXIT_OK
    assert len(captured.out) > 0


def test_docx_output_to_file_returns_zero(tmp_path: Path) -> None:
    out = tmp_path / "out.docx"
    rc = _patched_main([str(CLEAN), "-o", str(out), "--quiet"])
    assert rc == EXIT_OK
    assert out.exists()
    # Word files are zip archives starting with PK\x03\x04.
    head = out.read_bytes()[:4]
    assert head == b"PK\x03\x04"


# ---------------------------------------------------------------------------
# --debug-json sidecar
# ---------------------------------------------------------------------------


def test_debug_json_sidecar_written(tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    sidecar = tmp_path / "debug.json"
    rc = _patched_main(
        [
            str(CLEAN),
            "-o",
            str(out),
            "--debug-json",
            str(sidecar),
            "--quiet",
        ]
    )
    assert rc == EXIT_OK
    assert sidecar.exists()
    import json

    data = json.loads(sidecar.read_text())
    assert "regions" in data
    assert "n_pages" in data
    assert data["n_pages"] >= 1


# ---------------------------------------------------------------------------
# Progress streams
# ---------------------------------------------------------------------------


def test_text_progress_writes_to_stderr(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.md"
    _patched_main([str(CLEAN), "-o", str(out)])
    captured = capsys.readouterr()
    assert "page" in captured.err.lower() or "summary" in captured.err.lower()


def test_quiet_suppresses_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.md"
    _patched_main([str(CLEAN), "-o", str(out), "--quiet"])
    captured = capsys.readouterr()
    assert captured.err == ""


def test_json_logs_emits_json_lines(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "out.md"
    _patched_main([str(CLEAN), "-o", str(out), "--json-logs"])
    captured = capsys.readouterr()
    import json

    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert lines
    for line in lines:
        json.loads(line)  # raises on invalid JSON


# ---------------------------------------------------------------------------
# Encrypted / corrupted handling — exit 3 / 4
# ---------------------------------------------------------------------------


def test_encrypted_pdf_returns_exit_3(tmp_path: Path) -> None:
    """Spec scenario 12. Patches the loader to raise EncryptedPDFError."""
    out = tmp_path / "out.md"

    def _boom(*args: object, **kwargs: object) -> object:
        raise EncryptedPDFError("encrypted")

    with patch("arabic_pdf_transcribe.cli.transcribe", new=_boom):
        rc = main([str(CLEAN), "-o", str(out), "--quiet"])
    assert rc == EXIT_ENCRYPTED


def test_corrupted_pdf_returns_exit_4(tmp_path: Path) -> None:
    """Spec scenario 13: corrupted PDF → exit 4."""
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf at all")
    out = tmp_path / "out.md"
    rc = main([str(bogus), "-o", str(out), "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


# ---------------------------------------------------------------------------
# --pages filter on a real PDF
# ---------------------------------------------------------------------------


def test_pages_filter_limits_output(tmp_path: Path) -> None:
    out = tmp_path / "out.md"
    rc = _patched_main([str(CLEAN), "-o", str(out), "--pages", "1", "--quiet"])
    assert rc == EXIT_OK
    assert out.exists()


def test_model_download_error_returns_exit_5(tmp_path: Path) -> None:
    """Spec: ModelDownloadError → exit 5 with hint."""
    from arabic_pdf_transcribe.cli import EXIT_MODEL_MISSING
    from arabic_pdf_transcribe.errors import ModelDownloadError

    out = tmp_path / "out.md"

    def _boom(*args: object, **kwargs: object) -> object:
        raise ModelDownloadError(
            "model X@rev Y not in cache; run huggingface-cli download X --revision Y"
        )

    with patch("arabic_pdf_transcribe.cli.transcribe", new=_boom):
        rc = main([str(CLEAN), "-o", str(out), "--quiet"])
    assert rc == EXIT_MODEL_MISSING


def test_format_docx_without_output_returns_exit_4(tmp_path: Path) -> None:
    """Codex feedback: ``--format docx`` to stdout is rejected up-front
    in validate_args, not later in the output writer."""
    rc = main([str(CLEAN), "--format", "docx", "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT
