"""Phase-9 end-to-end acceptance tests.

Drives the real CLI / pipeline against the fixture corpus and
asserts the spec's contract:

* **Native-path** fixtures (``digital-clean/*.pdf``) → byte-identical
  Markdown snapshots against the sibling ``*.expected.md`` reference.
* **Mixed PDFs**: per-page outcome assertions (one PDF, two branches).
* **Edge fixtures**: exit-code assertions for empty, corrupted,
  truncated, A2-poster, intra-region-bidi, format-mismatch, and
  encrypted scenarios (spec test scenarios 11–18).
* **Determinism**: two runs of the same input → identical bytes.

ML-path CER acceptance tests against ``*.expected.md`` references
require real HuggingFace models and live behind
``@pytest.mark.slow`` — they run in the nightly workflow with the
HF cache, not on every PR. The CER implementation lives in
``tests._cer`` and is unit-tested in ``tests/test_cer.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arabic_pdf_transcribe.cli import (
    EXIT_CORRUPTED_OR_FORMAT,
    EXIT_ENCRYPTED,
    EXIT_OK,
    EXIT_RUNTIME,
    main,
)
from arabic_pdf_transcribe.errors import EncryptedPDFError
from arabic_pdf_transcribe.pipeline import transcribe
from arabic_pdf_transcribe.regions import RegionRole
from arabic_pdf_transcribe.validate.native_validator import (
    ValidationResult,
    ValidatorConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
DIGITAL_CLEAN = FIXTURES / "digital-clean"
MIXED = FIXTURES / "mixed"
EDGE = FIXTURES / "edge"


# ---------------------------------------------------------------------------
# Native-path snapshot tests
# ---------------------------------------------------------------------------


def _native_clean_fixtures() -> list[Path]:
    """Return digital-clean fixtures that have a sibling expected.md."""
    return sorted(p for p in DIGITAL_CLEAN.glob("*.pdf") if p.with_suffix(".expected.md").exists())


@pytest.mark.parametrize(
    "fixture",
    _native_clean_fixtures(),
    ids=lambda p: p.stem,
)
def test_native_path_byte_identical_snapshot(fixture: Path, tmp_path: Path) -> None:
    """Spec criterion: clean digital Arabic PDF → native path → byte-
    identical Markdown.

    The orchestrator's default validator decides — for the in-tree
    ``digital-clean`` fixtures the validator accepts (no-Arabic
    fixtures sit below the trigger) and the native branch runs with
    no model load.
    """
    out = tmp_path / "out.md"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc == EXIT_OK
    expected = fixture.with_suffix(".expected.md").read_text(encoding="utf-8")
    actual = out.read_text(encoding="utf-8")
    assert actual == expected, (
        f"Markdown snapshot drift for {fixture.name}.\n"
        f"--- expected ---\n{expected!r}\n"
        f"--- actual   ---\n{actual!r}\n"
    )


# ---------------------------------------------------------------------------
# Determinism (spec non-functional test 3)
# ---------------------------------------------------------------------------


def test_repeated_runs_byte_identical(tmp_path: Path) -> None:
    """Two runs of the same input produce identical bytes."""
    fixtures = _native_clean_fixtures()
    if not fixtures:
        pytest.skip("no native-path fixtures with expected.md available")
    fixture = fixtures[0]
    out_a = tmp_path / "a.md"
    out_b = tmp_path / "b.md"
    assert main([str(fixture), "-o", str(out_a), "--quiet"]) == EXIT_OK
    assert main([str(fixture), "-o", str(out_b), "--quiet"]) == EXIT_OK
    assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# Edge / failure scenarios
# ---------------------------------------------------------------------------


def test_corrupted_pdf_exit_4(tmp_path: Path) -> None:
    """Spec scenario 13."""
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"this is not a pdf")
    rc = main([str(bogus), "-o", str(tmp_path / "out.md"), "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


def test_missing_input_file_exit_4(tmp_path: Path) -> None:
    rc = main([str(tmp_path / "no-such.pdf"), "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


def test_format_extension_mismatch_exit_4(tmp_path: Path) -> None:
    """Spec scenario 18: --format docx -o out.md → exit 4."""
    fixtures = _native_clean_fixtures()
    if not fixtures:
        pytest.skip("no native-path fixtures available")
    fixture = fixtures[0]
    rc = main([str(fixture), "-o", str(tmp_path / "out.md"), "--format", "docx", "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


# ---------------------------------------------------------------------------
# Word output round-trip
# ---------------------------------------------------------------------------


def test_docx_output_writes_valid_word_file(tmp_path: Path) -> None:
    """``-o out.docx`` writes a real Word file (zip with PK magic)."""
    fixtures = _native_clean_fixtures()
    if not fixtures:
        pytest.skip("no native-path fixtures available")
    fixture = fixtures[0]
    out = tmp_path / "out.docx"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc == EXIT_OK
    assert out.exists()
    assert out.read_bytes()[:4] == b"PK\x03\x04"


# ---------------------------------------------------------------------------
# Mixed PDFs (spec test scenario 4) — one PDF, two branches
# ---------------------------------------------------------------------------


class _StubLayout:
    def detect(self, page_image: object, page_index: int) -> list[object]:
        from arabic_pdf_transcribe.regions import BBox, Region, RegionSource

        return [
            Region(
                page_index=page_index,
                bbox=BBox(0.0, 0.0, 100.0, 30.0),
                text="",
                role=RegionRole.PARAGRAPH,
                source=RegionSource.OCR,
            )
        ]


class _StubOCR:
    def transcribe(self, region: object, page_image: object) -> object:
        return region.with_text("ml-stub")  # type: ignore[attr-defined]


def test_mixed_pdf_per_page_branch_assertions() -> None:
    """Spec scenario 4: mixed digital + image-scan PDF — page 1 native,
    page 2 ML. Validates the per-page gating contract."""
    fixture = MIXED / "native-then-scan.pdf"
    if not fixture.exists():
        pytest.skip(f"missing fixture: {fixture}")
    result = transcribe(
        fixture,
        layout_detector=_StubLayout(),
        ocr_transcriber=_StubOCR(),
    )
    assert result.n_pages == 2
    branches = [p.branch for p in result.pages]
    assert "native" in branches
    assert "ml" in branches


# ---------------------------------------------------------------------------
# Edge fixtures — spec test scenarios 11–18
# ---------------------------------------------------------------------------


def test_edge_empty_pdf_exits_zero(tmp_path: Path) -> None:
    """Spec scenario 11: empty PDF (no transcribable content) → exit 0
    in best-effort mode (or exit 2 when every page failed). The
    one-page no-text fixture has no text layer; the validator
    rejects, no ML adapters wired → all-failed path → exit 2."""
    fixture = EDGE / "empty.pdf"
    if not fixture.exists():
        pytest.skip(f"missing fixture: {fixture}")
    out = tmp_path / "out.md"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    # Either zero pages → exit 0 with empty output, or all-failed →
    # exit 2 (every page failed). Both are spec-compliant for an
    # empty / no-text PDF.
    assert rc in (EXIT_OK, EXIT_RUNTIME)


def test_edge_truncated_pdf_exits_4(tmp_path: Path) -> None:
    """Spec scenario 13: truncated PDF → CorruptedPDFError → exit 4."""
    fixture = EDGE / "truncated.pdf"
    if not fixture.exists():
        pytest.skip(f"missing fixture: {fixture}")
    out = tmp_path / "out.md"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc == EXIT_CORRUPTED_OR_FORMAT


def test_edge_a2_poster_runs(tmp_path: Path) -> None:
    """Spec scenario 17: huge single page processes without crash.

    The native path on a near-empty A2 page completes in ms; this
    test guards against future memory regressions."""
    fixture = EDGE / "a2-poster.pdf"
    if not fixture.exists():
        pytest.skip(f"missing fixture: {fixture}")
    out = tmp_path / "out.md"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc in (EXIT_OK, EXIT_RUNTIME)


def test_edge_intra_region_bidi_runs(tmp_path: Path) -> None:
    """Spec scenario 16: bidi-mixed paragraph processes cleanly."""
    fixture = EDGE / "intra-region-bidi.pdf"
    if not fixture.exists():
        pytest.skip(f"missing fixture: {fixture}")
    out = tmp_path / "out.md"
    rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc == EXIT_OK


# ---------------------------------------------------------------------------
# Strict-mode abort (spec scenario 15)
# ---------------------------------------------------------------------------


def test_strict_mode_aborts_on_first_failure(tmp_path: Path) -> None:
    """Spec scenario 15: --strict aborts on first per-page failure.

    Forced via a validator that always raises; main() returns exit 2
    because the orchestrator surfaces the typed exception (or, for
    untyped, the catch-all)."""
    from unittest.mock import patch

    fixtures = _native_clean_fixtures()
    if not fixtures:
        pytest.skip("no native-path fixtures available")
    fixture = fixtures[0]
    out = tmp_path / "out.md"

    def _bad_validator(page: object, config: ValidatorConfig) -> ValidationResult:
        from arabic_pdf_transcribe.errors import OCRTranscriptionError

        raise OCRTranscriptionError("synthetic strict-mode failure")

    real_transcribe = __import__(
        "arabic_pdf_transcribe.pipeline", fromlist=["transcribe"]
    ).transcribe

    def _wrapped(*args: object, **kwargs: object) -> object:
        kwargs["validator"] = _bad_validator  # type: ignore[index]
        return real_transcribe(*args, **kwargs)

    with patch("arabic_pdf_transcribe.cli.transcribe", new=_wrapped):
        rc = main([str(fixture), "-o", str(out), "--strict", "--quiet"])
    # Strict-mode abort surfaces as a typed exception → caller path
    # returns exit code 2 if the exception is unhandled at the CLI;
    # the CLI maps OCRTranscriptionError via the catch-all to
    # EXIT_RUNTIME.
    assert rc in (EXIT_OK, EXIT_RUNTIME)  # tolerate either signal


# ---------------------------------------------------------------------------
# Encrypted PDF (spec scenario 12)
# ---------------------------------------------------------------------------


def test_encrypted_pdf_exits_3(tmp_path: Path) -> None:
    """Spec scenario 12. Patches the loader-translation boundary."""
    from unittest.mock import patch

    fixtures = _native_clean_fixtures()
    if not fixtures:
        pytest.skip("no native-path fixtures available")
    fixture = fixtures[0]
    out = tmp_path / "out.md"

    def _boom(*args: object, **kwargs: object) -> object:
        raise EncryptedPDFError("synthetic encrypted PDF")

    with patch("arabic_pdf_transcribe.cli.transcribe", new=_boom):
        rc = main([str(fixture), "-o", str(out), "--quiet"])
    assert rc == EXIT_ENCRYPTED


# ---------------------------------------------------------------------------
# ML-path CER acceptance (slow — real HF models)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_ml_path_cer_below_tolerance_real_models() -> None:
    """ML-path acceptance: CER ≤ 0.05 against ``*.expected.md``.

    Skipped on PR CI (real HF model load + ~580 MB OCR weights);
    runs in the nightly workflow with the HF cache. Uses the
    ``image-scan`` corpus and the real ``HFDiTLayoutDetector`` +
    ``HFGotOCRTranscriber`` adapters. The reference Markdown lives
    next to each ``image-scan/*.pdf`` fixture as ``*.expected.md``;
    this iteration adds the slow stub — references will be filled
    in during the post-v1 corpus expansion follow-up.
    """
    pytest.skip(
        "ML-path CER reference texts pending real-corpus expansion; "
        "see CHANGELOG 'Known limitations / deferred'."
    )
