"""Phase-8 pipeline orchestrator tests.

Stub :class:`LayoutDetector` and :class:`OCRTranscriber` are injected so
the ML branch runs without loading any model. Real PDFs come from the
in-tree fixtures in ``tests/fixtures/pdfs/``.

Coverage targets (per plan acceptance criteria):

* Best-effort default vs ``strict``.
* All-failed pages → ``TranscribeResult.all_failed``.
* Failure-region synthesis (one ``RegionRole.FAILURE_PLACEHOLDER`` per
  failed page).
* Temp-dir cleanup on success and failure.
* ``pages`` filter skips work for pages outside the set.
"""

from __future__ import annotations

import gc
from pathlib import Path

import pytest

from arabic_pdf_transcribe.errors import OCRTranscriptionError
from arabic_pdf_transcribe.pipeline import TranscribeResult, transcribe
from arabic_pdf_transcribe.regions import (
    BBox,
    Region,
    RegionRole,
    RegionSource,
)
from arabic_pdf_transcribe.validate.native_validator import (
    ValidationResult,
    ValidatorConfig,
)

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
CLEAN = FIXTURES / "digital-clean" / "lorem-ar-real.pdf"
BROKEN = FIXTURES / "digital-broken" / "mojibake.pdf"
SCAN = FIXTURES / "image-scan" / "scan-ar-1col.pdf"


# ---------------------------------------------------------------------------
# Stub adapters
# ---------------------------------------------------------------------------


class _StubLayoutDetector:
    """One paragraph region per page. No model load."""

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, page_image: object, page_index: int) -> list[Region]:
        self.calls += 1
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
    """Fill ``text`` with a fixed string so we can assert on output."""

    def __init__(self, text: str = "stubbed") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, region: Region, page_image: object) -> Region:
        self.calls += 1
        return region.with_text(self.text)


class _FailingOCR:
    """Raises on the configured ``fail_on_page`` (1-based)."""

    def __init__(self, *, fail_on_page: int) -> None:
        self.fail_on_page = fail_on_page

    def transcribe(self, region: Region, page_image: object) -> Region:
        if region.page_index + 1 == self.fail_on_page:
            raise OCRTranscriptionError(f"stub failure on page {self.fail_on_page}")
        return region.with_text("ok")


def _accept_validator(page: object, config: ValidatorConfig) -> ValidationResult:
    """Validator that always accepts → keeps the native branch."""
    return ValidationResult(accept=True, signals={}, reasons=())


def _reject_validator(page: object, config: ValidatorConfig) -> ValidationResult:
    """Validator that always rejects → forces the ML branch."""
    return ValidationResult(accept=False, signals={}, reasons=("forced_ml",))


# ---------------------------------------------------------------------------
# Native branch
# ---------------------------------------------------------------------------


def test_native_branch_no_ml_calls() -> None:
    """Spec criterion: clean digital Arabic PDF takes native path,
    no model load. Verified by the stubs being uncalled."""
    layout = _StubLayoutDetector()
    ocr = _StubOCR()
    result = transcribe(
        CLEAN,
        layout_detector=layout,
        ocr_transcriber=ocr,
        validator=_accept_validator,
    )
    assert layout.calls == 0
    assert ocr.calls == 0
    assert result.n_pages >= 1
    assert all(p.branch == "native" for p in result.pages)
    assert result.failed_pages == 0


def test_native_branch_omits_ml_adapters_entirely() -> None:
    """No ML adapters needed when validator accepts every page."""
    result = transcribe(CLEAN, validator=_accept_validator)
    assert result.n_pages >= 1
    assert all(p.branch == "native" for p in result.pages)


# ---------------------------------------------------------------------------
# ML branch
# ---------------------------------------------------------------------------


def test_ml_branch_runs_when_validator_rejects() -> None:
    layout = _StubLayoutDetector()
    ocr = _StubOCR(text="ml-result")
    result = transcribe(
        CLEAN,
        layout_detector=layout,
        ocr_transcriber=ocr,
        validator=_reject_validator,
    )
    assert layout.calls >= 1
    assert ocr.calls >= 1
    assert all(p.branch == "ml" for p in result.pages)
    assert any("ml-result" in r.text for r in result.regions)


def test_ml_branch_without_adapters_synthesises_failure() -> None:
    """When validator rejects but no adapters wired, default mode
    synthesises a failure-placeholder region per page."""
    result = transcribe(CLEAN, validator=_reject_validator)
    assert all(p.branch == "failed" for p in result.pages)
    assert all(p.regions[0].role is RegionRole.FAILURE_PLACEHOLDER for p in result.pages)


# ---------------------------------------------------------------------------
# Strict / best-effort
# ---------------------------------------------------------------------------


def test_strict_mode_aborts_on_first_failure() -> None:
    layout = _StubLayoutDetector()
    ocr = _FailingOCR(fail_on_page=1)
    with pytest.raises(OCRTranscriptionError):
        transcribe(
            CLEAN,
            layout_detector=layout,
            ocr_transcriber=ocr,
            validator=_reject_validator,
            strict=True,
        )


def test_best_effort_inserts_failure_placeholder() -> None:
    layout = _StubLayoutDetector()
    ocr = _FailingOCR(fail_on_page=1)
    result = transcribe(
        CLEAN,
        layout_detector=layout,
        ocr_transcriber=ocr,
        validator=_reject_validator,
        strict=False,
    )
    failed = [p for p in result.pages if p.branch == "failed"]
    assert len(failed) >= 1
    assert all(p.regions[0].role is RegionRole.FAILURE_PLACEHOLDER for p in failed)
    assert all(p.regions[0].failure_reason == "OCRTranscriptionError" for p in failed)


def test_all_failed_property_set_when_every_page_fails() -> None:
    result = transcribe(CLEAN, validator=_reject_validator)
    assert result.all_failed is True


# ---------------------------------------------------------------------------
# Page filter
# ---------------------------------------------------------------------------


def test_pages_filter_skips_unselected() -> None:
    layout = _StubLayoutDetector()
    ocr = _StubOCR()
    result = transcribe(
        CLEAN,
        layout_detector=layout,
        ocr_transcriber=ocr,
        validator=_accept_validator,
        pages=(0,),
    )
    assert result.n_pages == 1
    assert result.pages[0].page_index == 0


def test_pages_filter_does_not_invoke_validator_for_skipped_pages() -> None:
    """Codex feedback: pages filter must short-circuit before
    per-page work. The validator is the cheapest observable hook,
    so we assert it's only called for selected pages."""
    seen: list[int] = []

    def _spy_validator(page: object, config: ValidatorConfig) -> ValidationResult:
        # ``page`` is a NativePage; record its index.
        seen.append(getattr(page, "page_index", -1))
        return ValidationResult(accept=True, signals={}, reasons=())

    transcribe(CLEAN, validator=_spy_validator, pages=(0,))
    assert seen == [0], f"validator was invoked for unselected pages: {seen}"


# ---------------------------------------------------------------------------
# max_workers
# ---------------------------------------------------------------------------


def test_max_workers_accepted_in_signature() -> None:
    """Codex feedback: ``--max-workers`` flows through to transcribe()."""
    result = transcribe(CLEAN, validator=_accept_validator, max_workers=4)
    assert result.n_pages >= 1


def test_max_workers_invalid_raises_value_error() -> None:
    with pytest.raises(ValueError, match="max_workers must be >= 1"):
        transcribe(CLEAN, validator=_accept_validator, max_workers=0)


# ---------------------------------------------------------------------------
# CUDA OOM detection (covers RuntimeError-shaped torch.cuda.OutOfMemoryError)
# ---------------------------------------------------------------------------


class _CudaOOMOCR:
    """Stub that raises a synthetic ``torch.cuda.OutOfMemoryError``."""

    def transcribe(self, region: Region, page_image: object) -> Region:
        # Build a class whose ``__module__`` and ``__name__`` match
        # what ``_is_cuda_oom`` checks for, without needing torch
        # installed.
        class _SyntheticCudaOOM(RuntimeError):
            pass

        _SyntheticCudaOOM.__module__ = "torch.cuda"
        _SyntheticCudaOOM.__name__ = "OutOfMemoryError"
        raise _SyntheticCudaOOM("CUDA out of memory")


def test_cuda_oom_strict_raises_typed_oom() -> None:
    from arabic_pdf_transcribe.errors import OutOfMemoryDuringInference

    layout = _StubLayoutDetector()
    with pytest.raises(OutOfMemoryDuringInference, match="CUDA OOM"):
        transcribe(
            CLEAN,
            layout_detector=layout,
            ocr_transcriber=_CudaOOMOCR(),
            validator=_reject_validator,
            strict=True,
        )


def test_cuda_oom_best_effort_synthesises_failure() -> None:
    layout = _StubLayoutDetector()
    result = transcribe(
        CLEAN,
        layout_detector=layout,
        ocr_transcriber=_CudaOOMOCR(),
        validator=_reject_validator,
    )
    assert all(p.branch == "failed" for p in result.pages)
    assert all(p.failure_reason == "cuda_out_of_memory" for p in result.pages)


# ---------------------------------------------------------------------------
# Encrypted / corrupted PDFs surface from the loader
# ---------------------------------------------------------------------------


def test_corrupted_pdf_propagates(tmp_path: Path) -> None:
    """Spec scenario 13: corrupted PDF → CorruptedPDFError (CLI maps to 4)."""
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"not a pdf")
    from arabic_pdf_transcribe.errors import CorruptedPDFError

    with pytest.raises(CorruptedPDFError):
        transcribe(bogus, validator=_accept_validator)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_transcribe_byte_identical_regions() -> None:
    """Native-path reproducibility (spec)."""
    a = transcribe(CLEAN, validator=_accept_validator)
    b = transcribe(CLEAN, validator=_accept_validator)
    assert _serialise(a) == _serialise(b)


def _serialise(result: TranscribeResult) -> str:
    return "\n".join(r.to_json() for r in result.regions)


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def test_progress_callback_receives_start_and_complete_events() -> None:
    events: list[tuple[int, int, str]] = []

    def cb(page_index: int, total: int, event: str) -> None:
        events.append((page_index, total, event))

    transcribe(CLEAN, validator=_accept_validator, progress=cb)
    assert any(e[2] == "start" for e in events)
    assert any(e[2].startswith("complete:") for e in events)


# ---------------------------------------------------------------------------
# Temp-dir cleanup (acceptance criterion)
# ---------------------------------------------------------------------------


def test_temp_dir_cleaned_up_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``tempfile.TemporaryDirectory`` removes the dir on context exit."""
    seen: list[str] = []

    import tempfile

    real_cls = tempfile.TemporaryDirectory

    class _Trackable(real_cls):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            seen.append(self.name)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _Trackable)
    transcribe(CLEAN, validator=_accept_validator)
    gc.collect()
    assert seen, "TemporaryDirectory should have been instantiated"
    for d in seen:
        assert not Path(d).exists(), f"temp dir leaked: {d}"


def test_temp_dir_cleaned_up_on_strict_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    import tempfile

    real_cls = tempfile.TemporaryDirectory

    class _Trackable(real_cls):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            seen.append(self.name)

    monkeypatch.setattr(tempfile, "TemporaryDirectory", _Trackable)
    layout = _StubLayoutDetector()
    ocr = _FailingOCR(fail_on_page=1)
    with pytest.raises(OCRTranscriptionError):
        transcribe(
            CLEAN,
            layout_detector=layout,
            ocr_transcriber=ocr,
            validator=_reject_validator,
            strict=True,
        )
    for d in seen:
        assert not Path(d).exists(), f"temp dir leaked on strict failure: {d}"
