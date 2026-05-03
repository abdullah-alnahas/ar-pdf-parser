"""End-to-end pipeline orchestrator.

Wires every prior phase into one ``transcribe(pdf_path, ...)`` entry
point:

1. **Open** the PDF (encrypted / corrupted PDFs raise typed errors).
2. **Per page**:
   a. **Native extract** + per-page **validate**.
   b. If validation accepts → take the native regions.
   c. Else → **rasterise** → **layout-detect** → **per-region OCR**.
   d. **Reorder** + **classify roles**.
   e. On any caught failure, synthesise a single
      :data:`RegionRole.FAILURE_PLACEHOLDER` region for the page so
      the emitter still produces output for that page slot.
3. **Yield** a :class:`PageOutcome` per page; collect into
   :class:`TranscribeResult`.

The orchestrator depends on ``LayoutDetector`` and ``OCRTranscriber``
*Protocols* — production wiring uses the HF adapters, tests inject
stubs. The validator, the rasteriser, the reorderer, the role
classifier, and the emitters are pluggable too (callable kwargs with
sensible defaults).

A process-scoped :class:`tempfile.TemporaryDirectory` is created on
entry and removed on exit (success or failure). Page rasterisation
keeps images in memory by default; the temp dir is reserved for
adapters that need a path-shaped artefact.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from arabic_pdf_transcribe.errors import (
    ArabicPdfTranscribeError,
    OutOfMemoryDuringInference,
)
from arabic_pdf_transcribe.extract.native import (
    NativePage,
    extract_native_from_document,
)
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource
from arabic_pdf_transcribe.roles.classify import ClassifyConfig, classify_page
from arabic_pdf_transcribe.validate.native_validator import (
    ValidationResult,
    ValidatorConfig,
    validate_page,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from arabic_pdf_transcribe.layout import LayoutDetector
    from arabic_pdf_transcribe.ocr import OCRTranscriber

# Reorder is a small pure helper — re-exported through the package.
from arabic_pdf_transcribe.order.reorder import reorder as _reorder_default

DEFAULT_DPI = 200


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PageOutcome:
    """Per-page record: which branch ran and what came out."""

    page_index: int
    branch: str  # "native" | "ml" | "failed"
    regions: tuple[Region, ...]
    validation: ValidationResult | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class TranscribeResult:
    """Aggregate result returned by :func:`transcribe`."""

    pages: tuple[PageOutcome, ...]
    regions: tuple[Region, ...]  # flattened across pages, in reading order

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def ok_pages(self) -> int:
        return sum(1 for p in self.pages if p.branch != "failed")

    @property
    def failed_pages(self) -> int:
        return sum(1 for p in self.pages if p.branch == "failed")

    @property
    def all_failed(self) -> bool:
        return self.n_pages > 0 and self.ok_pages == 0


# ---------------------------------------------------------------------------
# Page-rendering helper (lazy pypdfium2 + Pillow imports)
# ---------------------------------------------------------------------------


def _rasterise_page_from_document(document: object, page_index: int, *, dpi: int) -> PILImage:
    """Rasterise ``page_index`` at ``dpi`` using a shared document handle.

    The render uses :mod:`arabic_pdf_transcribe.layout._rasterise` which
    pulls Pillow lazily (gated behind the ``[ml]`` extra). The orchestrator
    owns the document lifecycle; this helper only borrows the handle to
    fetch a page, render it, and close that page.
    """
    from arabic_pdf_transcribe.layout._rasterise import rasterise_page

    page = document[page_index]  # type: ignore[index]
    try:
        return rasterise_page(page, dpi=dpi)
    finally:
        page.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


ProgressCallback = Callable[[int, int, str], None]
"""``(page_index, total_pages, event)`` — best-effort, fire-and-forget.

The orchestrator calls this for ``"start"``, ``"complete"``, and
``"failure"`` events on each page. Callbacks must not raise; the
orchestrator does not catch.
"""


def transcribe(
    pdf_path: Path | str,
    *,
    layout_detector: LayoutDetector | None = None,
    ocr_transcriber: OCRTranscriber | None = None,
    validator: Callable[[NativePage, ValidatorConfig], ValidationResult] | None = None,
    validator_config: ValidatorConfig | None = None,
    classify_config: ClassifyConfig | None = None,
    reorder_fn: Callable[..., list[Region]] | None = None,
    rtl: bool = True,
    dpi: int = DEFAULT_DPI,
    pages: Iterable[int] | None = None,
    strict: bool = False,
    max_workers: int = 1,
    progress: ProgressCallback | None = None,
) -> TranscribeResult:
    """Transcribe ``pdf_path`` and return a :class:`TranscribeResult`.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF. Encrypted / corrupted PDFs surface as
        :class:`EncryptedPDFError` / :class:`CorruptedPDFError` from
        the loader; the caller maps these to CLI exit codes.
    layout_detector:
        Optional :class:`LayoutDetector` for the ML branch. When
        ``None`` and the ML branch is needed, the orchestrator raises
        :class:`RuntimeError`. The CLI wires the production HF
        adapter; tests inject a stub.
    ocr_transcriber:
        Optional :class:`OCRTranscriber` — same pattern as
        ``layout_detector``.
    validator:
        Pluggable per-page validator. Defaults to
        :func:`validate_page`.
    validator_config:
        Optional :class:`ValidatorConfig` override.
    classify_config:
        Optional :class:`ClassifyConfig` override.
    reorder_fn:
        Pluggable reorderer. Defaults to
        :func:`arabic_pdf_transcribe.order.reorder.reorder`.
    rtl:
        Pass through to ``reorder_fn`` and ``classify_page``. Default
        ``True`` for Arabic-first usage.
    dpi:
        Rasterisation DPI for the ML branch. Default 200.
    pages:
        Optional iterable of *0-based* page indices to process; pages
        outside the set are skipped before any per-page work — the
        loader is opened once for the page count, then native
        extraction, validation, and the ML branch run only for
        selected pages.
    strict:
        When ``True``, the first per-page failure aborts the run by
        re-raising the underlying exception. When ``False`` (default,
        best-effort mode), failures synthesise a placeholder region
        and the pipeline continues.
    max_workers:
        Reserved for future per-page parallelism on the ML branch
        (default ``1``, sequential). v1 keeps the loop sequential to
        bound peak RSS — the parameter is plumbed through so the
        CLI surface is stable; phase 9 will enable parallel ML
        runs after benchmarking.
    progress:
        Optional ``(page_index, total_pages, event)`` callback for
        progress reporting.
    """
    pdf_path = Path(pdf_path)
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    selected = _normalise_page_filter(pages)
    actual_validator = validator or _default_validator
    cfg = validator_config or ValidatorConfig()
    reorder_call = reorder_fn or _reorder_default
    classify_cfg = classify_config or ClassifyConfig(rtl=rtl)

    pages_out: list[PageOutcome] = []
    flat_regions: list[Region] = []

    # Lazy-imported here so phase 9's TOML-config loader can stub the
    # loader for unit tests without a full pypdfium2 install.
    from arabic_pdf_transcribe.pdf._pypdfium2_loader import open_pdf

    with _process_temp_dir() as _tmp, open_pdf(pdf_path) as document:
        # Single document handle: page count, native extraction, and
        # ML-branch rasterisation all share this one ``pypdfium2``
        # instance. The loader translates encrypted/corrupted errors
        # at the boundary, so those exit codes fire here before any
        # extraction work.
        total = len(document)
        for native_page in extract_native_from_document(document, pages=selected):
            outcome = _process_page(
                document=document,
                native_page=native_page,
                total=total,
                validator=actual_validator,
                validator_config=cfg,
                layout_detector=layout_detector,
                ocr_transcriber=ocr_transcriber,
                reorder_call=reorder_call,
                classify_cfg=classify_cfg,
                rtl=rtl,
                dpi=dpi,
                strict=strict,
                progress=progress,
            )
            pages_out.append(outcome)
            flat_regions.extend(outcome.regions)

    return TranscribeResult(pages=tuple(pages_out), regions=tuple(flat_regions))


# ---------------------------------------------------------------------------
# Per-page processing
# ---------------------------------------------------------------------------


def _process_page(
    *,
    document: object,
    native_page: NativePage,
    total: int,
    validator: Callable[[NativePage, ValidatorConfig], ValidationResult],
    validator_config: ValidatorConfig,
    layout_detector: LayoutDetector | None,
    ocr_transcriber: OCRTranscriber | None,
    reorder_call: Callable[..., list[Region]],
    classify_cfg: ClassifyConfig,
    rtl: bool,
    dpi: int,
    strict: bool,
    progress: ProgressCallback | None,
) -> PageOutcome:
    page_index = native_page.page_index
    _emit_progress(progress, page_index, total, "start")
    # Track which branch was active when an exception fires so the
    # failure-placeholder region carries the correct ``RegionSource``
    # (NATIVE for validator/native-extract failures; OCR for ML-branch
    # failures). Mutated as the page progresses.
    branch_state = {"branch": "native"}
    try:
        result = validator(native_page, validator_config)
        if result.accept:
            regions = _post_process_regions(
                native_page.regions,
                page_width=native_page.page_width,
                page_height=native_page.page_height,
                reorder_call=reorder_call,
                classify_cfg=classify_cfg,
                rtl=rtl,
            )
            _emit_progress(progress, page_index, total, "complete:native")
            return PageOutcome(
                page_index=page_index,
                branch="native",
                regions=tuple(regions),
                validation=result,
            )
        # ML fallback.
        branch_state["branch"] = "ml"
        regions = _run_ml_branch(
            document=document,
            native_page=native_page,
            total=total,
            layout_detector=layout_detector,
            ocr_transcriber=ocr_transcriber,
            reorder_call=reorder_call,
            classify_cfg=classify_cfg,
            rtl=rtl,
            dpi=dpi,
            progress=progress,
        )
        _emit_progress(progress, page_index, total, "complete:ml")
        return PageOutcome(
            page_index=page_index,
            branch="ml",
            regions=tuple(regions),
            validation=result,
        )
    except MemoryError as exc:
        reason = "out_of_memory"
        if strict:
            raise OutOfMemoryDuringInference(
                f"page {page_index + 1}: {exc}; reduce --max-workers or rasterisation DPI"
            ) from exc
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch_state["branch"])
    except ArabicPdfTranscribeError as exc:
        # Include ``str(exc)`` in the reason so JSON-log consumers see
        # the actionable hint baked into the exception (e.g. the
        # "prefetch the weights with: arabic-pdf-transcribe
        # --prefetch-models" message on ModelDownloadError). Previous
        # behaviour reported only the class name, hiding the hint.
        # Issue #16 root cause #3.
        reason = f"{type(exc).__name__}:{exc}"
        if strict:
            raise
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch_state["branch"])
    except RuntimeError as exc:
        # ``torch.cuda.OutOfMemoryError`` subclasses RuntimeError —
        # detect by class name + module so we don't have to import
        # torch (it may not be installed). Other RuntimeErrors are
        # treated as generic per-page failures below.
        if _is_cuda_oom(exc):
            if strict:
                raise OutOfMemoryDuringInference(
                    f"page {page_index + 1}: CUDA OOM: {exc}; "
                    f"reduce --max-workers or rasterisation DPI"
                ) from exc
            _emit_progress(progress, page_index, total, "failure:cuda_out_of_memory")
            return _failure_outcome(
                page_index, native_page, "cuda_out_of_memory", branch_state["branch"]
            )
        reason = f"{type(exc).__name__}:{exc}"
        if strict:
            raise
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch_state["branch"])
    except Exception as exc:
        # Bare-except trade-off: this is the per-page boundary for the
        # spec's "best-effort default" contract — if an upstream stage
        # raises an unexpected exception we MUST NOT abort the whole
        # run; a single failed page becomes a placeholder so the rest
        # of the document still transcribes. Typed exceptions
        # (Memory/ArabicPdfTranscribe/RuntimeError-CUDA-OOM) are
        # handled in dedicated arms above and surface specific
        # ``failure_reason`` strings; this catch-all only fires for
        # genuinely unexpected errors and records the type+message
        # verbatim. With ``--strict`` the original exception is
        # re-raised, preserving the traceback.
        reason = f"{type(exc).__name__}:{exc}"
        if strict:
            raise
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch_state["branch"])


def _is_cuda_oom(exc: BaseException) -> bool:
    cls = type(exc)
    name = cls.__name__
    module = (cls.__module__ or "").split(".", 1)[0]
    return name == "OutOfMemoryError" and module == "torch"


def _run_ml_branch(
    *,
    document: object,
    native_page: NativePage,
    total: int,
    layout_detector: LayoutDetector | None,
    ocr_transcriber: OCRTranscriber | None,
    reorder_call: Callable[..., list[Region]],
    classify_cfg: ClassifyConfig,
    rtl: bool,
    dpi: int,
    progress: ProgressCallback | None,
) -> list[Region]:
    if layout_detector is None or ocr_transcriber is None:
        raise RuntimeError(
            "ML branch needed for page "
            f"{native_page.page_index + 1} but no layout_detector / ocr_transcriber wired"
        )
    page_index = native_page.page_index
    _emit_progress(progress, page_index, total, "layout")
    page_image = _rasterise_page_from_document(document, page_index, dpi=dpi)
    detected = list(layout_detector.detect(page_image, page_index))
    transcribed: list[Region] = []
    # Issue #18 RC#2: emit a per-region progress event before each
    # OCR call so a long CPU run shows visible progress instead of
    # appearing to hang. Encoded into the event string so the
    # ``ProgressCallback`` signature stays stable.
    n_regions = len(detected)
    for idx, region in enumerate(detected, start=1):
        role_name = region.role.value
        _emit_progress(progress, page_index, total, f"region:{idx}/{n_regions}:{role_name}")
        if region.role is RegionRole.FIGURE:
            transcribed.append(region)
            continue
        transcribed.append(ocr_transcriber.transcribe(region, page_image))
    # ML-branch bboxes are in pixel coords on the rasterised image, so the
    # page dimensions handed to the post-processor must match (otherwise
    # full-page regions land in the header/footer band by mistake).
    pixel_width, pixel_height = page_image.size
    return _post_process_regions(
        transcribed,
        page_width=float(pixel_width),
        page_height=float(pixel_height),
        reorder_call=reorder_call,
        classify_cfg=classify_cfg,
        rtl=rtl,
    )


def _post_process_regions(
    regions: Sequence[Region],
    *,
    page_width: float,
    page_height: float,
    reorder_call: Callable[..., list[Region]],
    classify_cfg: ClassifyConfig,
    rtl: bool,
) -> list[Region]:
    """Run reorder + role classification."""
    if not regions:
        return []
    reordered = reorder_call(regions, page_width, page_height, rtl=rtl)
    return classify_page(reordered, page_width, page_height, config=classify_cfg)


def _failure_outcome(
    page_index: int,
    native_page: NativePage,
    reason: str,
    branch: str,
) -> PageOutcome:
    """Synthesise a single-region failure placeholder for the page.

    ``branch`` records which stage was running when the exception
    fired (``"native"`` or ``"ml"``); the resulting placeholder
    region's :class:`RegionSource` reflects that so downstream tools
    know whether the failure happened in the deterministic native
    path or the ML branch.
    """
    source = RegionSource.OCR if branch == "ml" else RegionSource.NATIVE
    placeholder = Region.as_failure_placeholder(
        reason=reason,
        page_index=page_index,
        bbox=BBox(0.0, 0.0, native_page.page_width, native_page.page_height),
        source=source,
    )
    return PageOutcome(
        page_index=page_index,
        branch="failed",
        regions=(placeholder,),
        failure_reason=reason,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_validator(page: NativePage, config: ValidatorConfig) -> ValidationResult:
    return validate_page(page, config=config)


def _normalise_page_filter(pages: Iterable[int] | None) -> set[int] | None:
    if pages is None:
        return None
    return {int(p) for p in pages}


def _emit_progress(
    progress: ProgressCallback | None,
    page_index: int,
    total: int,
    event: str,
) -> None:
    if progress is None:
        return
    progress(page_index, total, event)


@contextmanager
def _process_temp_dir() -> Iterator[Path]:
    """Process-scoped temp dir; removed on context exit (always)."""
    with tempfile.TemporaryDirectory(prefix="arabic-pdf-transcribe-") as tmp:
        yield Path(tmp)


__all__ = [
    "DEFAULT_DPI",
    "PageOutcome",
    "ProgressCallback",
    "TranscribeResult",
    "transcribe",
]
