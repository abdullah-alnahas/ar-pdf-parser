"""End-to-end pipeline orchestrator (Strategy B, layout-then-OCR).

Wires every prior phase into one ``transcribe(pdf_path, ...)`` entry point
using a *staged* pipeline rather than per-page interleaving:

1. **Open** the PDF (encrypted / corrupted PDFs raise typed errors).
2. **Phase A — Validate**. Walk every page; native-accepted pages land in
   their final outcome immediately, ML-bound pages queue up.
3. **Phase B — Rasterise**. All ML-bound pages render to images in one
   pass (optionally parallel; ``pypdfium2`` is serialised under
   ``document_lock``).
4. **Phase C — Layout**. Run the layout detector across every ML page
   sequentially with a single model loaded.
5. **Phase D — OCR**. Run OCR across every ML page sequentially with a
   single model loaded — the layout model is already done by this point
   so VRAM contention is minimised.
6. **Phase E — Stitch**. Post-process (reorder + classify) per page and
   assemble :class:`PageOutcome` records back into source page-index
   order.

Design notes:

* Per-page failure isolation is preserved: a single page's failure in
  any phase becomes a :data:`RegionRole.FAILURE_PLACEHOLDER` outcome and
  the rest of the run continues. ``strict=True`` re-raises on first
  failure (test contract).
* ``max_workers`` parallelises only Phase B (rasterise). Layout and
  OCR run sequentially because GPU model calls serialise on CUDA
  anyway, and a sequential phase keeps a single model resident on the
  GPU at a time.
* The orchestrator depends on ``LayoutDetector`` and ``OCRTranscriber``
  *Protocols* — production wiring uses the HF adapters, tests inject
  stubs. Validator, rasteriser, reorderer, classifier, and emitters are
  pluggable callables.
* A process-scoped :class:`tempfile.TemporaryDirectory` is created on
  entry and removed on exit (success or failure).
"""

from __future__ import annotations

import tempfile
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
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


@dataclass(frozen=True, slots=True)
class _MLJob:
    """Per-page payload threading through Phases B → E."""

    native_page: NativePage
    validation: ValidationResult


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
    """Transcribe ``pdf_path`` with a staged layout-then-OCR pipeline.

    See module docstring for phase ordering. Per-page failure isolation
    is preserved across phases; ``strict=True`` re-raises on first
    failure.

    Parameters
    ----------
    pdf_path:
        Path to a local PDF. Encrypted / corrupted PDFs surface as
        :class:`EncryptedPDFError` / :class:`CorruptedPDFError` from
        the loader; the caller maps these to CLI exit codes.
    layout_detector:
        Optional :class:`LayoutDetector` for the ML branch. When
        ``None`` and the ML branch is needed, every ML-bound page
        records a failure outcome (or, with ``strict=True``, the run
        aborts on the first such page).
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
        Worker count for Phase B (rasterise). Layout and OCR run
        sequentially regardless because GPU model calls serialise on
        CUDA. Default ``1`` (fully sequential).
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

    # Lazy-imported here so phase 9's TOML-config loader can stub the
    # loader for unit tests without a full pypdfium2 install.
    from arabic_pdf_transcribe.pdf._pypdfium2_loader import open_pdf

    with _process_temp_dir() as _tmp, open_pdf(pdf_path) as document:
        total = len(document)
        # Native extraction iterates the document handle sequentially —
        # materialise once so downstream phases work from a stable
        # ordered list of pages and so a strict-mode failure during
        # extraction fires before any phase work begins.
        native_pages = list(extract_native_from_document(document, pages=selected))

        # Phase header events let renderers (rich progress bars) re-target
        # their per-phase task with the correct total before any work in
        # that phase fires. Format: ``phase:start:<name>:<count>``. Page
        # index is 0 (placeholder — phase events are not page-scoped).
        _emit_progress(progress, 0, total, f"phase:start:validate:{len(native_pages)}")

        # ----- Phase A: validate every page (native-accept inline, else queue ML)
        outcomes: dict[int, PageOutcome] = {}
        ml_jobs: list[_MLJob] = []
        for native_page in native_pages:
            outcome_or_job = _run_validate_phase(
                native_page=native_page,
                total=total,
                validator=actual_validator,
                validator_config=cfg,
                reorder_call=reorder_call,
                classify_cfg=classify_cfg,
                rtl=rtl,
                strict=strict,
                progress=progress,
            )
            if isinstance(outcome_or_job, PageOutcome):
                outcomes[native_page.page_index] = outcome_or_job
            else:
                ml_jobs.append(outcome_or_job)

        if ml_jobs:
            m = len(ml_jobs)
            _emit_progress(progress, 0, total, f"phase:start:rasterise:{m}")
            # ----- Phase B: rasterise all ML pages (parallel; doc_lock-serialised)
            page_images = _run_rasterise_phase(
                ml_jobs=ml_jobs,
                document=document,
                total=total,
                dpi=dpi,
                max_workers=max_workers,
                strict=strict,
                progress=progress,
                outcomes=outcomes,
            )

            _emit_progress(progress, 0, total, f"phase:start:layout:{m}")
            # ----- Phase C: layout sequentially across all ML pages
            page_regions = _run_layout_phase(
                ml_jobs=ml_jobs,
                page_images=page_images,
                total=total,
                layout_detector=layout_detector,
                ocr_transcriber=ocr_transcriber,
                strict=strict,
                progress=progress,
                outcomes=outcomes,
            )
            # Sequential actor lifecycle: the layout adapter is no
            # longer needed; releasing it frees its VRAM (Ray actor
            # mode kills the worker process, tearing down the CUDA
            # context) before Phase D loads the OCR model. On 6 GB
            # GPUs this is what keeps Phase D out of OOM territory.
            # No-op for in-process adapters that don't expose
            # ``release()``.
            _release_adapter(layout_detector)

            _emit_progress(progress, 0, total, f"phase:start:ocr:{m}")
            # ----- Phase D: OCR sequentially across all ML pages
            _run_ocr_phase(
                ml_jobs=ml_jobs,
                page_images=page_images,
                page_regions=page_regions,
                total=total,
                ocr_transcriber=ocr_transcriber,
                reorder_call=reorder_call,
                classify_cfg=classify_cfg,
                rtl=rtl,
                strict=strict,
                progress=progress,
                outcomes=outcomes,
            )
            _release_adapter(ocr_transcriber)

    # Phase E (stitch): assemble in source-page order.
    pages_out = [outcomes[np.page_index] for np in native_pages]
    flat_regions: list[Region] = [r for outcome in pages_out for r in outcome.regions]
    return TranscribeResult(pages=tuple(pages_out), regions=tuple(flat_regions))


# ---------------------------------------------------------------------------
# Phase A — validate (+ inline native post-process)
# ---------------------------------------------------------------------------


def _run_validate_phase(
    *,
    native_page: NativePage,
    total: int,
    validator: Callable[[NativePage, ValidatorConfig], ValidationResult],
    validator_config: ValidatorConfig,
    reorder_call: Callable[..., list[Region]],
    classify_cfg: ClassifyConfig,
    rtl: bool,
    strict: bool,
    progress: ProgressCallback | None,
) -> PageOutcome | _MLJob:
    """Validate one page.

    Returns either a finished :class:`PageOutcome` (native-accept or
    failure-during-validation) or an :class:`_MLJob` to be processed by
    the ML phases.
    """
    page_index = native_page.page_index
    _emit_progress(progress, page_index, total, "start")
    try:
        result = validator(native_page, validator_config)
    except BaseException as exc:  # narrowed in _handle_phase_exc
        return _handle_phase_exc(
            exc,
            page_index=page_index,
            native_page=native_page,
            total=total,
            branch="native",
            strict=strict,
            progress=progress,
        )
    if not result.accept:
        return _MLJob(native_page=native_page, validation=result)
    # Native accept → post-process inline.
    try:
        regions = _post_process_regions(
            native_page.regions,
            page_width=native_page.page_width,
            page_height=native_page.page_height,
            reorder_call=reorder_call,
            classify_cfg=classify_cfg,
            rtl=rtl,
        )
    except BaseException as exc:
        return _handle_phase_exc(
            exc,
            page_index=page_index,
            native_page=native_page,
            total=total,
            branch="native",
            strict=strict,
            progress=progress,
        )
    _emit_progress(progress, page_index, total, "complete:native")
    return PageOutcome(
        page_index=page_index,
        branch="native",
        regions=tuple(regions),
        validation=result,
    )


# ---------------------------------------------------------------------------
# Phase B — rasterise (parallel, doc_lock)
# ---------------------------------------------------------------------------


def _run_rasterise_phase(
    *,
    ml_jobs: Sequence[_MLJob],
    document: object,
    total: int,
    dpi: int,
    max_workers: int,
    strict: bool,
    progress: ProgressCallback | None,
    outcomes: dict[int, PageOutcome],
) -> dict[int, PILImage]:
    """Rasterise every ML page; failures land directly in ``outcomes``."""
    # ``pypdfium2`` is not thread-safe across page handles on the same
    # document; the lock serialises ``document[i]`` + render so multi-thread
    # rasterisation stays sound (the work is mostly I/O + Pillow encode).
    document_lock = threading.Lock()

    def _do_one(job: _MLJob) -> tuple[int, PILImage | PageOutcome]:
        page_index = job.native_page.page_index
        try:
            with document_lock:
                image = _rasterise_page_from_document(document, page_index, dpi=dpi)
        except BaseException as exc:
            return page_index, _handle_phase_exc(
                exc,
                page_index=page_index,
                native_page=job.native_page,
                total=total,
                branch="ml",
                strict=strict,
                progress=progress,
            )
        _emit_progress(progress, page_index, total, "rasterise")
        return page_index, image

    if max_workers > 1 and len(ml_jobs) > 1:
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="apt-ras"
        ) as pool:
            results = list(pool.map(_do_one, ml_jobs))
    else:
        results = [_do_one(job) for job in ml_jobs]

    page_images: dict[int, PILImage] = {}
    for page_index, result in results:
        if isinstance(result, PageOutcome):
            outcomes[page_index] = result
        else:
            page_images[page_index] = result
    return page_images


# ---------------------------------------------------------------------------
# Phase C — layout (sequential, single model loaded)
# ---------------------------------------------------------------------------


def _run_layout_phase(
    *,
    ml_jobs: Sequence[_MLJob],
    page_images: dict[int, PILImage],
    total: int,
    layout_detector: LayoutDetector | None,
    ocr_transcriber: OCRTranscriber | None,
    strict: bool,
    progress: ProgressCallback | None,
    outcomes: dict[int, PageOutcome],
) -> dict[int, list[Region]]:
    page_regions: dict[int, list[Region]] = {}
    for job in ml_jobs:
        page_index = job.native_page.page_index
        if page_index in outcomes:
            # Already failed in an earlier phase.
            continue
        if layout_detector is None or ocr_transcriber is None:
            exc = RuntimeError(
                f"ML branch needed for page {page_index + 1} but no "
                f"layout_detector / ocr_transcriber wired"
            )
            outcomes[page_index] = _handle_phase_exc(
                exc,
                page_index=page_index,
                native_page=job.native_page,
                total=total,
                branch="ml",
                strict=strict,
                progress=progress,
            )
            continue
        image = page_images[page_index]
        _emit_progress(progress, page_index, total, "layout")
        try:
            detected = list(layout_detector.detect(image, page_index))
        except BaseException as exc:
            outcomes[page_index] = _handle_phase_exc(
                exc,
                page_index=page_index,
                native_page=job.native_page,
                total=total,
                branch="ml",
                strict=strict,
                progress=progress,
            )
            continue
        # Issue #18 RC#2: emit a per-region progress event so the UI
        # shows progress within a long page run instead of appearing
        # to hang.
        n_regions = len(detected)
        for idx, region in enumerate(detected, start=1):
            _emit_progress(
                progress, page_index, total, f"region:{idx}/{n_regions}:{region.role.value}"
            )
        page_regions[page_index] = detected
    return page_regions


# ---------------------------------------------------------------------------
# Phase D — OCR (sequential, single model loaded)
# ---------------------------------------------------------------------------


def _run_ocr_phase(
    *,
    ml_jobs: Sequence[_MLJob],
    page_images: dict[int, PILImage],
    page_regions: dict[int, list[Region]],
    total: int,
    ocr_transcriber: OCRTranscriber | None,
    reorder_call: Callable[..., list[Region]],
    classify_cfg: ClassifyConfig,
    rtl: bool,
    strict: bool,
    progress: ProgressCallback | None,
    outcomes: dict[int, PageOutcome],
) -> None:
    if ocr_transcriber is None:
        # Layout phase already filled outcomes with adapter-missing failures.
        return
    # Adapter-level batching: when the OCR backend exposes
    # ``transcribe_page`` (surya, easyocr — see
    # :mod:`arabic_pdf_transcribe.ocr`), all non-figure regions on the
    # page go to the model in a single call so detection and recognition
    # warm-up cost amortises across the page. Adapters without
    # ``transcribe_page`` fall back to the per-region loop.
    page_batch = getattr(ocr_transcriber, "transcribe_page", None)
    for job in ml_jobs:
        page_index = job.native_page.page_index
        if page_index in outcomes:
            continue
        image = page_images[page_index]
        detected = page_regions[page_index]
        try:
            if callable(page_batch):
                transcribed = list(page_batch(detected, image))  # type: ignore[arg-type]
            else:
                transcribed = []
                for region in detected:
                    if region.role is RegionRole.FIGURE:
                        transcribed.append(region)
                        continue
                    transcribed.append(ocr_transcriber.transcribe(region, image))
            # ML-branch bboxes are in pixel coords on the rasterised image,
            # so the page dimensions handed to the post-processor must
            # match (otherwise full-page regions land in the header/footer
            # band by mistake).
            pixel_width, pixel_height = image.size
            regions = _post_process_regions(
                transcribed,
                page_width=float(pixel_width),
                page_height=float(pixel_height),
                reorder_call=reorder_call,
                classify_cfg=classify_cfg,
                rtl=rtl,
            )
        except BaseException as exc:
            outcomes[page_index] = _handle_phase_exc(
                exc,
                page_index=page_index,
                native_page=job.native_page,
                total=total,
                branch="ml",
                strict=strict,
                progress=progress,
            )
            continue
        _emit_progress(progress, page_index, total, "complete:ml")
        outcomes[page_index] = PageOutcome(
            page_index=page_index,
            branch="ml",
            regions=tuple(regions),
            validation=job.validation,
        )


# ---------------------------------------------------------------------------
# Shared exception → outcome handling
# ---------------------------------------------------------------------------


def _handle_phase_exc(
    exc: BaseException,
    *,
    page_index: int,
    native_page: NativePage,
    total: int,
    branch: str,
    strict: bool,
    progress: ProgressCallback | None,
) -> PageOutcome:
    """Map a phase exception to either a re-raise (strict) or a placeholder.

    Mirrors the per-page exception arms of the previous interleaved
    pipeline so failure semantics (typed exceptions, CUDA OOM
    detection, generic catch-all) are preserved across phases.
    """
    if isinstance(exc, MemoryError):
        if strict:
            raise OutOfMemoryDuringInference(
                f"page {page_index + 1}: {exc}; reduce --max-workers or rasterisation DPI"
            ) from exc
        reason = "out_of_memory"
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch)
    if isinstance(exc, RuntimeError) and _is_cuda_oom(exc):
        if strict:
            raise OutOfMemoryDuringInference(
                f"page {page_index + 1}: CUDA OOM: {exc}; "
                f"reduce --max-workers or rasterisation DPI"
            ) from exc
        _emit_progress(progress, page_index, total, "failure:cuda_out_of_memory")
        return _failure_outcome(page_index, native_page, "cuda_out_of_memory", branch)
    if isinstance(exc, ArabicPdfTranscribeError):
        # Include ``str(exc)`` in the reason so JSON-log consumers see
        # the actionable hint baked into the exception (e.g. the
        # "prefetch the weights with: arabic-pdf-transcribe
        # --prefetch-models" message on ModelDownloadError). Issue #16
        # root cause #3.
        reason = f"{type(exc).__name__}:{exc}"
        if strict:
            raise exc
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch)
    if isinstance(exc, Exception):
        # Bare-except trade-off: this is the per-page boundary for the
        # spec's "best-effort default" contract — if an upstream stage
        # raises an unexpected exception we MUST NOT abort the whole
        # run; a single failed page becomes a placeholder so the rest
        # of the document still transcribes. With ``--strict`` the
        # original exception is re-raised, preserving the traceback.
        reason = f"{type(exc).__name__}:{exc}"
        if strict:
            raise exc
        _emit_progress(progress, page_index, total, f"failure:{reason}")
        return _failure_outcome(page_index, native_page, reason, branch)
    # Non-Exception BaseException (KeyboardInterrupt, SystemExit, etc.):
    # never swallow.
    raise exc


def _release_adapter(adapter: object | None) -> None:
    """Call an adapter's ``release()`` if it exposes one.

    The Ray-actor proxies (:class:`RayLayoutProxy`, :class:`RayOCRProxy`)
    use ``release()`` to ``ray.kill`` their worker process between
    phases — this is what frees GPU memory on a 6 GB card so Phase D
    isn't squeezed by Phase C's leftover CUDA context. In-process
    adapters (thread mode) don't ship a ``release()`` and this is a
    no-op for them; ``hasattr`` keeps the orchestrator agnostic to
    the executor mode.

    Errors raised by the hook are swallowed: a failed teardown must
    not abort an otherwise successful run.
    """
    release = getattr(adapter, "release", None)
    if callable(release):
        try:
            release()
        except Exception:  # pragma: no cover — best-effort teardown
            pass


def _is_cuda_oom(exc: BaseException) -> bool:
    cls = type(exc)
    name = cls.__name__
    module = (cls.__module__ or "").split(".", 1)[0]
    return name == "OutOfMemoryError" and module == "torch"


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
