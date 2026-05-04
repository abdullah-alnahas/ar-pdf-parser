"""CLI entry point for ``arabic-pdf-transcribe``.

Command surface (per spec "Resolved Decisions" + plan phase 8):

::

    arabic-pdf-transcribe FILE
        [-o PATH] [--format {md,docx}]
        [--pages RANGES]
        [--strict] [--quiet] [--json-logs]
        [--config PATH]
        [--debug-json PATH]
        [--max-workers N]
        [--dpi N]
    arabic-pdf-transcribe --prefetch-models [--config PATH]

Exit codes (mapped one-to-one to spec test scenarios 11-18):

* ``0`` — success, including best-effort runs where some pages fail.
* ``2`` — strict-mode abort, or every page failed in best-effort mode.
* ``3`` — encrypted / password-protected PDF.
* ``4`` — corrupted PDF, or ``--format`` and ``-o`` extension disagree.
* ``5`` — ML model missing (download/cache miss).

Argparse parses; :func:`validate_args` resolves format/extension
conflicts and produces a :class:`ValidatedArgs` view; :func:`main`
calls into :func:`pipeline.transcribe`, then dispatches to the
emitters. The CLI never imports ``transformers`` / ``torch`` at the
top level — the ML deps load only when the validator forces a page
into the ML branch.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

from arabic_pdf_transcribe._logging import ProgressLogger, ProgressMode
from arabic_pdf_transcribe.errors import (
    ArabicPdfTranscribeError,
    CorruptedPDFError,
    EncryptedPDFError,
    FormatExtensionMismatch,
    ModelDownloadError,
    OutOfMemoryDuringInference,
)
from arabic_pdf_transcribe.pipeline import TranscribeResult, transcribe
from arabic_pdf_transcribe.validate.native_validator import ValidatorConfig

OutputFormat = Literal["md", "docx"]

EXIT_OK = 0
EXIT_RUNTIME = 2
EXIT_ENCRYPTED = 3
EXIT_CORRUPTED_OR_FORMAT = 4
EXIT_MODEL_MISSING = 5

_MD_EXTS = frozenset({".md", ".markdown"})
_DOCX_EXTS = frozenset({".docx"})


# ---------------------------------------------------------------------------
# Argparse + post-validation
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arabic-pdf-transcribe",
        description="Arabic-first PDF transcriber: extract → layout → ML fallback → MD/Word.",
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="path to a PDF file (omit when using --prefetch-models)",
    )
    parser.add_argument(
        "--prefetch-models",
        action="store_true",
        help="download ML layout + OCR weights into the local HF cache, then exit",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output file path (defaults to stdout in Markdown)",
    )
    parser.add_argument(
        "--format",
        choices=("md", "docx"),
        default=None,
        help="output format; required when --output has no recognised extension",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="page ranges to process, e.g. 1-3,7,10-12 (1-based, inclusive)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="abort on the first per-page failure (default: best-effort)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress progress reporting on stderr",
    )
    parser.add_argument(
        "--json-logs",
        action="store_true",
        help="emit progress as JSON lines on stderr (overrides text mode)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="TOML config file for validator / layout / ocr / render overrides",
    )
    parser.add_argument(
        "--debug-json",
        type=Path,
        default=None,
        help="write a JSON sidecar with per-region confidences to PATH",
    )
    parser.add_argument(
        "--max-workers",
        type=str,
        default="1",
        help="per-page parallelism for the ML branch (1, N, or 'auto'; default 1)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="rasterisation DPI for the ML branch (default 200)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default=None,
        help=(
            "ML inference device: 'auto' (default; uses CUDA when "
            "torch.cuda.is_available()), 'cuda', or 'cpu'. Overrides "
            "[runtime].device in the config file."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default=None,
        help=(
            "ML model precision: 'auto' (default; bf16 on Ampere+ CUDA, "
            "fp16 on older CUDA, fp32 on CPU), 'float32', 'float16', or "
            "'bfloat16'. fp16/bf16 halve VRAM use vs fp32 with negligible "
            "OCR quality loss; required for 6 GB GPUs. Overrides "
            "[layout].dtype / [ocr].dtype in the config file."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("full-page", "doclayout-yolo"),
        default="full-page",
        help=(
            "ML layout backend: 'full-page' (default; whole page = one "
            "region, lets the OCR backend handle layout internally) or "
            "'doclayout-yolo' (DocLayout-YOLO regions). See the model "
            "survey at notebooks/00_model_survey.ipynb for trade-offs."
        ),
    )
    parser.add_argument(
        "--ocr",
        choices=("surya", "easyocr-ara"),
        default="surya",
        help=(
            "OCR backend: 'surya' (default; multilingual line-level OCR, "
            "highest quality on the test corpus) or 'easyocr-ara' "
            "(Arabic-only, CPU-friendly, more typos). See "
            "notebooks/00_model_survey.ipynb."
        ),
    )
    parser.add_argument(
        "--no-formula",
        action="store_true",
        help=(
            "disable LaTeX/formula recognition; OCR runs in plain-text "
            "mode and any residual <math>...</math> or $...$ wrappers "
            "in the output are stripped."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "OCR batch size hint; forwarded to surya's "
            "recognition_batch_size / detection_batch_size. Tune up "
            "for higher GPU throughput, down to fit lower-VRAM cards."
        ),
    )
    parser.add_argument(
        "--ltr",
        action="store_true",
        help="emit a left-to-right Word document (default: RTL).",
    )
    parser.add_argument(
        "--no-page-breaks",
        action="store_true",
        help="do not insert a page break between source PDF pages in DOCX output.",
    )
    return parser


@dataclass(frozen=True, slots=True)
class ValidatedArgs:
    """Post-argparse, post-conflict-resolution view of CLI arguments."""

    input: Path
    output: Path | None
    format: OutputFormat
    pages: tuple[int, ...] | None  # 0-based, sorted, deduplicated
    strict: bool
    progress_mode: ProgressMode
    config: Path | None
    debug_json: Path | None
    max_workers: int  # resolved from "auto" / int
    dpi: int
    device: str | None  # "auto" | "cuda" | "cpu" | None (use TOML / default)
    dtype: str | None  # "auto" | "float32" | "float16" | "bfloat16" | None
    layout_backend: str  # "full-page" | "doclayout-yolo"
    ocr_backend: str  # "surya" | "easyocr-ara"
    disable_formula: bool
    batch_size: int | None
    rtl_docx: bool
    page_breaks: bool


def validate_args(ns: argparse.Namespace) -> ValidatedArgs:
    """Resolve format/extension conflicts and normalise inputs.

    Raises :class:`FormatExtensionMismatch` (CLI exit 4) when
    ``--format`` and the ``-o PATH`` extension disagree, per spec
    test scenario 18, AND when ``--format docx`` is requested without
    a ``-o PATH`` (a Word file cannot be written to stdout).
    """
    fmt = _resolve_format(ns.output, ns.format)
    if fmt == "docx" and ns.output is None:
        raise FormatExtensionMismatch(
            "--format docx requires -o PATH (cannot write a Word file to stdout)"
        )
    pages = _parse_pages(ns.pages) if ns.pages else None
    progress_mode = _resolve_progress_mode(quiet=ns.quiet, json_logs=ns.json_logs)
    max_workers = _resolve_max_workers(ns.max_workers)
    dpi = ns.dpi if ns.dpi is not None else 200
    if ns.batch_size is not None and ns.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {ns.batch_size}")
    return ValidatedArgs(
        input=ns.input,
        output=ns.output,
        format=fmt,
        pages=pages,
        strict=ns.strict,
        progress_mode=progress_mode,
        config=ns.config,
        debug_json=ns.debug_json,
        max_workers=max_workers,
        dpi=dpi,
        device=ns.device,
        dtype=ns.dtype,
        layout_backend=ns.layout,
        ocr_backend=ns.ocr,
        disable_formula=ns.no_formula,
        batch_size=ns.batch_size,
        rtl_docx=not ns.ltr,
        page_breaks=not ns.no_page_breaks,
    )


def _resolve_format(output: Path | None, explicit: str | None) -> OutputFormat:
    """Decide the output format from ``-o`` extension and ``--format``.

    Rules (per spec):

    * If both are present and disagree → :class:`FormatExtensionMismatch`.
    * If ``-o`` has a recognised extension → extension wins.
    * If ``-o`` is None or unknown extension → ``--format`` wins.
    * If neither carries information → default to Markdown.
    """
    ext_format = _format_from_ext(output) if output is not None else None
    if ext_format is not None and explicit is not None and ext_format != explicit:
        raise FormatExtensionMismatch(
            f"--format {explicit!r} disagrees with output extension {output!s} "
            f"(extension implies {ext_format!r}); pass matching --format or change the path"
        )
    if ext_format is not None:
        return ext_format
    if explicit is not None:
        return _cast_format(explicit)
    return "md"


def _format_from_ext(path: Path) -> OutputFormat | None:
    suffix = path.suffix.lower()
    if suffix in _MD_EXTS:
        return "md"
    if suffix in _DOCX_EXTS:
        return "docx"
    return None


def _cast_format(value: str) -> OutputFormat:
    if value not in ("md", "docx"):
        raise ValueError(f"unsupported format: {value!r}")  # pragma: no cover — argparse-gated
    return value  # type: ignore[return-value]


def _parse_pages(spec: str) -> tuple[int, ...]:
    """Parse ``"1-3,7,10-12"`` into 0-based, sorted, deduplicated indices.

    Raises :class:`ValueError` on malformed spec; argparse surfaces
    this to the user as a usage error (exit 2).
    """
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo < 1 or hi < lo:
                raise ValueError(f"invalid page range {chunk!r}")
            for p in range(lo, hi + 1):
                out.add(p - 1)
        else:
            n = int(chunk)
            if n < 1:
                raise ValueError(f"invalid page number {chunk!r}")
            out.add(n - 1)
    return tuple(sorted(out))


def _resolve_progress_mode(*, quiet: bool, json_logs: bool) -> ProgressMode:
    if quiet:
        return ProgressMode.QUIET
    if json_logs:
        return ProgressMode.JSON
    return ProgressMode.TEXT


class _ProgressUI:
    """Unified progress observer used by :func:`main`.

    On a TTY in default TEXT mode, builds a two-bar live display via
    :mod:`arabic_pdf_transcribe._progress_ui`. Otherwise falls back to
    the line-format :class:`ProgressLogger` so piped stderr, ``--quiet``,
    and ``--json-logs`` keep their byte-stable behaviour. The rich UI is
    constructed lazily on the first event so we know the document's
    total page count before drawing.
    """

    def __init__(self, mode: ProgressMode, stream: TextIO) -> None:
        self._mode = mode
        self._stream = stream
        self._use_rich = mode is ProgressMode.TEXT and _stream_is_tty(stream)
        self._rich = None  # type: ignore[var-annotated]
        self._logger: ProgressLogger | None = (
            None if self._use_rich else ProgressLogger(mode, stream=stream)
        )

    def on_event(self, page_index: int, total: int, event: str) -> None:
        if self._use_rich:
            if self._rich is None:
                from arabic_pdf_transcribe._progress_ui import RichTwoBarUI

                self._rich = RichTwoBarUI(total_pages=total, stream=self._stream)
            self._rich.handle_event(page_index, total, event)
            return
        self._dispatch_to_logger(page_index, total, event)

    def summary(self, *, of: int, ok_pages: int, failed_pages: int) -> None:
        if self._use_rich and self._rich is not None:
            self._rich.write_summary(
                f"summary: {of} pages, ok={ok_pages} failed={failed_pages}"
            )
            return
        if self._logger is not None:
            self._logger.summary(of=of, ok_pages=ok_pages, failed_pages=failed_pages)

    def close(self) -> None:
        if self._rich is not None:
            self._rich.close()

    # ------------------------------------------------------------------
    # Line-logger dispatch (mirrors the previous inline ``_progress`` fn).
    # ------------------------------------------------------------------

    def _dispatch_to_logger(self, page_index: int, total: int, event: str) -> None:
        if self._logger is None:  # pragma: no cover — defensive
            return
        # Phase header events are rich-only; the line logger ignores them.
        if event.startswith("phase:start:"):
            return
        page = page_index + 1
        if event == "start":
            self._logger.start(page=page, of=total)
        elif event == "layout":
            self._logger.layout(page=page, of=total)
        elif event == "rasterise":
            # The line logger has no dedicated rasterise renderer; fold
            # it into the layout-style "ml step" line.
            return
        elif event.startswith("region:"):
            payload = event.split(":", 2)[1:]  # ["i/n", "role"]
            if len(payload) == 2 and "/" in payload[0]:
                idx_s, n_s = payload[0].split("/", 1)
                try:
                    idx = int(idx_s)
                    n_regions = int(n_s)
                except ValueError:  # pragma: no cover — defensive
                    return
                self._logger.region(
                    page=page,
                    of=total,
                    region=idx,
                    of_regions=n_regions,
                    role=payload[1],
                )
        elif event.startswith("complete:"):
            branch = event.split(":", 1)[1]
            self._logger.complete(page=page, of=total, branch=branch)
        elif event.startswith("failure:"):
            reason = event.split(":", 1)[1]
            self._logger.failure(page=page, of=total, reason=reason)


def _stream_is_tty(stream: TextIO) -> bool:
    """Best-effort isatty check; returns ``False`` on broken streams."""
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError, OSError):  # pragma: no cover — defensive
        return False


def _resolve_max_workers(value: str) -> int:
    if value == "auto":
        try:
            import os

            return min(os.cpu_count() or 1, 4)
        except (AttributeError, OSError):  # pragma: no cover — defensive
            return 1
    try:
        n = int(value)
    except ValueError as exc:
        raise ValueError(f"--max-workers must be an int or 'auto', got {value!r}") from exc
    if n < 1:
        raise ValueError(f"--max-workers must be >= 1, got {n}")
    return n


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.prefetch_models:
        if ns.input is not None:
            parser.error("--prefetch-models does not accept an input file; pass it on its own")
        return _prefetch_models(
            _load_config_doc(ns.config),
            layout_backend=ns.layout,
            ocr_backend=ns.ocr,
        )
    if ns.input is None:
        parser.error("the following arguments are required: input (or pass --prefetch-models)")
    try:
        args = validate_args(ns)
    except FormatExtensionMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CORRUPTED_OR_FORMAT
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    ui = _ProgressUI(args.progress_mode, sys.stderr)

    config_doc = _load_config_doc(args.config)
    validator_cfg = _validator_cfg_from_doc(config_doc)
    device = _resolve_device(args.device, config_doc)
    dtype = _resolve_dtype(args.dtype, config_doc)
    layout_detector, ocr_transcriber = _maybe_build_ml_adapters(
        config_doc,
        device=device,
        dtype=dtype,
        layout_backend=args.layout_backend,
        ocr_backend=args.ocr_backend,
        disable_formula=args.disable_formula,
        batch_size=args.batch_size,
    )
    dpi = _resolve_dpi(args.dpi, config_doc)

    try:
        try:
            result = transcribe(
                args.input,
                layout_detector=layout_detector,
                ocr_transcriber=ocr_transcriber,
                validator_config=validator_cfg,
                pages=args.pages,
                strict=args.strict,
                dpi=dpi,
                max_workers=args.max_workers,
                progress=ui.on_event,
            )
        finally:
            # Tear down any rich Live display before error messages
            # below (or the summary print) reach stderr — otherwise the
            # bars overwrite the message.
            ui.close()
    except FileNotFoundError as exc:
        print(f"error: input file not found: {exc.filename or args.input}", file=sys.stderr)
        return EXIT_CORRUPTED_OR_FORMAT
    except EncryptedPDFError as exc:
        print(f"error: encrypted PDF: {exc}", file=sys.stderr)
        return EXIT_ENCRYPTED
    except CorruptedPDFError as exc:
        print(f"error: corrupted PDF: {exc}", file=sys.stderr)
        return EXIT_CORRUPTED_OR_FORMAT
    except ModelDownloadError as exc:
        print(f"error: ML model unavailable: {exc}", file=sys.stderr)
        return EXIT_MODEL_MISSING
    except OutOfMemoryDuringInference as exc:
        print(f"error: out-of-memory during inference: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except ArabicPdfTranscribeError as exc:
        # Strict-mode abort surfaces a typed exception (e.g.
        # OCRTranscriptionError) that none of the specific arms above
        # match. Per spec scenario 15: clean exit 2 with a stderr
        # message, not a Python traceback.
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    ui.summary(of=result.n_pages, ok_pages=result.ok_pages, failed_pages=result.failed_pages)

    if result.all_failed and result.n_pages > 0:
        print("error: every page failed; nothing to write", file=sys.stderr)
        return EXIT_RUNTIME

    _write_output(args, result)

    if args.debug_json is not None:
        _write_debug_json(args.debug_json, result)

    return EXIT_OK


def _load_config_doc(config_path: Path | None) -> dict[str, object]:
    """Parse the TOML config file once into a section-keyed dict.

    Returns an empty dict when no config is given. Each section
    (``[validator]``, ``[layout]``, ``[ocr]``, ``[render]``) is
    consumed by its respective adapter. Unknown sections are ignored
    for forward compatibility.
    """
    if config_path is None:
        return {}
    import tomllib

    return tomllib.loads(config_path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def _validator_cfg_from_doc(doc: dict[str, object]) -> ValidatorConfig | None:
    section = doc.get("validator")
    if not isinstance(section, dict):
        return None
    kwargs = {k: v for k, v in section.items() if k in ValidatorConfig.__dataclass_fields__}
    return ValidatorConfig(**kwargs)  # type: ignore[arg-type]


def _resolve_dpi(cli_dpi: int | None, doc: dict[str, object]) -> int:
    """``--dpi`` CLI flag wins; otherwise read ``[render].dpi``; default 200."""
    if cli_dpi is not None:
        return cli_dpi
    section = doc.get("render")
    if isinstance(section, dict):
        value = section.get("dpi")
        if isinstance(value, int) and value > 0:
            return value
    return 200


def _resolve_device(cli_device: str | None, doc: dict[str, object]) -> str:
    """Resolve the runtime device.

    ``--device`` wins over ``[runtime].device`` over ``"auto"``.

    Issue #18 RC#1: gives the user a knob to force CPU when GPU OOMs
    early or to force CUDA when auto-detection misses (e.g. ROCm
    builds).
    """
    if cli_device is not None:
        return cli_device
    section = doc.get("runtime")
    if isinstance(section, dict):
        value = section.get("device")
        if isinstance(value, str) and value:
            return value
    return "auto"


def _resolve_dtype(cli_dtype: str | None, doc: dict[str, object]) -> str:
    """Resolve the ML dtype.

    Mirrors :func:`_resolve_device`: ``--dtype`` wins over
    ``[runtime].dtype`` over ``"auto"``.

    Issue #20 RC#1: gives the user a knob to force fp16 / bf16 for
    6 GB GPUs, or to force fp32 when reduced precision degrades a
    specific corpus.
    """
    if cli_dtype is not None:
        return cli_dtype
    section = doc.get("runtime")
    if isinstance(section, dict):
        value = section.get("dtype")
        if isinstance(value, str) and value:
            return value
    return "auto"


def _prefetch_models(
    config_doc: dict[str, object],
    *,
    layout_backend: str = "full-page",
    ocr_backend: str = "surya",
) -> int:
    """Trigger weight download for the selected layout + OCR backends.

    The adapters' loaders own the actual download logic (HF Hub for
    DocLayout-YOLO weights, surya / easyocr for their own caches). This
    function just invokes one inference per adapter on a 1×1 dummy
    image so the model files end up in the local cache.
    """
    from PIL import Image

    dummy = Image.new("RGB", (32, 32), color=(255, 255, 255))
    targets: list[tuple[str, object | None]] = []

    layout = _build_layout(layout_backend, device="cpu")
    if layout is None:
        print(
            f"error: --prefetch-models could not construct layout backend "
            f"{layout_backend!r}; install missing optional deps",
            file=sys.stderr,
        )
        return EXIT_MODEL_MISSING
    targets.append(("layout", layout))

    ocr = _build_ocr(ocr_backend, device="cpu")
    if ocr is None:
        print(
            f"error: --prefetch-models could not construct OCR backend "
            f"{ocr_backend!r}; install missing optional deps",
            file=sys.stderr,
        )
        return EXIT_MODEL_MISSING
    targets.append(("ocr", ocr))

    for label, adapter in targets:
        print(f"prefetching {label} backend ({type(adapter).__name__})", file=sys.stderr)
        try:
            if label == "layout":
                adapter.detect(dummy, page_index=0)  # type: ignore[attr-defined]
            else:
                from arabic_pdf_transcribe.regions import (
                    BBox,
                    Region,
                    RegionRole,
                    RegionSource,
                )

                region = Region(
                    page_index=0,
                    bbox=BBox(0.0, 0.0, 32.0, 32.0),
                    text="",
                    role=RegionRole.PARAGRAPH,
                    source=RegionSource.OCR,
                )
                adapter.transcribe(region, dummy)  # type: ignore[attr-defined]
        except Exception as exc:
            print(
                f"error: failed to prefetch {label} backend ({type(adapter).__name__}): {exc}",
                file=sys.stderr,
            )
            return EXIT_MODEL_MISSING
    print("prefetch complete; the ML branch can now run offline.", file=sys.stderr)
    return EXIT_OK


def _maybe_build_ml_adapters(
    doc: dict[str, object] | None = None,
    *,
    device: str | None = None,
    dtype: str | None = None,
    layout_backend: str = "full-page",
    ocr_backend: str = "surya",
    disable_formula: bool = False,
    batch_size: int | None = None,
) -> tuple[object | None, object | None]:
    """Return ``(layout_detector, ocr_transcriber)`` or ``(None, None)``.

    The backends are chosen by the CLI flags ``--layout`` and ``--ocr``.
    Adapters load their underlying models lazily on the first call.
    When a backend's optional dependencies are not installed,
    construction fails with a :class:`ModelDownloadError` at first use,
    which the orchestrator maps to a per-page failure (or strict abort).
    """
    resolved_device = device or "auto"
    layout_detector = _build_layout(layout_backend, device=resolved_device)
    ocr_transcriber = _build_ocr(
        ocr_backend,
        device=resolved_device,
        disable_formula=disable_formula,
        batch_size=batch_size,
    )
    return layout_detector, ocr_transcriber


def _build_layout(name: str, *, device: str) -> object | None:
    if name == "full-page":
        from arabic_pdf_transcribe.layout.full_page import FullPageLayoutDetector

        return FullPageLayoutDetector()
    if name == "doclayout-yolo":
        try:
            from arabic_pdf_transcribe.layout.doclayout_yolo import (
                DocLayoutYoloDetector,
            )
        except ImportError:
            return None
        return DocLayoutYoloDetector(device=device)
    raise ValueError(f"unknown layout backend: {name!r}")


def _build_ocr(
    name: str,
    *,
    device: str,
    disable_formula: bool = False,
    batch_size: int | None = None,
) -> object | None:
    if name == "surya":
        try:
            from arabic_pdf_transcribe.ocr.surya_ocr import SuryaOCRTranscriber
        except ImportError:
            return None
        return SuryaOCRTranscriber(
            disable_formula=disable_formula, batch_size=batch_size
        )
    if name == "easyocr-ara":
        try:
            from arabic_pdf_transcribe.ocr.easy_ocr import EasyOCRTranscriber
        except ImportError:
            return None
        return EasyOCRTranscriber(
            device=device,
            disable_formula=disable_formula,
            batch_size=batch_size,
        )
    raise ValueError(f"unknown OCR backend: {name!r}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_output(args: ValidatedArgs, result: TranscribeResult) -> None:
    if args.format == "md":
        from arabic_pdf_transcribe.emit.markdown import emit_markdown

        markdown = emit_markdown(result.regions)
        _write_text(args.output, markdown, sys.stdout)
    else:
        # docx + missing output path is rejected up-front in validate_args.
        assert args.output is not None
        from arabic_pdf_transcribe.emit.docx import emit_docx

        emit_docx(
            result.regions,
            args.output,
            rtl=args.rtl_docx,
            page_breaks=args.page_breaks,
        )


def _write_text(output: Path | None, text: str, stdout: TextIO) -> None:
    if output is None:
        stdout.write(text)
        if not text.endswith("\n"):
            stdout.write("\n")
        stdout.flush()
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def _write_debug_json(path: Path, result: TranscribeResult) -> None:
    """Emit a per-region JSON sidecar with confidences."""
    payload = {
        "n_pages": result.n_pages,
        "ok_pages": result.ok_pages,
        "failed_pages": result.failed_pages,
        "regions": [
            {
                "page_index": r.page_index,
                "role": r.role.value,
                "source": r.source.value,
                "confidence": r.confidence,
                "text_len": len(r.text),
                "failure_reason": r.failure_reason,
            }
            for r in result.regions
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


__all__ = [
    "EXIT_CORRUPTED_OR_FORMAT",
    "EXIT_ENCRYPTED",
    "EXIT_MODEL_MISSING",
    "EXIT_OK",
    "EXIT_RUNTIME",
    "ValidatedArgs",
    "build_parser",
    "main",
    "validate_args",
]
