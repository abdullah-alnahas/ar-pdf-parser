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
    parser.add_argument("input", type=Path, help="path to a PDF file")
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
    try:
        args = validate_args(ns)
    except FormatExtensionMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_CORRUPTED_OR_FORMAT
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME

    logger = ProgressLogger(args.progress_mode, stream=sys.stderr)

    validator_cfg = _load_validator_config(args.config)
    layout_detector, ocr_transcriber = _maybe_build_ml_adapters()

    def _progress(page_index: int, total: int, event: str) -> None:
        page = page_index + 1
        if event == "start":
            logger.start(page=page, of=total)
        elif event.startswith("complete:"):
            branch = event.split(":", 1)[1]
            logger.complete(page=page, of=total, branch=branch)
        elif event.startswith("failure:"):
            reason = event.split(":", 1)[1]
            logger.failure(page=page, of=total, reason=reason)

    try:
        result = transcribe(
            args.input,
            layout_detector=layout_detector,
            ocr_transcriber=ocr_transcriber,
            validator_config=validator_cfg,
            pages=args.pages,
            strict=args.strict,
            dpi=args.dpi,
            max_workers=args.max_workers,
            progress=_progress,
        )
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

    logger.summary(of=result.n_pages, ok_pages=result.ok_pages, failed_pages=result.failed_pages)

    if result.all_failed and result.n_pages > 0:
        print("error: every page failed; nothing to write", file=sys.stderr)
        return EXIT_RUNTIME

    _write_output(args, result)

    if args.debug_json is not None:
        _write_debug_json(args.debug_json, result)

    return EXIT_OK


def _load_validator_config(config_path: Path | None) -> ValidatorConfig | None:
    if config_path is None:
        return None
    return ValidatorConfig.from_toml(config_path)


def _maybe_build_ml_adapters() -> tuple[object | None, object | None]:
    """Return ``(layout_detector, ocr_transcriber)`` or ``(None, None)``.

    The CLI only attempts to wire the ML adapters when their imports
    succeed; the adapters themselves load their model lazily on the
    first call. When the optional ``[ml]`` extra is not installed,
    the CLI proceeds without ML support — pages that need the ML
    branch will surface :class:`RuntimeError` from the orchestrator,
    which we map to a per-page failure (or strict abort).
    """
    try:
        from arabic_pdf_transcribe.layout.hf_detector import HFDiTLayoutDetector
        from arabic_pdf_transcribe.ocr.hf_ocr import HFGotOCRTranscriber
    except ImportError:
        return None, None
    try:
        return HFDiTLayoutDetector(), HFGotOCRTranscriber()
    except Exception:  # pragma: no cover — defensive; constructors are cheap
        return None, None


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

        emit_docx(result.regions, args.output)


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
