"""Phase-7 emitters: Markdown and Word (.docx).

Converts an ordered, role-classified :class:`Region` stream into a
human-readable artefact while preserving Unicode bidi correctness and
Arabic presentation-form fidelity.

Both emitters are pure functions of their input region stream (the
docx emitter takes an output path so it can write the file directly,
but it does not read environment state). Snapshots are stable across
runs: no timestamps, no environment-dependent data, deterministic
ordering.

Importing this package does *not* pull in :mod:`docx`; that import
is deferred to :func:`emit_docx`.
"""

from __future__ import annotations

from arabic_pdf_transcribe.emit._normalise import (
    NormalisationForm,
    normalise_text,
)
from arabic_pdf_transcribe.emit.markdown import emit_markdown


def emit_docx(*args: object, **kwargs: object) -> None:
    """Lazy proxy: defers ``python-docx`` import until first call."""
    from arabic_pdf_transcribe.emit.docx import emit_docx as _emit_docx

    return _emit_docx(*args, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "NormalisationForm",
    "emit_docx",
    "emit_markdown",
    "normalise_text",
]
