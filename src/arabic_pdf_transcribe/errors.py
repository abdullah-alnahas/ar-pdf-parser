"""Typed exception hierarchy.

The full set of CLI-mapped exceptions is defined in phase 8 (orchestrator).
Phase 1 declares the two exceptions referenced from the ML adapter modules
(phases 4 and 5) so those phases do not need a forward reference back to the
pipeline module.

CLI exit-code mapping (from the spec; implemented in phase 8):

    0   success (or partial-success in default best-effort mode)
    2   strict-mode abort, or "all pages failed" in best-effort mode
    3   encrypted PDF
    4   corrupted PDF / format-extension mismatch
    5   model download / cache miss in offline mode
"""

from __future__ import annotations


class ArabicPdfTranscribeError(Exception):
    """Base class for every typed error this package raises."""


class ModelDownloadError(ArabicPdfTranscribeError):
    """Raised when a Hugging Face model cannot be loaded.

    Common causes: cache miss while running offline, no network connectivity,
    the pinned revision being unavailable, or the resolved license being
    incompatible with the project's allow-list.

    Mapped to CLI exit code 5.
    """


class OCRTranscriptionError(ArabicPdfTranscribeError):
    """Raised when the per-region OCR step fails for a recoverable reason.

    The orchestrator catches this and either inserts a failure-placeholder
    region (default best-effort mode) or aborts the run (``--strict``).
    """


__all__ = [
    "ArabicPdfTranscribeError",
    "ModelDownloadError",
    "OCRTranscriptionError",
]
