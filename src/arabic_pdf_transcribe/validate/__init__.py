"""Native-text quality validation.

The validator is the gate that decides per-page whether to trust the native
extraction or fall back to the ML branch. Phase 3 ships a multi-signal
validator with three independent signals (Arabic codepoint ratio,
replacement-glyph ratio, word-boundary plausibility); a page is accepted
iff all three signals fall inside their accept band.
"""

from arabic_pdf_transcribe.validate.native_validator import (
    ValidationResult,
    ValidatorConfig,
    arabic_codepoint_ratio,
    replacement_glyph_ratio,
    reset_reference_cache,
    validate_page,
    word_boundary_plausibility,
)

__all__ = [
    "ValidationResult",
    "ValidatorConfig",
    "arabic_codepoint_ratio",
    "replacement_glyph_ratio",
    "reset_reference_cache",
    "validate_page",
    "word_boundary_plausibility",
]
