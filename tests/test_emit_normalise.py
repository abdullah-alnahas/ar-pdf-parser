"""Phase-7 normalisation tests.

Verifies the architect's NFC-default decision: presentation forms
preserved by default, collapsed only under explicit NFKC opt-in.
"""

from __future__ import annotations

import unicodedata

from arabic_pdf_transcribe.emit._normalise import (
    DEFAULT_FORM,
    normalise_text,
)

# U+FDF2: ARABIC LIGATURE ALLAH ISOLATED FORM (presentation form)
ALLAH_LIGATURE = "ﷲ"

# U+FEFB: ARABIC LIGATURE LAM WITH ALEF ISOLATED FORM
LAM_ALEF_LIGATURE = "ﻻ"

# Decomposed sequence: ALEF + COMBINING HAMZA ABOVE = U+0623 (canonical)
DECOMPOSED_ALEF_HAMZA = "أ"  # ا + ٔ
COMPOSED_ALEF_HAMZA = "أ"  # أ


def test_default_is_nfc() -> None:
    assert DEFAULT_FORM == "NFC"


def test_nfc_preserves_allah_ligature() -> None:
    """Architect decision: NFC must NOT collapse U+FDF2 to base letters."""
    assert normalise_text(ALLAH_LIGATURE) == ALLAH_LIGATURE


def test_nfc_preserves_lam_alef_ligature() -> None:
    assert normalise_text(LAM_ALEF_LIGATURE) == LAM_ALEF_LIGATURE


def test_nfkc_collapses_allah_ligature() -> None:
    """Opt-in NFKC must collapse U+FDF2 into the base letter sequence."""
    out = normalise_text(ALLAH_LIGATURE, form="NFKC")
    assert out != ALLAH_LIGATURE
    # Decomposes to a 4-char sequence
    assert len(out) >= 3


def test_nfc_recomposes_canonical_sequences() -> None:
    """Decomposed ا + ٔ should canonical-compose to أ under NFC."""
    out = normalise_text(DECOMPOSED_ALEF_HAMZA)
    assert out == COMPOSED_ALEF_HAMZA


def test_empty_input() -> None:
    assert normalise_text("") == ""


def test_latin_text_passthrough_under_nfc() -> None:
    assert normalise_text("Hello world") == "Hello world"


def test_nfkc_collapses_latin_full_width_digits() -> None:
    full_width_one = "１"  # FULLWIDTH DIGIT ONE
    out = normalise_text(full_width_one, form="NFKC")
    assert out == "1"


def test_normalise_is_pure() -> None:
    text = ALLAH_LIGATURE + " test"
    assert normalise_text(text) == normalise_text(text)


def test_arabic_presentation_form_b_preserved_under_nfc() -> None:
    """U+FE70..U+FEFF range stays intact under NFC."""
    contextual = "ﺎ"  # ARABIC LETTER ALEF FINAL FORM
    assert normalise_text(contextual) == contextual
    # And NFKC collapses it.
    assert normalise_text(contextual, form="NFKC") != contextual


def test_normalize_idempotent() -> None:
    text = "تجربة " + ALLAH_LIGATURE
    once = normalise_text(text)
    twice = normalise_text(once)
    assert once == twice


def test_unicodedata_consistency() -> None:
    """Sanity: our wrapper matches stdlib output exactly."""
    sample = ALLAH_LIGATURE + DECOMPOSED_ALEF_HAMZA + "abc"
    assert normalise_text(sample, form="NFC") == unicodedata.normalize("NFC", sample)
    assert normalise_text(sample, form="NFKC") == unicodedata.normalize("NFKC", sample)
