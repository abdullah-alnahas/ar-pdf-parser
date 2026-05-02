"""Bidi helpers for emitter output.

Spec rule: when an Arabic-dominant paragraph contains LTR runs (Latin
words, Western digits in non-numeric positions, URLs), the output
must keep the visual ordering stable across renderers. Many Markdown
renderers and Word do the right thing without any hint, but a
stray strong-LTR character at the start of a logical-Arabic paragraph
can cause whole-line direction flips on naïve renderers.

The conservative fix: prefix the paragraph with U+200F (RIGHT-TO-LEFT
MARK, RLM) when the dominant script is Arabic *and* the text
contains at least one LTR character. RLM is invisible, has no width,
and forces a strong-RTL context.

We do **not** insert RLM in every paragraph — empty paragraphs and
pure-Arabic / pure-LTR paragraphs need none. The decision is purely
deterministic so snapshot tests stay stable.

References:
- Unicode TR9 (Bidirectional Algorithm)
- :func:`unicodedata.bidirectional` returns the bidi class:
  ``"AL"`` (Arabic letter), ``"R"`` (Hebrew etc), ``"L"`` (LTR),
  ``"EN"`` / ``"AN"`` (digits — weak, not classified as LTR/RTL).
"""

from __future__ import annotations

import unicodedata

RLM = "‏"
"""RIGHT-TO-LEFT MARK — invisible strong-RTL character."""


def _classify_char(ch: str) -> str:
    """Return ``"R"``, ``"L"``, or ``""`` (neutral / weak)."""
    cls = unicodedata.bidirectional(ch)
    if cls in ("AL", "R"):
        return "R"
    if cls == "L":
        return "L"
    return ""


def is_arabic_dominant(text: str) -> bool:
    """Return ``True`` when more strong-RTL than strong-LTR characters."""
    rtl = ltr = 0
    for ch in text:
        kind = _classify_char(ch)
        if kind == "R":
            rtl += 1
        elif kind == "L":
            ltr += 1
    return rtl > 0 and rtl > ltr


def has_ltr_run(text: str) -> bool:
    """Return ``True`` when ``text`` contains a strong-LTR character."""
    return any(_classify_char(ch) == "L" for ch in text)


def add_rlm_if_needed(text: str) -> str:
    """Prefix ``text`` with U+200F when Arabic-dominant + contains LTR.

    Idempotent: re-applying does not double-prefix because a leading
    RLM is itself strong-RTL, so the dominance check still triggers,
    but the function detects an existing leading RLM and returns the
    input unchanged.
    """
    if not text:
        return text
    if text.startswith(RLM):
        return text
    if not is_arabic_dominant(text):
        return text
    if not has_ltr_run(text):
        return text
    return RLM + text


__all__ = [
    "RLM",
    "add_rlm_if_needed",
    "has_ltr_run",
    "is_arabic_dominant",
]
