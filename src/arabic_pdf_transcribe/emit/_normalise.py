"""Unicode normalisation policy for the emitters.

Architect carry-over (phase 3 PR-4 approval): phase 7 must explicitly
decide between NFC and NFKC for Arabic presentation forms.

## Decision: NFC default, NFKC opt-in

Arabic text in PDF documents frequently arrives in **presentation
forms** — codepoints in the ranges:

* ``U+FB50..U+FDFF`` (Arabic Presentation Forms-A): ligatures and
  isolated/initial/medial/final variants of base letters.
* ``U+FE70..U+FEFF`` (Arabic Presentation Forms-B): contextual shaping
  forms (isolated/initial/medial/final).

These are **compatibility decompositions** of base letters. NFKC
collapses them into their base forms; NFC does not.

We default to **NFC** because:

1. Many Arabic publishers intentionally emit presentation forms (e.g.
   the Allah ligature ``U+FDF2`` ﷲ, lam-alef ligatures ``U+FEFB`` etc.)
   for typographic reasons. NFKC would silently rewrite ﷲ → ا+ل+ل+ه,
   losing the publisher's intent.
2. Round-trip fidelity: a user copying transcribed text back into a
   typesetting tool expects the same glyphs they saw in the source.
3. NFC still applies canonical equivalence — combining marks are
   ordered, decomposed-then-recomposed sequences are unified — so
   text remains canonically stable across renderers without losing
   compatibility-only information.

NFKC is **opt-in** via :func:`normalise_text` ``form="NFKC"``. Users
who want search-friendly text (where ﷲ matches a search for
"الله") can request it explicitly. Phase 8's CLI may surface this
as a flag in a future iteration; phase 9's corpus retune will
revisit whether the default needs to change.

## Empty / non-Arabic input

The function applies the chosen form to all input regardless of
script — this is correct behaviour: a Latin region in an Arabic
document still needs canonical normalisation. NFKC's effects on
Latin (full-width digits, ligatures, superscripts) are well-defined
and non-controversial.
"""

from __future__ import annotations

import unicodedata
from typing import Literal

NormalisationForm = Literal["NFC", "NFKC"]
"""Allowed normalisation forms.

Deliberately excludes NFD / NFKD: emitters require composed output
(Word's :class:`docx` and most Markdown renderers prefer composed
forms; combining-mark sequences in decomposed form render poorly in
some Arabic-aware fonts).
"""

DEFAULT_FORM: NormalisationForm = "NFC"


def normalise_text(text: str, *, form: NormalisationForm = DEFAULT_FORM) -> str:
    """Normalise ``text`` to the requested Unicode form.

    Parameters
    ----------
    text:
        Input text (any script, any length).
    form:
        Either ``"NFC"`` (default — preserves Arabic presentation
        forms) or ``"NFKC"`` (opt-in — collapses presentation forms
        into base letters; lossy w.r.t. typography).

    Returns
    -------
    str
        Normalised text.
    """
    return unicodedata.normalize(form, text)


__all__ = ["DEFAULT_FORM", "NormalisationForm", "normalise_text"]
