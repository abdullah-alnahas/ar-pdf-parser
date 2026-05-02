"""Phase-9 CER unit tests.

Covers correctness of the in-tree :func:`tests._cer.cer` /
:func:`tests._cer.levenshtein` implementation. The e2e tests use
``cer ≤ 0.05`` against ``*.expected.md`` references; if this
implementation is wrong, the whole acceptance criterion lies.
"""

from __future__ import annotations

import pytest

from tests._cer import cer, levenshtein

# ---------------------------------------------------------------------------
# Levenshtein primitives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("kitten", "sitting", 3),  # canonical example
        ("flaw", "lawn", 2),
        ("", "abc", 3),
        ("abc", "", 3),
        ("a", "b", 1),
        ("ab", "ba", 2),  # one insertion + one deletion
    ],
)
def test_levenshtein_canonical_cases(a: str, b: str, expected: int) -> None:
    assert levenshtein(a, b) == expected


def test_levenshtein_symmetric() -> None:
    assert levenshtein("hello", "world") == levenshtein("world", "hello")


def test_levenshtein_unicode_codepoints() -> None:
    """Edit distance counts codepoints, not bytes."""
    arabic_a = "السلام"  # 6 codepoints
    arabic_b = "السلامة"  # 7 codepoints — one inserted
    assert levenshtein(arabic_a, arabic_b) == 1


def test_levenshtein_full_substitution() -> None:
    assert levenshtein("aaa", "bbb") == 3


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------


def test_cer_identical_zero() -> None:
    assert cer("hello world", "hello world") == 0.0


def test_cer_both_empty_zero() -> None:
    assert cer("", "") == 0.0


def test_cer_full_substitution_one() -> None:
    """All characters wrong → CER = 1.0."""
    assert cer("aaa", "bbb") == 1.0


def test_cer_one_substitution_proportional() -> None:
    """One char wrong out of three → CER = 1/3."""
    assert cer("abc", "axc") == pytest.approx(1 / 3)


def test_cer_insertion_proportional() -> None:
    """Reference 'abc', hypothesis 'abcd' → 1 edit, denom = 3 → CER = 1/3."""
    assert cer("abc", "abcd") == pytest.approx(1 / 3)


def test_cer_empty_reference_with_hypothesis_clamped() -> None:
    """Empty reference + non-empty hypothesis → CER = len(hyp) / 1."""
    assert cer("", "abc") == 3.0


def test_cer_below_spec_tolerance() -> None:
    """OCR within the spec's CER ≤ 0.05 tolerance.

    Two single-codepoint substitutions in a 100-codepoint reference
    yields CER 0.02 — below the spec floor.
    """
    ref = "x" * 100
    hyp = ref[:49] + "Y" + ref[50:79] + "Y" + ref[80:]
    assert cer(ref, hyp) == pytest.approx(0.02)
    assert cer(ref, hyp) <= 0.05


def test_cer_above_spec_tolerance() -> None:
    """CER above the 0.05 floor is correctly flagged."""
    ref = "x" * 100
    hyp = "Y" * 100
    assert cer(ref, hyp) == 1.0
    assert cer(ref, hyp) > 0.05


def test_cer_arabic_passthrough() -> None:
    """Arabic codepoints round-trip with zero error when identical."""
    arabic = "السلام عليكم ورحمة الله وبركاته"
    assert cer(arabic, arabic) == 0.0
