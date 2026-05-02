"""Character Error Rate — small in-tree implementation.

Used by phase-9 e2e tests to assert ML-path output against
``*.expected.md`` references with a tolerance of ``≤ 0.05`` per the
spec's reproducibility contract. We keep this in-tree (no
``editdistance`` dependency) to keep the dep tree small and
license-clean.

CER is computed on Unicode codepoints — the right granularity for
Arabic where word boundaries are ambiguous and OCR may drop or merge
short connectors.
"""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Edit distance between ``a`` and ``b`` over Unicode codepoints.

    Pure-Python Wagner-Fischer with a rolling row to keep memory at
    ``O(min(len(a), len(b)))``. Insertion / deletion / substitution
    each cost 1; transpositions are not counted (Damerau variant is
    not needed for CER).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Ensure ``b`` is the shorter one so the rolling row stays small.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost_sub = previous[j - 1] + (0 if ca == cb else 1)
            cost_ins = current[j - 1] + 1
            cost_del = previous[j] + 1
            current[j] = min(cost_sub, cost_ins, cost_del)
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate.

    ``CER = edit_distance(ref, hyp) / max(len(ref), 1)``. The
    denominator never goes below 1 so an empty reference with a
    non-empty hypothesis returns ``len(hypothesis)`` (clamped at the
    upper bound of the test).

    Returns ``0.0`` when both strings are empty.
    """
    if not reference and not hypothesis:
        return 0.0
    distance = levenshtein(reference, hypothesis)
    denom = max(len(reference), 1)
    return distance / denom
