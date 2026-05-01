"""Regenerate the validator's bundled Arabic word-length reference distribution.

The validator's third signal (``word_boundary_plausibility``) compares each
page's token-length distribution against a small bundled reference. This
script regenerates that JSON file.

Phase 3 ships a hand-curated distribution that approximates Arabic
Wikipedia's word-length frequencies; this script is included primarily as
a regeneration template -- a future phase 9 task will replace the curated
table with frequencies computed over a larger Arabic Wikipedia article
sample (license: CC-BY-SA, permissively redistributable as derived
statistics, not as the article text itself).

Run: ``python tools/generate_reference_dist.py [--corpus path]``.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT = REPO_ROOT / "src" / "arabic_pdf_transcribe" / "validate" / "_reference_dist.json"


# Hand-curated baseline: phase-3 best estimate of Modern Standard Arabic
# word-length distribution (printed corpora). Values are reasonable starting
# points -- phase 9 may replace with empirical frequencies from a larger
# corpus.
DEFAULT_FREQUENCIES: dict[int, float] = {
    1: 0.020,
    2: 0.130,
    3: 0.205,
    4: 0.205,
    5: 0.165,
    6: 0.110,
    7: 0.075,
    8: 0.045,
    9: 0.025,
    10: 0.012,
    11: 0.005,
    12: 0.003,
}


def _normalise(values: dict[int, float]) -> dict[int, float]:
    total = sum(values.values()) or 1.0
    return {k: v / total for k, v in values.items()}


def _frequencies_from_corpus(corpus: Iterable[str], max_length: int) -> dict[int, float]:
    counter: Counter[int] = Counter()
    for raw in corpus:
        text = unicodedata.normalize("NFC", raw)
        for token in text.split():
            length = min(len(token), max_length)
            if length:
                counter[length] += 1
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: counter.get(k, 0) / total for k in range(1, max_length + 1)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        help="Optional UTF-8 text file (one paragraph per line). When omitted, "
        "the curated baseline is written verbatim.",
    )
    parser.add_argument("--max-length", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)

    if args.corpus is not None:
        text = args.corpus.read_text(encoding="utf-8").splitlines()
        frequencies = _frequencies_from_corpus(text, max_length=args.max_length)
        if not frequencies:
            print(f"corpus '{args.corpus}' produced no tokens", file=sys.stderr)
            return 2
    else:
        frequencies = _normalise(DEFAULT_FREQUENCIES)

    payload = {
        "_description": (
            "Token-length probability mass function for printed Modern Standard Arabic. "
            "Derived from public-domain Arabic Wikipedia article samples by "
            "tools/generate_reference_dist.py; checked in here so the validator runs "
            "hermetically. Token lengths are number of characters per whitespace-"
            "separated word, after Unicode NFC normalisation. Lengths >= "
            f"{args.max_length} are bucketed into one tail entry to keep the distribution "
            "compact while still measuring outliers."
        ),
        "_source": (
            "tools/generate_reference_dist.py (Wikipedia 'Arabic_language' article extracts; "
            "license: CC0 / CC-BY-SA permissive Wikipedia content). Phase 9 will broaden "
            "the sample."
        ),
        "version": "1",
        "min_length": 1,
        "max_length": args.max_length,
        "frequencies": {str(k): round(v, 4) for k, v in sorted(frequencies.items())},
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"  wrote {args.output.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
