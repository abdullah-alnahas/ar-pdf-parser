"""Phase-9 native-path performance smoke.

The spec quotes a 5-second floor for the native-only path on a
10-page born-digital Arabic PDF. We don't have a 10-page corpus
fixture in-tree (yet) — the smoke test here uses the largest
available native-clean fixture and asserts a generous slack so a
slow CI runner doesn't false-fail.

Not a hard CI gate: this test runs in the regular pytest pass and
is allowed to be removed in a follow-up if it proves flaky.
ML-path performance is documented in the README, not gated in CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from arabic_pdf_transcribe.cli import EXIT_OK, main

FIXTURES = Path(__file__).parent / "fixtures" / "pdfs"
DIGITAL_CLEAN = FIXTURES / "digital-clean"

# Generous slack: the spec target is 5s for 10 pages, our fixture is
# tiny (one page), and a busy CI runner can be 10x slower than a
# laptop. We assert a wall-clock under 5s for a one-page native run,
# which is ~2x slack on a healthy CI runner.
NATIVE_PATH_BUDGET_S = 5.0


def test_native_path_under_budget(tmp_path: Path) -> None:
    """Wall-clock smoke for the native path on a clean Arabic PDF.

    Asserts the CLI returns 0 within ``NATIVE_PATH_BUDGET_S``. We
    don't try to be precise — the test fails only if the native
    path regresses by an order of magnitude.
    """
    candidate = DIGITAL_CLEAN / "lorem-ar-real.pdf"
    if not candidate.exists():
        pytest.skip(f"missing fixture: {candidate}")
    out = tmp_path / "out.md"
    start = time.perf_counter()
    rc = main([str(candidate), "-o", str(out), "--quiet"])
    elapsed = time.perf_counter() - start
    assert rc == EXIT_OK
    assert elapsed < NATIVE_PATH_BUDGET_S, (
        f"native path took {elapsed:.2f}s on {candidate.name}; "
        f"budget is {NATIVE_PATH_BUDGET_S}s"
    )
