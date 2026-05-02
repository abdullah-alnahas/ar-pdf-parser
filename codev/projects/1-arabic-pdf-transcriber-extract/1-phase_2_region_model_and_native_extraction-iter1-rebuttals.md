# Phase 2 (Region + Native Extraction) — Iteration 1 Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where |
|---|---|---|
| Phase-2 Arabic-scope evidence missing: fixtures and tests assert ASCII tokens only | Added `lorem-ar-real.pdf` fixture (generated when an Arabic-capable system TTF — Noto / DejaVu — is available) carrying real Arabic codepoints in the text layer. New `test_extract_native_real_arabic_codepoints_round_trip` asserts every Arabic codepoint we wrote round-trips through extraction. The test gracefully skips on systems without an Arabic-capable TTF so CI on a minimal image does not fail; phase 9's corpus will land bidi-shaped Arabic with realistic font embedding. | `tools/generate_fixtures.py` (`_build_real_arabic`), `tests/test_extract_native.py`, `tests/fixtures/pdfs/digital-clean/lorem-ar-real.pdf`. |
| `_line_from_words` sorted by X, reversing logical order on RTL Arabic lines (phase 6 cannot fix afterwards) | `_group_lines` rewritten to **not sort** at all — it walks words in PDF-stream order, binning each into the most recent line band whose Y is within `median_height * 0.5`. `_line_from_words` no longer sorts either. Sorting by Y also removed because glyph bboxes in the same visual line typically differ by sub-pixel amounts (descenders, accents) and a strict Y-sort can flip word order *within* a single line. New `test_line_grouping_preserves_pdf_stream_order` asserts that three Arabic words submitted in stream order (high-x-first as per real Arabic PDFs) come back in the same order. | `src/arabic_pdf_transcribe/extract/native.py` (`_group_lines`, `_line_from_words`), `tests/test_extract_native.py`. |
| `_split_line_into_cells` assumed left-to-right ordering when splitting cells, which would break for RTL lines | Updated to use the absolute horizontal distance between successive word bboxes (next word may be to the left of previous on RTL streams). Cell ordering is preserved as input (= stream = logical). | `src/arabic_pdf_transcribe/extract/_native_tables.py` (`_split_line_into_cells`). |
| Fixture provenance / licensing incomplete (`LICENSES.md` still says "_none yet_") | Populated with all four phase-2 fixtures: source (generator script + font), license (MIT for project-owned generated artefacts; embedded font subset rights documented). | `tests/fixtures/pdfs/LICENSES.md`. |

**Disagreements with Codex**: none.

## Claude (COMMENT) — addressed

| Claude point | Action | Where |
|---|---|---|
| `LICENSES.md` not updated for phase-2 fixtures | Same as codex point above — populated. | `tests/fixtures/pdfs/LICENSES.md`. |
| `requires-python` bumped from spec/plan's >=3.10 to >=3.11 without documented justification | Added a comment block on the `requires-python` line citing the `tomllib`-stdlib boundary (the original phase-1 reason) and pointing back to plan / Expert-Review iteration 1 where the bump was accepted. | `pyproject.toml`. |
| Fixtures named "ar" but contain only Latin text | `lorem-ar-real.pdf` now carries real Arabic codepoints when an Arabic-capable TTF is locally available. The other three fixtures (2col, mixed, table) remain Latin — their job is to exercise the extractor's geometry, not Arabic typesetting. The split is documented in both `tools/generate_fixtures.py` and `tests/fixtures/pdfs/LICENSES.md`. Phase 9 will land bidi-shaped real Arabic content for the full corpus. | `tools/generate_fixtures.py`, `tests/fixtures/pdfs/LICENSES.md`. |

**Disagreements with Claude**: none.

## Gemini Pro — SKIPPED (upstream infrastructure)

Gemini quota was exhausted on every attempt across the project (spec, plan, phase 1, phase 2). Two attempts in this iteration; both returned `HTTP 429`. Skipped per architect rule. See `1-phase_2_…-iter1-gemini.txt`.

## Verification after revisions

- `make lint` clean (ruff check + format check)
- `make audit` `license_audit: OK`
- `make test` **64 tests pass** (up from 62), 91% coverage
  - New: `test_extract_native_real_arabic_codepoints_round_trip` (skips when no Arabic TTF locally)
  - New: `test_line_grouping_preserves_pdf_stream_order` (asserts the RTL fix without depending on a real PDF)

## Summary

- Codex `REQUEST_CHANGES` → addressed in full (4/4 issues, including the RTL bug Codex correctly flagged as the most important).
- Claude `COMMENT` → addressed in full (3/3 issues).
- Gemini → skipped per architect instruction.
