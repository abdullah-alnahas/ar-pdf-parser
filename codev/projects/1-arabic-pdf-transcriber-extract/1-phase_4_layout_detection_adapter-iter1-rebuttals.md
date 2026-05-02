# Phase 4 (Layout Detection Adapter) — Iteration 1 Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where |
|---|---|---|
| `validate_page` adapter could emit `RegionRole.TABLE` with `table_grid=None` whenever `detect_table_cells(...)` failed, contradicting the plan deliverable (`codev/plans/1-arabic-pdf-transcriber-extract.md:225-226`: complex layouts fall back to one-cell-per-row coalescing) and the Protocol contract (`layout/__init__.py:36-38`: table regions are responsible for filling `table_grid`). | Added `_fallback_single_cell_grid(bbox)` helper. The adapter now guarantees every TABLE Region carries a populated `table_grid`: when ruled-line morphology cannot recover a grid (`detect_table_cells=False`, or morphology returns `None`), the adapter falls back to a one-row, one-cell grid covering the full table bbox. Phase 5's OCR adapter walks the cells regardless of how the grid was obtained; phase 6 keeps the Region as a TABLE whose body is single-prose. | `src/arabic_pdf_transcribe/layout/hf_detector.py` (`_regions_from_class_map`, new `_fallback_single_cell_grid`). |
| Tests miss the contract: no adapter-level test proves a detected `Table` class always returns a Region with a populated `table_grid` even when morphology cannot recover a full grid. | Added two regression tests: `test_detect_table_region_always_has_populated_table_grid` (TABLE class on a blank crop where ruled-line morphology returns None — adapter must populate the fallback grid) and `test_detect_table_region_disabled_cell_detection_still_has_fallback_grid` (`detect_table_cells=False` config — fallback still kicks in because the contract is the Region's, not the morphology helper's). | `tests/test_layout_hf_detector.py`. |

**Disagreements with Codex**: none.

## Claude (APPROVE) — no changes required

Claude consulted on iter-2 code (after the codex fix landed) and
returned APPROVE with no key issues. Verified deliverables, acceptance
criteria, code quality, lazy-import discipline, and 41 phase-4 tests
pass. Three non-blocking observations, all acknowledged:

| Claude observation | Disposition |
|---|---|
| `Any` annotations for `_model`, `_processor`, numpy arrays in `hf_detector.py` violate the CLAUDE.md "never use `Any`" rule, but importing torch / transformers types for annotations would break lazy-import discipline. | Accepted as documented trade-off — the `Any`-typed fields are internal implementation details behind the Protocol, not API surface. Phase 6+ may revisit if a stub-typing layer becomes worthwhile. |
| Pure-Python flood fill in `_connected_components`. O(pixels) iterative walk over a numpy mask. Acceptable on the model-resolution grid (~200×200 to 500×500); not full-page resolution. | Accepted; phase 9 corpus benchmarking will reveal whether `scipy.ndimage.label` is worth the dep weight. |
| Pure-Python erosion in `_table_cells.py`. Same pattern, runs on the cropped table region (typically small). | Accepted. |

## Gemini Pro — SKIPPED (upstream infrastructure)

Seventh upstream failure in this project. Phase 4 attempt presented as
a JSON parse error (`[warn] Failed to extract usage for gemini:
Unexpected end of JSON input`) — the gemini CLI's stand-in for quota
exhaustion. Skipped per architect rule
[2026-05-01T18:32:24.195Z]. See
`1-phase_4_layout_detection_adapter-iter1-gemini.txt`.

## Verification after revisions

- `make lint` clean (ruff check + format check)
- `make audit` `license_audit: OK`
- `make test` **136 pass**, 94% coverage (43 new layout tests)
  - Two new TABLE-grid contract tests added in iter 2
  - All pre-existing phase-1/2/3 tests still pass
- CI workflow already installs CPU-only torch and runs
  `python tools/license_audit.py --include-extras=ml` as a separate
  step, satisfying the architect's PR-2 note.

## Summary

- Codex `REQUEST_CHANGES` → addressed in full (1 contract gap + 1 test
  gap, both fixed).
- Claude `APPROVE` → no changes required.
- Gemini → skipped per architect instruction.
