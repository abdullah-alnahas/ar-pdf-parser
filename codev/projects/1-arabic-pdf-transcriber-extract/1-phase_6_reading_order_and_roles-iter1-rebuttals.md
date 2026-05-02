# Phase 6 (RTL Reorder + Role Classification) — Iteration 1 Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where |
|---|---|---|
| Native headings without `font_size` metadata are incorrectly classified from bbox height instead of falling back to H2. The spec/plan require native-path headings with no font-size signal to default to H2. Height is NOT a substitute on the native path because PDF text-extractor bboxes track glyph extents, not visual heading prominence (a long heading line would be ranked taller than a short one). | `_size_signal_for` is now source-sensitive: returns `None` for `RegionSource.NATIVE` regions whose `meta["font_size"]` is missing, which routes them through the no-signal-→-H2 fallback. ML / OCR regions still use bbox height as the signal (the only signal they have). | `src/arabic_pdf_transcribe/roles/classify.py` (`_size_signal_for`). |
| Test suite reinforced the same spec violation by expecting height-based heading levels on `RegionSource.NATIVE` regions. Needed source-sensitive split: native-without-font-size → all H2; OCR/ML headings may use height-based inference. | Renamed `test_heading_levels_from_region_height_when_no_meta` to `test_heading_levels_from_region_height_on_ml_path` and switched its source to `RegionSource.OCR`. Added `test_native_headings_without_font_size_meta_default_to_h2` asserting native-path headings with varying bbox heights all emit H2 when no `font_size` meta is present. | `tests/test_roles_classify.py`. |

**Disagreements with Codex**: none.

## Claude (APPROVE) — minor nits addressed

| Claude observation | Disposition |
|---|---|
| `StatisticsError` import at line 159 (after the function that uses it, with `# noqa: E402`). Works fine but unconventional — should be at the top with `from statistics import quantiles`. | Fixed: imported alongside `quantiles` at module top; `# noqa: E402` removed. |
| `page_height` unused in `reorder()`; explicitly silenced with `_ = page_height` and documented as reserved. | Accepted (kept for API symmetry with `classify_page`; phase 9 may use it for page-shape-aware reordering). |
| `uuid5` for `group_id` is better than the plan's suggested `f"p{page_index}-g{counter}"` — deterministic from bbox coordinates, no mutable counter to thread. Spec only requires "stable identifier"; this satisfies it. | Accepted (no action). |
| Two figures with identical `page_index` + `bbox.x0` + `bbox.y0` would generate the same `group_id`. Extremely unlikely in practice. | Accepted; phase 9 corpus may surface real cases. |

## Gemini Pro — SKIPPED (upstream unavailable)

Ninth upstream failure in this project. Skipped per architect rule
[2026-05-01T18:32:24.195Z]. See
`1-phase_6_reading_order_and_roles-iter1-gemini.txt`.

## Verification after revisions

- `make lint` clean (ruff check + format check)
- `make audit` `license_audit: OK`
- `make test` **201 pass, ~94% coverage** (44 new phase-6 tests; 2 deselected `@slow`).
  - New: `test_native_headings_without_font_size_meta_default_to_h2`
  - Renamed: `test_heading_levels_from_region_height_on_ml_path` (source = OCR)

## Summary

- Codex `REQUEST_CHANGES` → addressed in full (1 spec violation in
  source + 1 corresponding test gap, both fixed).
- Claude `APPROVE` → 1 of 4 nits addressed in code (import
  placement); 3 accepted as documented design choices.
- Gemini → skipped per architect instruction.
