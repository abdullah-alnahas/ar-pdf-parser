# Plan Iteration 1 — Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where in plan |
|---|---|---|
| `Region` schema too thin (no heading level, list ordinal, table grid, caption-figure linkage, failure placeholder) | Region rewritten with `heading_level: int \| None`, `list_marker: ListMarker \| None` (with ordinal + raw marker), `table_grid: TableGrid \| None`, `group_id: str \| None` for caption-figure linkage, `failure_reason: str \| None`, plus a new `failure_placeholder` role and `REGION_SCHEMA_VERSION = "1"`. | Phase 2 deliverable for `regions.py` rewritten. |
| Table support not end-to-end | Native-table detection added to phase 2 (`extract/_native_tables.py`). ML-branch table cell detection added to phase 4 (`layout/_table_cells.py`). Per-cell OCR specified in phase 5. Emitters in phase 7 consume `table_grid` directly with explicit cell-rendering rules; phase 7 also notes that the emitters never see partial grids — phase 4 + phase 6 guarantee `table_grid is not None` whenever `role == table`. | Phases 2, 4, 5, 7 updated. |
| Phase 8 dependency inconsistency (says "stubs for 4–5" but production wiring uses real adapters) | Dependencies rewritten to "Phases 1–7" with an explicit note: production wiring uses the real HF adapters from phases 4 & 5; unit tests inject Protocol-conformant stubs. | Phase 8 dependencies line. |
| Validator + emitter not exposed as pluggable hooks (only layout + OCR were) | `pipeline.transcribe(...)` signature now includes `validator`, `markdown_emitter`, `docx_emitter` kwargs alongside `layout_detector` / `ocr_transcriber`. Each maps to a Protocol; defaults wired at package level. Spec's "library API hooks" requirement explicitly addressed. | Phase 8 deliverable for `pipeline.py`. |
| Phase 6 H2/H3 contradiction (impl says H3, AC says H2; spec says H2) | Single rule: when no size signal exists, `heading_level == 2`. Implementation text and acceptance criterion now agree; matches spec. | Phase 6 implementation details + acceptance criteria. |

**Disagreements with Codex**: none. All five `KEY_ISSUES` accepted.

## Claude (COMMENT) — addressed (none was a blocker)

| Claude point | Action | Where in plan |
|---|---|---|
| `errors.py` deferred to phase 8 but referenced in phases 4–5 | `errors.py` created in phase 1 with `ModelDownloadError` and `OCRTranscriptionError`; phase 8 extends it with the rest. | Phase 1 deliverable + phase 8 errors note. |
| Heading-level contradiction (H2 vs H3 for missing font-size) | Same fix as Codex point above — single rule, H2. | Phase 6. |
| CER tolerance mismatch (plan said 0.10, spec says 0.05) | Phase 9 e2e changed to ≤ 0.05. | Phase 9 deliverable. |
| Mixed Arabic/Latin fixture not covered until phase 9 | Added `lorem-ar-en-mixed.pdf` to phase 2 fixtures so spec test scenario 2 is exercisable from phase 2 onward. | Phase 2 deliverables. |
| Dependency diagram shows Phase 4 → Phase 1 (should be → Phase 2) | Diagram redrawn — Phase 4 now correctly depends on Phase 2. | Dependency Map section. |
| `--max-workers` and `--config` underspecified | Both spelled out in phase 8 deliverables: `--max-workers` (default 1 on CPU, `auto` selects `min(cpu_count, 4)`); `--config` is a TOML file with `[validator] / [layout] / [ocr] / [render]` sections, also accepts `AR_PDF_*` env vars. | Phase 8 deliverable for `cli.py`. |
| Spec open questions (Arabic varieties, handwriting, etc.) not closed in plan | New "Plan-Level Assumptions" section at the top of the plan closes them: MSA primary, Classical stretch, dialectal handwriting OOS; tables = basic grid only; figures = placeholder; equations not OCR'd; `--pages 1-3,7,10-12` syntax; `--debug-json` for confidences. | New top-level section. |

**Disagreements with Claude**: none.

## Gemini Pro — SKIPPED (upstream infrastructure)

Gemini Pro quota was exhausted on every attempt across both spec and plan phases. Per architect instructions:

> [ARCHITECT INSTRUCTION | 2026-05-01T18:32:24.195Z] If still quota-blocked, skip — 2/3 acceptable when 3rd is infra-blocked. Document in spec that gemini was unavailable.
> [ARCHITECT INSTRUCTION | 2026-05-01T18:40:20.304Z] Re-attempt gemini in plan-phase consultation per spec note; if quota still blocked, proceed 2/3.

Action taken:
- Codex feedback fully addressed (table above).
- Claude feedback fully addressed (table above).
- Gemini retried once after the address pass; same `HTTP 429`. Skipped per architect rule.
- Gemini unavailability is documented in the **Expert Review** section of the plan.
- The `1-plan-iter1-gemini.txt` file records the SKIPPED status with explanation.

This is **not a substantive rebuttal** to a Gemini review (no review was produced).

## Summary

- Codex `REQUEST_CHANGES` → addressed in full (5/5 issues).
- Claude `COMMENT` → addressed in full (7/7 issues).
- Gemini → skipped per architect instruction; documented in plan and consult-output file.
- Plan committed: `[Spec 1] Plan with multi-agent review`.
