# Phase 5 (OCR Transcription Adapter) — Iteration 1 Rebuttals

## Codex (APPROVE) — no changes required

> Phase 5 is implemented cleanly: the OCR protocol, HF adapter, model
> pinning, confidence handling, lazy imports, cropping, and
> adapter-focused tests all align with the plan.

No key issues; no changes required.

## Claude (APPROVE) — no changes required

Verified all plan deliverables:

- `OCRTranscriber` Protocol with `runtime_checkable`, correct signature.
- `HFGotOCRTranscriber` adapter with model selection rationale documented.
- `_crop.py` with padding + page-bounds clipping + RGB conversion.
- `models.toml` updated with Apache-2.0 GOT-OCR-2.0 entry pinned by SHA.
- Confidence reporting in `[0, 1]` via geometric mean of softmax probs.
- 19 stub tests + 1 `@pytest.mark.slow` real-model + 3 lazy-import tests.

Coverage: 93% on OCR modules. Uncovered lines are `pragma: no cover`
slow paths (real model loading, real transformers import failure).

Five non-blocking observations, all acknowledged:

| Claude observation | Disposition |
|---|---|
| Model selection rationale well-documented; sound reasoning. | Accepted (no action). |
| Confidence computation correctly handles edge cases (no scores, count=0). | Accepted (no action). |
| Degenerate cell crop silently returns empty for table cells vs raises for top-level regions. | Accepted — phase-9 review may revisit if real-corpus data shows the silent path swallows real failures. |
| `processor(images=image, ...)` keyword-arg pattern is processor-specific; only the slow test catches API mismatches. | Accepted as inherent stub-test trade-off. The slow test gates real-model behaviour pre-merge. |
| `_ensure_loaded` broad `except Exception` is pragmatic for model loading where failure modes are diverse; wraps cleanly into `ModelDownloadError` with the `huggingface-cli download` hint. | Accepted (no action). |

## Gemini Pro — SKIPPED (upstream unavailable)

Eighth upstream failure in this project. Phase 5 attempt timed out
after both codex (87s) and claude (222s) had completed. Skipped per
architect rule [2026-05-01T18:32:24.195Z]. See
`1-phase_5_ocr_transcription_adapter-iter1-gemini.txt`.

## Verification

- `make lint` clean (ruff check + format check)
- `make audit` `license_audit: OK` (GOT-OCR-2.0 Apache-2.0 entry passes)
- `make test` **157 pass, 94% coverage** (21 new OCR tests across 4
  files; 2 slow tests deselected by default).
- All pre-existing phase-1/2/3/4 tests still pass.

## Summary

- Codex `APPROVE` → no changes.
- Claude `APPROVE` → no changes.
- Gemini → skipped per architect instruction.

Iteration-1 implementation lands as-is on PR #6.
