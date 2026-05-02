# Phase 8 (CLI + pipeline orchestrator) — Iter1 Rebuttals

**Phase**: `phase_8_cli_and_pipeline_orchestrator`
**Iteration**: 1
**Reviewers**: codex (REQUEST_CHANGES, HIGH), claude (REQUEST_CHANGES, HIGH), gemini (SKIPPED — quota)

## Verdict Summary

| Reviewer | Verdict | Confidence | Notes |
|---|---|---|---|
| codex | REQUEST_CHANGES | HIGH | 4 issues (pages filter, max_workers, format-docx-no-output, GPU OOM) |
| claude | REQUEST_CHANGES | HIGH | Overlapping issues (docx-stdout crash, max_workers silent discard, CUDA OOM, exit-5 test) |
| gemini | SKIPPED | N/A | API quota exhausted (10 retries) — same as phases 4–7 |

Net: **2/3 reviewers reachable**, both REQUEST_CHANGES with strongly overlapping concrete issues. All accepted and addressed.

## Issues addressed

### 1. `--pages` filtered too late (codex)

> `transcribe()` materializes `list(extract_native(pdf_path))` for the whole PDF before filtering.

**Fixed.**

- Replaced the pre-materialised list with a streaming pass: `_iter_selected_native_pages(pdf_path, selected)` yields only pages in the filter set; non-selected pages never reach the validator, layout detector, or OCR.
- Total page count comes from a one-shot `_count_pages(pdf_path)` that opens the document just to read `len(document)` (encrypted/corrupted errors still surface here at the boundary).
- Regression test added: `test_pages_filter_does_not_invoke_validator_for_skipped_pages` — spies on the validator and asserts it sees only selected page indices.

### 2. `--max-workers` parsed but discarded (codex + claude)

> CLI parses, validates, and stores `max_workers`, but `transcribe()` has no `max_workers` parameter and `main()` never passes it.

**Fixed (with documented v1 limitation).**

- Added `max_workers: int = 1` to `transcribe()`; CLI now passes `args.max_workers` through.
- Validates `max_workers >= 1` (raises `ValueError`).
- v1 keeps the loop sequential to bound peak RSS — the parameter is plumbed through so the **CLI surface is stable**; phase 9 will enable parallel ML page processing after the benchmark corpus is in place. Documented in the `transcribe()` docstring.
- Regression tests added: `test_max_workers_accepted_in_signature`, `test_max_workers_invalid_raises_value_error`.

### 3. `--format docx` without `-o` raised in output writer (codex + claude)

> `_write_output()` raises `FormatExtensionMismatch` after `main()`'s exception handler block — user sees a traceback, not a clean exit code.

**Fixed.**

- Check moved up to `validate_args(ns)` — the conflict is now caught alongside the `--format docx -o out.md` mismatch and surfaces as exit 4 cleanly.
- `_write_output` keeps an `assert args.output is not None` for the docx branch as a contract check.
- Regression test added: `test_format_docx_without_output_returns_exit_4`.

### 4. `torch.cuda.OutOfMemoryError` not caught (codex + claude)

> Plan requires catching MemoryError **and** torch.cuda.OutOfMemoryError; only `MemoryError` is caught.

**Fixed.**

- Added `_is_cuda_oom(exc)` that detects the synthetic class by `__module__ == "torch"` and `__name__ == "OutOfMemoryError"` — no `import torch` required, so the orchestrator stays light.
- A new `except RuntimeError` arm dispatches: CUDA OOM raises `OutOfMemoryDuringInference` in strict mode (with the spec-required hint); other `RuntimeError`s flow into the generic per-page failure path.
- Regression tests added: `test_cuda_oom_strict_raises_typed_oom`, `test_cuda_oom_best_effort_synthesises_failure`.

### 5. No CLI test for exit code 5 (claude)

> Spec defines `ModelDownloadError → exit 5`; CLI maps it but no integration test exercises the path.

**Fixed.** `test_model_download_error_returns_exit_5` patches `transcribe` to raise `ModelDownloadError` and asserts the CLI returns `EXIT_MODEL_MISSING`.

## Out-of-scope (acknowledged, deferred)

- `--config` only loads `[validator]`. Both reviewers flagged this; claude marked it as "documented TODO". The plan section reads `[validator] | [layout] | [ocr] | [render]` schema but the layout / OCR adapters already accept their config classes via constructor arguments — wiring them through TOML is mechanical but adds surface area beyond the phase-8 acceptance criteria. Will land in phase 9 alongside the README docs that the schema appears in.
- gemini SKIPPED: same architect-approved 2/3 acceptance as phases 4–7.

## Final state

- **Tests**: 363 passed, 2 deselected (was 356; +7 regression tests for the 4 codex/claude issues + exit-5).
- **Lint / format**: clean.
- **License audit**: clean.
- **CLI smoke**: `arabic-pdf-transcribe FILE` writes Arabic Markdown to stdout; `-o out.docx` writes a valid Word file; `--quiet`, `--json-logs`, `--pages`, `--strict`, `--debug-json`, `--format` all wired and tested.

Ready for PR.
