# Iteration 1 — Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where in spec |
|---|---|---|
| Unresolved "Critical (Blocks Progress)" items (license, validator, packaging) | Resolved with explicit defaults: MIT license, multi-signal validator (codepoint ratio + replacement-glyph ceiling + word-boundary plausibility), HF cache download-on-first-use. | New section: **Resolved Decisions**. Old "Critical" subsection now empty (defer-to-review only). |
| Reproducibility internally inconsistent (byte-identical vs CER-threshold for ML) | Split: native path = byte-identical; ML path = CER tolerance ≤ 0.05. PDF-library versions added to the pin-list. | **Resolved Decisions → Reproducibility scope**; **Success Criteria** (split into native and ML bullets); **Constraints → Technical**. |
| Semantic output requirements underspecified (headings, lists, tables) | Added v1 region-classification contract (`heading` / `paragraph` / `list-item` / `table` / `figure` / `caption` / `header-footer`), heading-level inference rules (cap at H3), list-prefix detection, basic-grid tables. | **Resolved Decisions → Semantic output contract (v1 minimum)**. |
| Failure behaviour missing (encrypted, corrupted, OOM, offline-cache-miss, fail-fast vs best-effort, exit codes) | Added explicit behaviour for every named scenario, with CLI exit codes `0/2/3/4/5`, `--strict` opt-in, default best-effort with placeholder regions. | **Resolved Decisions → Failure behaviour and CLI exit codes**. |
| Acceptance criteria need a precision pass on test fixtures, "RTL-correct", "non-empty" | Added a **Benchmark Corpus** section (paths, flavours, ≥-N counts, license file). Operationalised "RTL-correct" in test scenario 6 (right-most-first within row band). Replaced "non-empty" with concrete CER bound for ML-path tests. | New section: **Benchmark Corpus**; **Test Scenarios → Functional Tests** (scenarios 3, 6 rewritten; 11–18 added). |

**Disagreements with Codex**: none. All five `KEY_ISSUES` accepted.

## Claude (COMMENT) — addressed (none was a blocker; included anyway)

| Claude point | Action | Where in spec |
|---|---|---|
| No error-handling contract for partial page failures | Added best-effort default + `--strict` with placeholder Markdown comment / styled Word paragraph. | **Resolved Decisions → Failure behaviour and CLI exit codes**. |
| No progress / logging spec for long-running ML | Added stderr progress (page X of N), `--quiet`, `--json-logs`. | **Resolved Decisions → Progress reporting**. |
| CLI format-selection mechanism unspecified | Stated: extension drives format if present; `--format` otherwise; conflicting combinations are an error (exit `4`). | **Resolved Decisions → CLI format selection**; **Test Scenario 18**. |
| Missing test scenarios (empty, password-protected, intra-region bidi) | Added scenarios 11, 12, 13, 16, 17, 18. | **Test Scenarios → Functional Tests**. |
| Temp file cleanup not mentioned | Added process-scoped `TemporaryDirectory` constraint. | **Resolved Decisions → Temp file lifecycle**. |
| Reproducibility should pin PDF-library versions, not just model weights | Added PDF-extraction libraries to the pin-list. | **Resolved Decisions → Reproducibility scope**; **Success Criteria**. |

**Disagreements with Claude**: none.

## Gemini Pro — SKIPPED (upstream infrastructure)

The Gemini consultation was attempted twice (initial run and retry, ~4 minutes each, 10 retries each). Both attempts returned `HTTP 429 — You have exhausted your capacity on this model.` and `consult` exited with code 1.

Per architect instruction `[ARCHITECT INSTRUCTION | 2026-05-01T18:32:24.195Z]`:

> Address codex REQUEST_CHANGES + claude comments first. Then retry gemini once. If still quota-blocked, skip — 2/3 acceptable when 3rd is infra-blocked. Document in spec that gemini was unavailable.

Action taken:
- Codex feedback fully addressed (table above).
- Claude feedback fully addressed (table above).
- Gemini retried once after the address pass; same `HTTP 429`. Skipped per architect rule.
- Gemini unavailability is documented in the **Expert Consultation** section of the spec.
- The `1-specify-iter1-gemini.txt` file records the SKIPPED status with explanation.

This is **not a substantive rebuttal** to a Gemini review (no review was produced). It is a record that the third reviewer slot was infra-blocked and the architect explicitly approved 2/3 acceptance.

## Summary

- Codex `REQUEST_CHANGES` → addressed in full.
- Claude `COMMENT` → addressed in full.
- Gemini → skipped per architect instruction; documented in spec and consult-output file.
- Spec committed: `[Spec 1] Specification with multi-agent review` (`c287522`).
