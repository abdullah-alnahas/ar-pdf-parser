# Phase 1 (Skeleton + License Audit) — Iteration 1 Rebuttals

## Codex (REQUEST_CHANGES) — addressed

| Codex point | Action | Where |
|---|---|---|
| Audit narrower than plan: "every dependency added in later phases is automatically vetted" but extras/dev are skipped | Implemented `--include-extras NAME` flag and a `SHIPPED_EXTRAS` set on the closure walker. Phase 1 default is empty (only `pypdfium2` + `python-docx` are shipped today); phases 4-5 will pass `--include-extras=ml` in CI when the `[ml]` extra carries pinned model deps and the documented overrides for the NVIDIA CUDA bundle (NVIDIA's redistributable EULA, not an SPDX license). Dev tools (`pytest`, `ruff`, `coverage`) remain out of scope by design — they never ship to end users and are not part of the project's license-compliance surface. | `tools/license_audit.py` (`SHIPPED_EXTRAS`, `--include-extras`); module docstring documents the policy. |
| `arabic-pdf-transcribe = arabic_pdf_transcribe.cli:main` declared but `cli.py` does not exist → broken metadata | Added `src/arabic_pdf_transcribe/cli.py` with a typed `main(argv: Sequence[str] \| None = None) -> int` that raises `NotImplementedError("…lands in plan phase 8 …")`. The console script now resolves cleanly during `pip install` and `--help` introspection; calling it before phase 8 fails loudly rather than silently producing empty output. | `src/arabic_pdf_transcribe/cli.py` + new test `test_cli_stub_raises_not_implemented`. |

**Disagreement with Codex (partial)**: Codex implied dev tools should be audited too. Disagree — dev deps are not shipped (build-time only) and an LGPL/GPL pytest plugin would be perfectly fine to use during testing. Audit policy explicitly scopes to *what we ship*; the harness accepts `--include-extras dev` for ad-hoc curiosity but won't run it in CI.

## Claude (REQUEST_CHANGES) — addressed

| Claude point | Action | Where |
|---|---|---|
| **[BLOCKER]** `tomllib` not in stdlib on Python 3.10; CI 3.10 leg would fail | Bumped `requires-python = ">=3.11"`; CI matrix updated to `["3.11", "3.12"]`; classifier list trimmed to 3.11 + 3.12. (`tomllib` is stdlib from 3.11.) | `pyproject.toml`, `.github/workflows/ci.yml`. |
| Classifier/CI mismatch: classifiers listed 3.12 but CI ran 3.10/3.11 | Same fix as above. CI now runs the same matrix declared in classifiers. | `pyproject.toml`, `.github/workflows/ci.yml`. |
| `_expression_license` lacked a direct unit test | Added `test_expression_license_splits_clauses` (parametrised over: trivial, `OR`, `AND`, nested-paren mixed, empty, None) and `test_extra_marker_extraction`. | `tests/test_license_audit.py`. |

**Disagreements with Claude**: none.

**Side note on Claude's "extra-marker filter is fragile" comment**: agreed — the new `_extra_marker` function uses a regex (`extra\s*==\s*["']([^"']+)["']`) and is unit-tested directly. The naive substring match was replaced.

## Gemini Pro — SKIPPED (upstream infrastructure)

Gemini Pro quota was exhausted on every consultation in this project (spec, plan, phase 1). Two attempts in this iteration; both returned `HTTP 429 — You have exhausted your capacity on this model.`

Per architect instruction `[ARCHITECT INSTRUCTION | 2026-05-01T18:32:24.195Z]`:
> "If still quota-blocked, skip — 2/3 acceptable when 3rd is infra-blocked."

Documented in `1-phase_1_skeleton_and_license_audit-iter1-gemini.txt`.

## Verification after revisions

- `make lint` — clean (ruff check + format check).
- `make audit` — `license_audit: OK` (runtime closure: `pypdfium2`, `python-docx`, `lxml`, `typing-extensions`).
- `make test` — **28 tests pass** (up from 20), **100 % coverage** on the package modules.
- `python tools/license_audit.py --include-extras=ml` — currently fails on the polluted local env's NVIDIA CUDA bundle, as expected. This is the correct failure mode: when phase 4 lands the `[ml]` deps it will also land the documented NVIDIA-EULA overrides.

## Summary

- Codex `REQUEST_CHANGES` → addressed (2/2 issues, 1 partial disagreement documented).
- Claude `REQUEST_CHANGES` → addressed (3/3 issues, including the `tomllib`/3.10 blocker).
- Gemini → skipped per architect instruction; documented.
