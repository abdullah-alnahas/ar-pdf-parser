# Review: bugfix-16 — real-arabic-pdf-still-fails-after-v0.1.2

## Issue

GitHub #16 — running the real Foulabook Arabic PDF on v0.1.2 still
produced `ok=1 failed=6` even after issue #14's `--prefetch-models`
landed. Page 1 emitted only the figure placeholder
(`![figure on page 1](#)`); pages 2-7 failed with
`{"reason": "ModelDownloadError"}` carrying no actionable detail.

## Root Causes

Three independent regressions, piled on top of each other:

1. **`transformers==4.46.3` predates the GOT-OCR-2.0 `got_ocr2`
   model type.** The `GotOcr2ForConditionalGeneration` class
   (and the `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING` registration for
   `model_type='got_ocr2'`) only landed in transformers 4.49.0
   (Feb 2025). `AutoProcessor.from_pretrained` partially succeeded
   (returned a tokenizer-only `PreTrainedTokenizerFast`); the
   immediate `AutoModelForImageTextToText.from_pretrained` then raised
   `ValueError: model type 'got_ocr2' not recognized`. The HF cache
   confirmed the regression: only the 18 MB tokenizer blobs were
   present — the model weights never came down because transformers
   refused to instantiate the config.

2. **`Formula → UNKNOWN` flooded the log with spurious warnings.**
   The default DiT layout model
   (`cmarkea/dit-base-layout-detection`) was trained primarily on
   English research papers and frequently mislabels justified
   Arabic body text as `Formula`. `layout/_classes.py` mapped
   `Formula → RegionRole.UNKNOWN`. **The text was not actually
   dropped** — the markdown emitter renders both UNKNOWN and
   PARAGRAPH through the same paragraph renderer, and the role
   classifier preserves UNKNOWN regions — but every mislabelled
   region triggered the hf_detector's
   `"label 'Formula' mapped to UNKNOWN; phase 6 will handle it"`
   log warning (39 such warnings on a single Foulabook page). The
   visible "missing body text" the issue reporter saw was actually
   caused by RC#1: with the OCR model failing to load, every
   ML-branch page collapsed into a `FAILURE_PLACEHOLDER`, wiping
   the (correctly captured) layout regions including those
   labelled UNKNOWN. The remap to PARAGRAPH is still warranted: it
   is semantically more accurate (this *is* body text, not
   unknown content) and removes the noise that caused the
   misdiagnosis in the first place. (Codex CMAP iter-2 caught the
   incorrect "silently dropped" claim; rationale corrected here
   and in the code comment.)

3. **JSON-log failures lost the actionable hint.** Pipeline's
   `ArabicPdfTranscribeError` arm recorded only `type(exc).__name__`
   in `failure_reason`. `ModelDownloadError`'s message already
   contained the recovery hint ("prefetch the weights with:
   arabic-pdf-transcribe --prefetch-models"), but the user saw only
   `{"reason": "ModelDownloadError"}` and no path forward. Existing
   catch-all + RuntimeError arms already used the
   `f"{ClassName}:{exc}"` format — only the typed-error arm was
   inconsistent.

## Fixes

| RC | File                                         | Change                                                |
|----|----------------------------------------------|-------------------------------------------------------|
| 1  | `pyproject.toml`                             | `transformers==4.46.3` → `transformers>=4.49,<4.50`   |
| 2  | `src/arabic_pdf_transcribe/layout/_classes.py` | `"Formula": RegionRole.UNKNOWN` → `RegionRole.PARAGRAPH` |
| 3  | `src/arabic_pdf_transcribe/pipeline.py`      | `reason = type(exc).__name__` → `f"{type(exc).__name__}:{exc}"` |

Total: ~190 LOC across 6 files (3 src/, 3 test/), one .gitignore
line. Within the BUGFIX 300-LOC budget.

## Tests

New file: `tests/test_issue_16_regression.py` (5 tests).

1. `test_pyproject_pins_transformers_supporting_got_ocr2` — parses
   `pyproject.toml`, asserts the lower bound is ≥4.49. Pure
   text-based; no network; protects against accidental downgrade.
2. `test_got_ocr2_class_is_registered_in_auto_image_text_to_text` —
   inspects `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` for the
   exact failing API path. No weights, no network, no HF cache
   needed — just the mapping that auto-classes resolve at import
   time. (Strengthened in iteration 2 after Codex flagged that the
   original config-only smoke did not exercise the registration that
   actually exploded.)
3. `test_got_ocr2_config_instantiates_when_transformers_installed` —
   `AutoConfig.from_pretrained(..., local_files_only=True)`. Skipped
   when the cache is empty; meaningful in the dev environment.
4. `test_formula_label_maps_to_paragraph_not_unknown` — guards the
   `_classes.py` mapping against future regressions.
5. `test_json_failure_event_includes_exception_message` — exercises
   the JSON serialiser with a `ModelDownloadError`-shaped reason
   string, asserts the actionable hint reaches the log line.
6. `test_formula_label_does_not_log_unknown_warning_on_detection`
   (added in iter-3) — end-to-end stub run through
   `HFDiTLayoutDetector.detect`: a Formula-labelled segmentation
   must not log `"mapped to UNKNOWN"`, must produce
   PARAGRAPH-roled regions, and the transcribed text must reach
   the markdown emitter's output.

Updated: `tests/test_layout_protocol.py` parametrize entry for
`Formula`; `tests/test_pipeline.py` failure-reason assertion (now
checks the `<class>:<msg>` format).

Full suite: **434 passed** (428 baseline + 6 new). Ruff clean.

## Out of Scope (Deferred)

- **Figure-only pages** still emit `![figure on page N](#)` (dead
  href). Embedding crops requires `Region` to carry image data — an
  architectural change well past the BUGFIX budget. Already a v0.2.0
  backlog item; recommend promoting.
- **Post-prefetch weight-size integrity check.** Issue #16 listed
  this as RC#5; with the new transformers pin, `from_pretrained`
  refuses to instantiate the wrong `model_type`, closing the
  silent-partial-success failure mode that motivated the check. The
  defensive size verification would now be belt-and-suspenders for
  a closed failure surface; deferred.
- **Real Arabic OCR quality benchmarks** on the Foulabook PDF (CER,
  per-page accuracy). Out of scope per the issue body's "Out of
  scope" list; defer to v0.2.0 corpus expansion.

## CMAP Verdicts

| Reviewer | Iter | Verdict | Notes |
|----------|------|---------|-------|
| Claude   | 1    | APPROVE | HIGH confidence, no key issues |
| Codex    | 1    | REQUEST_CHANGES → addressed | (a) RC#1 smoke only exercised `AutoConfig`, not the failing `AutoModelForImageTextToText` path. (b) Missing `codev/reviews/16-*.md`. (c) Untracked `.claude/scheduled_tasks.lock`. All addressed in iter-2. |
| Codex    | 2    | REQUEST_CHANGES → addressed | (a) RC#2 "silently dropped" claim was wrong — UNKNOWN renders as paragraph; rationale corrected and a stronger end-to-end regression test added. (b) Commit-message format concern (`[Spec N]...`) — past bugfixes (#12, #14) use the same `Fix #N: ...` format this PR uses, so the convention is bugfix-specific and the commits are consistent with project history. |
| Gemini   | 1    | N/A | Quota-exhausted on every retry (10 attempts); reviewer infrastructure failure, not a content verdict. |

## Lessons Learned

- **A pinned dependency that "works" in CI is not the same as a
  pinned dependency that works for the model.** The phase-5 selector
  for GOT-OCR-2.0 was correct in spirit, but the tests stub the
  `transformers` API with fakes — they never exercised the real
  model-class registry. A single-line "is `got_ocr2` in the
  auto-class mapping?" smoke would have caught this on day one.
  This regression test is now permanent.
- **Silent UNKNOWN regions are a footgun.** When a layout model is
  trained on a different corpus than the input language, frequent
  mislabelling will dump real content into the UNKNOWN bucket. For
  Arabic-first usage, the safer default is to pass UNKNOWN through
  as PARAGRAPH (so the body text survives) rather than drop it.
  Worth considering a tighter logging contract for v0.2.0: warn
  *and* count, then degrade to PARAGRAPH above a per-page
  threshold.
- **The first piece of information a user sees on failure has to be
  actionable.** Class-name-only reasons forced the user to grep
  source for the recovery hint. The PR aligns the typed-error arm
  with the pre-existing pattern in two other arms — consistency
  across the catch-block table matters.
