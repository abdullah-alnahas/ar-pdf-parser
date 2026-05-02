# Phase 7 (Emitters: Markdown + Word) — Iter1 Rebuttals

**Phase**: `phase_7_emitters_markdown_and_docx`
**Iteration**: 1
**Reviewers**: codex (REQUEST_CHANGES, HIGH), claude (APPROVE, HIGH), gemini (SKIPPED — quota)

## Verdict Summary

| Reviewer | Verdict | Confidence | Notes |
|---|---|---|---|
| codex | REQUEST_CHANGES | HIGH | Found one real bug: orphan grouped captions silently dropped |
| claude | APPROVE | HIGH | No bugs; 101 tests cover spec criteria |
| gemini | SKIPPED | N/A | API quota exhausted (10 retry attempts) — same as phases 4–6 |

Net: **2/3 reviewers reachable; 1 valid REQUEST_CHANGES**. Address codex's bug, then proceed.

## codex — REQUEST_CHANGES

### Issue 1 — Orphan grouped captions dropped (markdown.py + docx.py)

> `markdown.py:103-129` (and the equivalent in `docx.py:67-78`) suppresses any `CAPTION` with a non-`None` `group_id`, even if no `FIGURE` with that `group_id` exists … contradicts the intended "caption suppressed because it was emitted with the figure" behavior.

**ACCEPTED — fixed.**

Root cause: `_collect_grouped_caption_ids` returned the set of `group_id`s of grouped *captions* and used that to suppress them, on the assumption that every grouped caption has a paired figure. That assumption breaks when phase 6 stamps `group_id` on a caption but the upstream figure was filtered out (e.g. `RegionRole.HEADER_FOOTER` reclassification, image-extraction failure, dead figure region).

Fix:
- Renamed both helpers to `_collect_paired_caption_group_ids(regions)` returning `figure_ids ∩ caption_ids` — only group ids that have **both** a figure and a caption present in the region stream are suppressed.
- Orphan grouped captions now flow through the normal caption renderer (italic paragraph in MD; italic run in docx).
- `_collect_caption_texts` in docx.py now also takes the `paired` set so it only registers caption text for groups that will actually be consumed by a figure.

### Issue 2 — Tests miss the orphan-caption edge case

**ACCEPTED — added regression tests.**

- `tests/test_emit_markdown.py::test_orphan_grouped_caption_still_renders` — orphan grouped caption renders as `*orphan*`.
- `tests/test_emit_docx.py::test_orphan_grouped_caption_still_renders` — orphan grouped caption renders as one italic-run paragraph.

Verification: 304 tests pass (was 302, +2 regression tests). Lint clean. License audit clean. Determinism + lazy-import tests still green.

## claude — APPROVE

No actionable issues. Confirmed:
- Markdown snapshot byte-identical
- docx structural test byte-identical (document.xml)
- Network-isolation socket-mock test passes
- 101 phase-7 tests across 6 modules
- Lazy `python-docx` import discipline preserved (subprocess test)
- HTML-comment `--` sanitisation in failure placeholders

## gemini — SKIPPED

`429 RetryableQuotaError: You have exhausted your capacity on this model.` after 10 retry attempts (~275s). Identical pattern to phases 4–6; architect previously approved 2/3 acceptance when gemini quota is hard-blocked.

## Final State

- **Bug fix**: orphan grouped captions no longer dropped (markdown + docx).
- **Tests added**: 2 regression tests targeting the codex issue.
- **Suite**: 304 passed, 2 deselected.
- **Lint / format**: clean.
- **License audit**: clean.

Ready for PR.
