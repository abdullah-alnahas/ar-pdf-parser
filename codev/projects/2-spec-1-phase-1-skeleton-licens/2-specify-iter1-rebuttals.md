# Iteration 1 — Rebuttals (Specify phase)

## Claude (APPROVE) — minor observations addressed

Claude verified the spec against the codebase (existing `emit/docx.py`,
`pipeline.py`, `cli.py`, `python-docx==1.1.2` already pinned) and returned
`VERDICT: APPROVE` with one substantive but plan-scope observation and three
minor non-blocking notes.

| Claude point | Scope | Action |
|---|---|---|
| Phase 3 deliverables should explicitly state that `cli.py:_write_output()` is updated to call `emit_docx_pages(result.pages, ...)` instead of the backward-compat `emit_docx(result.regions, ...)`, otherwise the page-break feature is silently defeated in CLI mode. | Plan (not spec) | Forwarded to the implementation plan owner — this is a Plan-phase clarification, not a spec change. The spec's Desired State item 9 and Success Criteria already mandate hard page breaks "via a download button in the web UI **and via a documented CLI flag**", so the requirement is unambiguous at the spec level; the plan must not let the CLI fall through to a flat-region wrapper. |
| Phase 4 introduces an asyncio loop in the CLI; consider whether Phase 1 should pre-scaffold the async entry so Phase 4 only adds the queue. | Plan | Plan-level sequencing note. Spec is silent on whether the CLI runs an asyncio loop; that's a plan choice. No spec change. |
| Phase 6 commits `frontend/build/` to the repo — git history bloat. | Plan | Plan acknowledges this trade-off. Spec is silent on artifact distribution (Open Question item exists). No spec change. |
| Phase 5 test plan should explicitly cover the warm-hit `warming` event (`phase: "ready"` only, no preceding `loading`). | Plan / Test | Spec already pins this in Functional Test scenarios 8 and 9. No spec change; plan should mirror it in Phase 5's test list. |

**Disagreements with Claude**: none. All four points are accepted; three are
plan-level forward-references and one (Phase 3 CLI wiring) is already implicitly
required by the spec's Success Criteria — the plan must make it explicit.

## Codex (GPT-5) — SKIPPED (upstream quota)

The codex consultation invocation returned:

> "You've hit your usage limit for premium. Try again at May 8th, 2026 9:27 PM."

This matches the project-wide pattern already documented in
`codev/specs/2-streaming-engine-and-frontends.md` (Expert Consultation section):

> "Codex (GPT-5) was unavailable due to a usage limit lasting until 2026-05-08."

`2-specify-iter1-codex.txt` records the SKIPPED status with explanation. This
is not a substantive rebuttal to a Codex review (no review was produced); it
is a record that the second reviewer slot was infra-blocked.

## Gemini (Gemini Pro) — SKIPPED (upstream quota)

The gemini consultation invocation made 10 retries over ~4.6 minutes; every
attempt returned:

> HTTP 429 — "You have exhausted your capacity on this model."

The `consult` process exited with code 1 after 276 seconds. This matches the
project-wide pattern already documented in the spec's Expert Consultation
section:

> "Gemini Pro was rate-limited at the time of consultation."
> "gemini quota-blocked on every attempt across the project."

`2-specify-iter1-gemini.txt` records the SKIPPED status with explanation.

## Architect approval for 2/3 acceptance

Per architect instruction `[ARCHITECT INSTRUCTION | 2026-05-03T15:59:40.369Z]`:

> "Confirmed. Proceed with SKIPPED stubs for codex (quota until 2026-05-08) and
> gemini (exhausted). Spec already has 5 review passes on main: Claude Opus,
> Gemini 3 Flash, ChatGPT, Qwen, Claude Opus + Gemini Flash second pass. All
> findings addressed. Run porch done 2 to advance."

This matches the precedent established in the previous project
(`codev/projects/1-arabic-pdf-transcriber-extract/1-specify-iter1-rebuttals.md`)
where the same 2/3 acceptance rule was applied under the same architect rule.

## Summary

- **Claude `APPROVE`** — minor observations are plan-scope; no spec change.
- **Codex** — SKIPPED, quota-blocked until 2026-05-08; documented in
  consult-output file and in the spec's Expert Consultation section.
- **Gemini** — SKIPPED, quota-exhausted; documented in consult-output file
  and in the spec's Expert Consultation section.
- The spec on `main` has already been through five external review passes
  (Claude Opus ×2, Gemini 3 Flash ×2, ChatGPT user-run, Qwen user-run); all
  findings from those passes are addressed in-document. The two-consultant
  minimum is met multiple times over.
- 2/3 acceptance approved by architect for this iteration.
