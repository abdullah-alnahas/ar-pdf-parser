# Specification: Streaming Engine + Web/Electron Frontends

## Metadata
- **ID**: spec-2026-05-03-streaming-engine-and-frontends
- **Status**: draft
- **Created**: 2026-05-03

## Clarifying Questions Asked

The following questions were posed to the user before drafting. Answers shape this spec.

| # | Question | Answer |
|---|----------|--------|
| Architecture | Will this need to deploy as full SaaS (cloud GPU, multi-tenant), or local-first only? | Local-first now. SaaS may come later; spec must not preclude. |
| Side-by-side layout | Vertical strip (row per page), synced two-pane, or tabbed single-page-at-a-time? | Vertical strip — one row per page, image left, transcription right, scroll the whole document. |
| Distribution | Bundle Python + models inside Electron installer, or assume the user runs the project from source? | Source-install only. Repo README documents how to run. Audience is developers / advanced users. |
| Concurrency | Allow multiple jobs in flight, or one job at a time per server? | One job at a time. |
| Auth | Require a token even on localhost, or trust localhost? | Trust localhost. No auth. |
| Naming / flags | Many flags (CLI parity) or curated UI subset? | Few defaults, easy to use. Power users still have full CLI flags via API body. |
| Cancel | Discard partial work, keep partial silently, or surface explicit choices? | Surface three explicit choices on cancel: **Resume**, **Keep**, **Delete**. |
| Fail-early discipline | Validate everything up front, or fail late at the point of use? | Fail early on every input — paths, write permissions, GPU availability, model presence — before any compute starts. Never lose minutes of work to a problem that was knowable at boot. |
| Streaming | Show transcription only when fully done, or stream per-page as ready? | Stream per-page. Save asynchronously. UI shows results the moment they exist. |
| Final post-processing | Drop whole-document post-processing entirely, or run it after streaming and refresh the UI? | Run it after streaming and refresh: per-page draft visible immediately, final pass replaces drafts when the document finishes. |
| PDF input methods | File picker only, or also drag-drop and URL paste? | File picker + drag-drop. URL paste deferred. |
| Service ↔ CLI flag parity | Curated UI fields only, or full CLI surface available via API? | Full parity in the API. UI exposes only the curated subset. |
| Page concurrency | Configurable in UI, or fixed defaults? | Fixed sensible defaults; advanced override via env var only. |
| Dev workflow | Vite dev server with hot-reload, or only built static SPA served by FastAPI? | Vite dev for local development; FastAPI-served static build for production runs. |
| Electron Python sidecar | Bundled Python or external? | External — Electron shells out to a Python service the user has already installed. |

## Problem Statement

The project currently exposes a single CLI entrypoint that runs the full transcription pipeline end-to-end and writes one Markdown file at the end. For local interactive use this is too slow and too opaque:

- Each invocation reloads layout and OCR models from disk, paying multi-second startup cost on every run.
- The user has no visibility into per-page progress; a long PDF appears to hang for tens of seconds.
- There is no preview of partial work, no way to evaluate transcription quality without waiting for the full document, and no way to abort a bad run early.
- A failure late in the pipeline (e.g. an unwritable output path) discards all completed work.
- The CLI cannot serve as the foundation for an interactive UI: it has no event channel, no notion of a job, no per-page artifacts.

The user wants the same engine to drive three frontends — CLI (existing), a local web UI (Svelte SPA), and a desktop UI (Electron shell wrapping the same SPA) — with live progress, per-page side-by-side preview, cancel-and-resume, and a strict fail-early discipline that surfaces every knowable problem before compute begins.

## Current State

- The pipeline is invoked once per CLI run; adapters (Surya OCR, DocLayout-YOLO or full-page layout, EasyOCR) are constructed inside `main()` and discarded at process exit.
- All page rasterization, layout detection, and OCR happen serially within a single Python process driven by `argparse`.
- The pipeline produces a single concatenated Markdown file written at the end of the run; there is no per-page artifact and no event stream.
- Per-page post-processing (header/footer pruning by page-local geometry, role classification by per-page heading-quantile statistics) already runs inside the pipeline as each page completes; there is no whole-document post-processing pass today.
- Errors that are knowable up front (missing model weights, unwritable output, invalid page ranges) surface only when the relevant code path executes, sometimes minutes into a run.
- There is no second entrypoint, no service layer, and no UI of any kind.

## Desired State

A user running the project locally can:

1. **Pick a frontend on launch.** A single command starts the CLI, the local web UI, or the desktop UI. Defaults are sensible; flags are minimal.
2. **See work happen.** While the engine runs, the UI shows a progress indicator with the current page count, total page count, percentage, and a stable estimate of remaining time. Page-level events appear in real time.
3. **Read pages as they finish.** Every page that completes OCR appears immediately as a row containing the rendered original page on the left and the transcribed Markdown on the right. The user can scroll the entire document while later pages are still processing.
4. **Trust what they see.** Each page's transcription is final at the moment it appears: the pipeline's per-page post-processing has already run before the page is emitted. There is no separate "draft → finalized" two-stage refresh. If a future cross-page polish step is added (e.g. document-wide heading-level normalization), it is out of scope for this work and belongs in a follow-up spec.
5. **Cancel safely.** A visible cancel control stops new work immediately and presents three choices: **Resume** (continue from the last completed page within the same running service process), **Keep** (leave the partial document on disk for review), or **Delete** (wipe the partial job). Whichever choice is made, work that was already on disk remains consistent. Cross-process resume — re-attaching to a partially completed job after the service has been restarted — is explicitly out of scope for this work and is listed in Open Questions for a follow-up.
6. **Fail before waiting.** Any error that can be detected before compute starts — unreadable PDF, unwritable job directory, missing GPU when one is required, missing model weights with no network, invalid page selection, unsupported backend combination — must surface within the first second of a run, not later.
7. **Reuse loaded models.** Repeated jobs against the same backend selection do not pay model-load cost a second time. Switching backends pays the cost of the new backend only.
8. **Get a faster transcription.** With unchanged hardware and the same backend selection, end-to-end runtime on a representative document is at least 30 percent shorter than today's baseline, with a stretch target of 50 percent or better. The exact speedup levers — candidates include page-render / OCR overlap via a bounded async queue, batched OCR inference within a single Surya call, and reduced-precision (fp16 / bf16) inference on capable hardware — are scoped and ordered in the implementation plan; not every lever needs to ship in this spec's work, but the success criterion above must be met before the work is considered complete.

The Markdown produced for any given input remains semantically equivalent to today's CLI output; the speedups and the streaming UI must not change the final transcription.

## Stakeholders
- **Primary Users**: The repository owner and other developers / advanced users running the project locally to transcribe Arabic PDFs interactively.
- **Secondary Users**: Future SaaS users (out of scope now, but the architecture must not block this path).
- **Technical Team**: The same maintainer plus AI agents collaborating through codev protocols.
- **Business Owners**: The repository owner.

## Success Criteria

- [ ] A single project install exposes three entrypoints — CLI, local web service, Electron shell — that share the same Python engine code path.
- [ ] On a representative multi-page Arabic PDF, the web UI shows the first transcribed page within five seconds of submission on a warm process (models already loaded), and additional pages stream in as they finish.
- [ ] The Markdown produced by the new streaming pipeline is byte-identical or semantically equivalent to today's CLI output for the same backends and inputs (verified on a fixture).
- [ ] End-to-end runtime on a representative document, with the same backend selection and hardware, is at least 30 percent shorter than today's CLI baseline (measured on the same fixture, warm process, GPU available).
- [ ] Models are loaded once per process lifetime; a second job against the same backends starts processing pages without paying model-load cost.
- [ ] Every input failure mode (missing PDF, unwritable target, invalid page range, missing GPU when required, missing model weights, incompatible backend pair) is detected and reported with an actionable error before any page is rasterized or OCR runs.
- [ ] The web UI shows a progress indicator (current/total pages, percentage, elapsed, smoothed ETA) that updates as page events arrive.
- [ ] The web UI shows side-by-side per-page rows: rendered page image on the left, rendered Markdown on the right, scrolling the whole document. Page images are served as PNG files via a per-job HTTP endpoint that reads them from the job directory; image bytes are not embedded in event payloads.
- [ ] A cancel control stops the run immediately and surfaces Resume / Keep / Delete; each option leaves the job directory in a documented, consistent state.
- [ ] After cancellation, a Resume action completes the document without reprocessing pages that were already finished.
- [ ] Job artifacts are stored under a single per-job directory whose layout is documented; no per-job state lives anywhere else.
- [ ] The web UI works in both Vite dev mode (developer hot reload) and as a static build served by the Python service in production.
- [ ] The Electron shell launches the Python service as an external sidecar, loads the same SPA build, and shuts the sidecar down cleanly on quit. It does not bundle Python.
- [ ] Documentation in the repository explains, in one place, how to run each of CLI, web, and Electron, including all prerequisites.
- [ ] All tests pass; new functionality has unit and integration coverage; no reduction in overall coverage.

## Constraints

### Technical Constraints
- The engine must remain a single Python package; the service and CLI must share one code path for layout, OCR, post-processing, and event emission. The CLI continues to import and run the engine in-process — it does **not** become a network client of the service. The service is an additional, optional entrypoint that wraps the same engine objects with a long-running HTTP layer.
- The service runs only on localhost. No remote access, no auth layer, no TLS termination is part of this work.
- Only one job runs at a time per service process. A second submission while a job is in flight is rejected with a clear error.
- Per-job artifacts (rasterized page PNGs, per-page Markdown, events log, final result) are retained on disk by default until the user explicitly cleans them via a documented command. They are never deleted as a side effect of finishing, cancelling, or restarting a job.
- The web UI is a Svelte single-page application using SvelteKit's static adapter so that the same build runs both inside the Electron BrowserWindow and served from the Python service. The static adapter implies no server-side rendering and no SvelteKit server routes; all dynamic behavior is client-side JavaScript talking to the Python service over HTTP and Server-Sent Events.
- The Electron shell is a thin Node.js wrapper that spawns the Python service as a child process; it does not embed a Python runtime. The shell is launched from the repository via a documented Node command (e.g. `npm run electron`) and assumes the Python project is already installed in the user's active environment. The exact command is fixed in the README during this work.
- Local model caches and HF Hub credentials remain unchanged. Adding a new frontend must not change how models are downloaded or where they live.
- The streaming pipeline must not change the final Markdown output for a given input and backend selection beyond what is already produced today.
- The new entrypoints must coexist with the existing CLI without breaking any currently passing test.

### Business Constraints
- No deployment, no hosting, no domain, no auth provider.
- No paid services. All dependencies must be open source or already in use by the project.
- The audience is developers; documentation is in the repo, not on a marketing site.

## Assumptions
- The current Surya + DocLayout-YOLO + EasyOCR stack remains the engine of record for this work; this spec does not introduce new ML models.
- The user's machine has a working Python environment with the project installed in editable mode for development; the same environment runs the service in production.
- Node.js is available on the user's machine for both the Svelte build toolchain and the Electron shell.
- A modern browser (current Chromium, Firefox, or Safari) is available for the web frontend.
- Disk space for per-job artifacts (rasterized page PNGs and per-page Markdown chunks) is acceptable on the user's machine; no cloud storage is involved.
- The pipeline today already exposes enough internal structure (per-page raster, per-page region list, per-page Markdown) that streaming and per-page artifacts are achievable without rewriting the OCR core.

## Solution Approaches

### Approach 1: Long-running Python service with shared SPA frontend (RECOMMENDED)

**Description**: Refactor the engine into a long-lived Python process that loads adapters once, exposes a small REST + Server-Sent-Events surface, writes per-job artifacts to a documented directory layout, and emits page-level events. A single Svelte SPA, built once, is served both by the Python service (web mode) and embedded in an Electron BrowserWindow (desktop mode). The CLI continues to exist as a thin client of the same engine.

**Pros**:
- One engine, three entrypoints, no duplication.
- Models load once per process; repeated jobs are fast.
- Native fit for streaming events and side-by-side preview.
- SaaS migration later is additive: add auth, queueing, and storage layers in front of the same service.
- Electron stays thin; the heavy work is in Python where it belongs.

**Cons**:
- Adds a service layer the project does not have today.
- Requires per-job state on disk and a clear contract for cancel/resume.
- Adds a Node.js toolchain and a SvelteKit codebase to the repository.

**Estimated Complexity**: Medium-High
**Risk Level**: Medium

### Approach 2: Two separate UI codepaths, no shared service

**Description**: Build the Electron app as a first-class desktop app that owns the pipeline directly via a Python child process spoken to over stdio, and build the web UI as a separate Flask or FastAPI app with its own templating. No SPA share.

**Pros**:
- Fewer moving parts inside Electron (no localhost HTTP).
- Web UI can be plain server-rendered HTML.

**Cons**:
- Duplicated UI work for two frontends.
- Diverging UX between desktop and web.
- Per-job state has to be implemented twice or factored into a shared library anyway.
- The "share one engine" intent is lost.

**Estimated Complexity**: High
**Risk Level**: Medium-High

### Approach 3: Streaming-only CLI, no UI at all

**Description**: Add per-page Markdown emission and a progress bar to the CLI. Skip the web and desktop frontends entirely. Side-by-side preview is left to the user's editor.

**Pros**:
- Smallest possible change.
- No new technologies.

**Cons**:
- Does not satisfy the user's stated requirement for web and Electron frontends with side-by-side preview.

**Estimated Complexity**: Low
**Risk Level**: Low

### Recommendation

Approach 1. It is the only approach that delivers the three-frontend goal with one engine and leaves room for SaaS later without a rewrite.

## Open Questions

### Critical (Blocks Progress)
- [ ] None remaining; clarifying questions covered all blocking decisions.

### Important (Affects Design)
- [ ] How is the SPA build artifact distributed alongside the Python package — checked in, built on install, or built on first launch? The plan must pick one and document it.
- [ ] What happens if the user resizes the browser to a narrow width? The vertical-strip layout works at desktop widths; the spec does not mandate a mobile-friendly layout, but the plan should document the minimum viewport.
- [ ] Cancel-and-Resume is scoped to in-process resume only in this work. The plan must specify the in-memory state machine the service uses to track completed pages and the pending-page queue, and how the Resume button re-arms it.
- [ ] Cross-process resume — surviving a service restart — is deferred. A follow-up spec should decide what subset of job state must be persisted on disk (job manifest, completed-page list, backend selection, input-PDF digest) and how the service detects a stale or moved input PDF on re-attach.

### Nice-to-Know (Optimization)
- [ ] Should the service expose a `/cancel` endpoint, or is cancellation triggered only via the UI? (Either is fine; the plan picks.)
- [ ] Should ETA smoothing be moving-average, median over a window, or something else? (The plan picks.)
- [ ] When the user runs the documented `clean` command, what is the cutoff (delete only fully completed jobs older than N days, or wipe everything)? Per-job retention is mandated by the Technical Constraints; this question is purely about the cleanup ergonomics.

## Performance Requirements

- **Time-to-first-page (web mode, default backends, warm process)**: the user sees the first finalized page row within a small number of seconds of submitting a multi-page PDF.
- **End-to-end runtime improvement**: on a representative document with the same hardware and backend selection, the new pipeline must be materially faster than the current CLI baseline. The plan defines the exact target ratio.
- **Model load cost amortization**: a second job submitted against the same backends as the previous job pays no model-load cost.
- **Memory footprint of a warm service**: bounded by the loaded model sizes plus a small per-job working set; no unbounded growth across repeated jobs.
- **UI responsiveness**: SSE event handling and page-row append must keep the UI thread responsive on a document of at least one hundred pages.

## Security Considerations

- The service binds to localhost only and is not authenticated. The threat model assumes a single trusted user on a single machine.
- The service must not accept absolute or path-traversing job identifiers; job IDs are server-generated and validated.
- File uploads must reject non-PDF inputs early and must not write outside the per-job directory.
- The Electron shell must not load remote URLs in the BrowserWindow; only the local SPA build is loaded.
- The Electron shell must shut down its Python child cleanly on quit so a stale service does not linger.

## Test Scenarios

### Functional Tests
1. CLI run on a multi-page Arabic PDF with default backends produces the same Markdown the current CLI produces today on the same input.
2. Web run on the same PDF produces the same Markdown after the document finishes.
3. Submitting a job against an unwritable job directory fails before any page is rasterized.
4. Submitting a job against a missing PDF fails immediately with an actionable error.
5. Submitting a job that requires the GPU on a machine without a GPU fails immediately if the chosen backend cannot fall back; otherwise, succeeds on CPU and reports the fallback.
6. Submitting a second job with the same backends as the previous job does not reload models.
7. Submitting a second job with different backends reloads only the changed adapter.
8. Cancelling a job mid-run leaves the job directory in a state where Resume completes the document and Delete removes it cleanly.
9. The Electron shell launches, processes a small document end-to-end, and shuts down without leaving a Python process behind.

### Non-Functional Tests
1. End-to-end runtime on a representative fixture is below the plan's target threshold.
2. Time-to-first-page on the same fixture is below the plan's target threshold.
3. The UI remains interactive during processing (no main-thread stalls visible in the browser).

## Dependencies

- **External Services**: None.
- **Internal Systems**: Existing Surya, DocLayout-YOLO, EasyOCR adapters; existing pipeline post-processing; existing Region / BBox / RegionRole types.
- **Libraries/Frameworks**: A Python ASGI server (e.g. uvicorn) and a Python web framework already in the ecosystem (e.g. FastAPI). SvelteKit with the static adapter on the frontend. Electron for the desktop shell. No new ML dependencies.

## References

- The model survey notebook in `notebooks/00_model_survey.py` documents why Surya and DocLayout-YOLO are the chosen backends.
- The original spec `codev/specs/1-arabic-pdf-transcriber-extract.md` documents the engine that this work extends.
- The existing CLI is the reference implementation for output equivalence.

## Risks and Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|---------------------|
| Streaming output drifts from the existing CLI's output for the same input | Medium | High | Add a fixture test that compares the concatenated streaming output to today's CLI output for the same backends; gate every phase commit on it staying green. |
| Engine refactor introduces regressions in existing CLI runs | Medium | High | Keep the CLI test suite as the regression gate; refactor in small, committed phases per the plan; run end-to-end fixtures after every phase. |
| In-process Resume races a finishing job (user clicks Resume after the last page already streamed) | Low | Low | Resume is a no-op when there is no remaining work; the UI hides the action once the job is complete. |
| Bundling SPA build alongside Python package becomes painful (build-on-install vs check-in trade-off) | Medium | Low | Decide early in the plan; document the choice in README. |
| Electron child-process management leaves zombie Python processes | Low | Medium | Explicit child lifecycle in the Electron main process; integration test that asserts no orphaned process on shutdown. |
| Single concurrent job assumption breaks future SaaS migration | Low | Low | Document the assumption explicitly; SaaS migration can layer a queue in front of the service without changing the engine. |
| ETA estimates are wildly inaccurate on documents with mixed page complexity (e.g. a heavy table page among text pages) | Medium | Low | Use a smoothed estimator over a small window and show the source ("based on last N pages") so users can interpret variance. |

## Expert Consultation

**Date**: 2026-05-03
**Models Consulted**: Claude Opus (independent reviewer process). Codex (GPT-5) was unavailable due to a usage limit lasting until 2026-05-08; Gemini Pro was rate-limited at the time of consultation. The user was notified and the spec was updated based on the available reviewer.

**Sections Updated**:

- **Current State**: Corrected a factual error. The earlier draft claimed whole-document post-processing runs after every page finishes OCR. Verification against `pipeline.py` and `roles/classify.py` showed post-processing already runs per page inside `_process_page`, with header/footer pruning using page-local geometry and heading classification using single-page heading-quantile statistics. Updated to describe the actual current behavior.
- **Desired State (item 4)**: Removed the "draft → finalized" two-stage refresh model. Each page's transcription is final the moment it appears, because the existing per-page post-processing has already run. A future cross-page polish step is explicitly out of scope and noted as a follow-up.
- **Desired State (item 5)**: Resume is scoped to in-process only (within a single running service process). Cross-process resume after a service restart is explicitly deferred to a follow-up spec.
- **Desired State (item 8) and Performance Requirements / Success Criteria**: Replaced "materially shorter" with a quantitative floor of at least 30 percent reduction in end-to-end runtime on a representative document with a warm process, plus a stretch target of 50 percent or better. Time-to-first-page success criterion is now bounded at five seconds on a warm process. Speedup levers (overlap, batching, reduced precision) are explicitly framed as plan-phase scope decisions, not commitments at the spec level.
- **Success Criteria**: Page image delivery mechanism made explicit — PNG files served via a per-job HTTP endpoint reading from the job directory, not embedded in event payloads.
- **Technical Constraints**: Made explicit that the SvelteKit static adapter implies no server-side rendering and no SvelteKit server routes, so all dynamic behavior is client-side JavaScript over HTTP + SSE.
- **Open Questions**: Removed the stale question about how the UI distinguishes draft rows from finalized rows (the model that required this distinction has been removed). Resume scoping is now stated as a design constraint rather than an open question, and cross-process resume is filed as the deferred follow-up.

**Second pass — Gemini 3 Flash Preview, 2026-05-03**:
- Flagged a contradiction in the **Risks and Mitigation** table: a row about marking "draft rows visibly" had survived from the original draft and contradicted the corrected Desired State. The risk row was rewritten to be about output equivalence instead of draft markers.
- Asked for the **CLI / service relationship** to be made explicit. Technical Constraints now state that the CLI imports and runs the engine in-process and is **not** a network client of the service.
- Asked for **per-job artifact retention** to be a spec-level guarantee rather than a plan choice, because Resume depends on it. Technical Constraints now mandate retention until explicit cleanup; the cleanup-ergonomics question stays in Open Questions.
- Asked how the **Electron entrypoint** is exposed in a source-install environment. Technical Constraints now state the shell launches via a documented Node command and assumes the Python environment is already installed.

**Note on consultation coverage**: SPIR's default policy is two consultants. Two reviewers were available across the two passes (Claude Opus and Gemini 3 Flash Preview); Codex (GPT-5.4) was unavailable at this checkpoint due to a usage limit lasting until 2026-05-08, so the second consultant slot was filled by Gemini once its rate limit recovered. The user is aware of and accepted this substitution.

## Approval
- [ ] Technical Lead Review
- [ ] Product Owner Review
- [ ] Stakeholder Sign-off
- [ ] Expert AI Consultation Complete

## Notes

- This spec deliberately keeps Surya, DocLayout-YOLO, and EasyOCR as the only OCR/layout choices. New models are out of scope; they belong in a follow-up survey and a separate spec.
- This spec deliberately keeps "SaaS" out of scope but does not preclude it. The plan must document any architectural decisions that would block a future SaaS layer.
- The fail-early discipline applies project-wide, not only to the new entrypoints; the plan may include a small refactor of the existing CLI to apply the same checks.
