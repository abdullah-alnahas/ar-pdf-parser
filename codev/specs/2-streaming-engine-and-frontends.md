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
5. **Cancel safely.** A visible cancel control stops new work and presents three choices: **Resume** (continue from the last completed page within the same running service process), **Keep** (leave the partial document on disk for review), or **Delete** (wipe the partial job). Cancel does not interrupt a page that is already inside the OCR call: the in-flight page is allowed to finish and is committed to disk before the engine stops; everything after that page is skipped. Whichever option the user picks, the on-disk job directory ends in a consistent state — defined in this spec as: every page index that appears in the events log as `page_done` has both its rendered PNG and its per-page Markdown on disk; no page index has a half-written artifact; the job manifest accurately reflects the set of completed pages. Cross-process resume — re-attaching to a partially completed job after the service has been restarted — is explicitly out of scope for this work.
6. **Fail before waiting.** Errors are detected in two tiers, both before any page is rasterized. **Tier 1 (input validation, ≤ 1 second after submit)**: unreadable PDF path, unwritable job directory, invalid page selection, unsupported backend combination, malformed request body. **Tier 2 (backend warm-up, completes before the first page event)**: missing GPU when the chosen backend requires one, missing model weights with no network, lazy-loaded model import failure. Tier 1 always runs synchronously inside the request handler so the client sees the error in the API response. Tier 2 may take several seconds if a model has to load; it runs before the engine starts page work and reports its outcome via the same events stream as page progress, so the UI can show "loading models…" honestly rather than appearing to hang.
7. **Reuse loaded models (service mode only).** Inside the long-running service process, repeated jobs against the same backend selection do not pay model-load cost a second time, and switching backends pays the cost of the new backend only. The CLI does not benefit from this — every CLI invocation is a fresh Python process and reloads its adapters; that is unchanged from today and out of scope here.
8. **Get a faster transcription.** With unchanged hardware and the same backend selection, end-to-end runtime on a representative document is at least 30 percent shorter than today's baseline, with a stretch target of 50 percent or better. To make this target reachable without a gamble, this spec commits to **at least one structural speedup lever shipping in this work**: page-render and OCR overlap via a bounded async queue, so that page N + 1 is being rasterized while page N is in OCR. Additional candidate levers — batched OCR inference within a single Surya call, reduced-precision (fp16 / bf16) inference on capable hardware — are scoped and ordered in the implementation plan; the plan may add or defer them, but the overlap lever is non-optional.
9. **Export to Word (.docx) with page boundaries preserved.** Once a job reaches `completed`, the user can download a Word document of the transcription. The docx mirrors the source PDF page-by-page: every PDF page becomes its own Word page (a hard page break is inserted between every pair of consecutive transcribed pages), and the text of one PDF page never appears on a different page of the docx. Heading levels, paragraphs, and list items present in the per-page Markdown are mapped to corresponding Word styles; figures and tables transcribed today carry over with their existing structure (figures as image placeholders, tables as Word tables). Right-to-left paragraph direction is set on every paragraph to render Arabic correctly. The docx is built from the same per-page Markdown the Markdown export uses, with no re-OCR. Markdown export remains the default; docx is offered alongside it as an additional download in both web and CLI surfaces.

The Markdown produced for any given input remains semantically equivalent to today's CLI output; the speedups and the streaming UI must not change the final transcription.

## Stakeholders
- **Primary Users**: The repository owner and other developers / advanced users running the project locally to transcribe Arabic PDFs interactively.
- **Secondary Users**: Future SaaS users (out of scope now, but the architecture must not block this path).
- **Technical Team**: The same maintainer plus AI agents collaborating through codev protocols.
- **Business Owners**: The repository owner.

## Success Criteria

- [ ] A single project install exposes three entrypoints — CLI, local web service, Electron shell — that share the same Python engine code path.
- [ ] On a representative multi-page Arabic PDF, the web UI shows the first transcribed page within five seconds of submission on a warm process (models already loaded), and additional pages stream in as they finish.
- [ ] The Markdown produced by the new streaming pipeline is **equivalent** to today's CLI output for the same backends and inputs, where equivalent is defined as: after applying the canonical normalization (NFC Unicode normalization, trailing whitespace stripped per line, runs of two or more blank lines collapsed to one, terminal newline enforced), the streaming and CLI outputs are byte-identical. The test runs on a checked-in fixture document and is gated in CI.
- [ ] **End-to-end runtime benchmark.** On a checked-in **reference fixture** — defined as a multi-page Arabic PDF with a documented, fixed page count and layout mix (body-text, headings, footnoted passages) committed alongside the test, executed on a documented hardware baseline (current local development machine class) with default backends, warm process, GPU available — the new pipeline is at least 30 percent shorter wall-clock than today's CLI on the same input. The fixture's exact identity, page count, and the measurement harness are pinned by the implementation plan and committed to the repository before the benchmark gate is declared green.
- [ ] **Service-mode** model reuse: inside the long-running service process, a second job against the same backends as the previous job starts processing pages without paying model-load cost.
- [ ] Every input failure mode (missing PDF, unwritable target, invalid page range, missing GPU when required, missing model weights, incompatible backend pair) is detected and reported with an actionable error before any page is rasterized or OCR runs.
- [ ] The web UI shows a progress indicator (current/total pages, percentage, elapsed, smoothed ETA) that updates as page events arrive.
- [ ] The web UI shows side-by-side per-page rows: rendered page image on the left, rendered Markdown on the right, scrolling the whole document. Page images are served as PNG files via a per-job HTTP endpoint that reads them from the job directory; image bytes are not embedded in event payloads.
- [ ] A cancel control stops the run and surfaces Resume / Keep / Delete; each option leaves the job directory in the consistent state defined in Desired State item 5 (every `page_done` page has both PNG and Markdown on disk; no page has a half-written artifact; the job manifest reflects the completed set).
- [ ] After cancellation, a Resume action completes the document without reprocessing pages that were already finished.
- [ ] Job artifacts are stored under a single per-job directory whose layout is documented; no per-job state lives anywhere else.
- [ ] The web UI works in both Vite dev mode (developer hot reload) and as a static build served by the Python service in production.
- [ ] The Electron shell launches the Python service as an external sidecar, loads the same SPA build, and shuts the sidecar down cleanly on quit. It does not bundle Python.
- [ ] Documentation in the repository explains, in one place, how to run each of CLI, web, and Electron, including all prerequisites.
- [ ] On a completed job, a docx export is available — via a download button in the web UI and via a documented CLI flag — that contains every transcribed page of the source PDF with hard page breaks between consecutive pages, right-to-left paragraph direction set on every paragraph, and no leakage of text between pages.
- [ ] All tests pass; new functionality has unit and integration coverage; no reduction in overall coverage.

## Constraints

### Technical Constraints
- The engine must remain a single Python package; the service and CLI must share one code path for layout, OCR, post-processing, and event emission. The CLI continues to import and run the engine in-process — it does **not** become a network client of the service. The service is an additional, optional entrypoint that wraps the same engine objects with a long-running HTTP layer.
- The service runs only on localhost. No remote access, no auth layer, no TLS termination is part of this work.
- Only one job runs at a time per service process. A second submission while a job is in flight is **rejected** by the API with a 409 Conflict response and a body explaining which job is still active; the request is not queued. Users who want a queue can run the CLI sequentially.
- Per-job artifacts (rasterized page PNGs, per-page Markdown, events log, final result) are retained on disk by default until the user explicitly cleans them via a documented command. They are never deleted as a side effect of finishing, cancelling, or restarting a job.
- All model inference (layout detection, OCR) runs **off the ASGI event loop** — via `asyncio.to_thread`, a `ThreadPoolExecutor`, or an equivalent off-loop primitive — so SSE event delivery, cancellation signalling, and incoming HTTP requests stay responsive while a page is in OCR. Implementations that block uvicorn's main loop on a multi-second OCR call are not acceptable.
- The web UI is a Svelte single-page application using SvelteKit's static adapter so that the same build runs both inside the Electron BrowserWindow and served from the Python service. The static adapter implies no server-side rendering and no SvelteKit server routes; all dynamic behavior is client-side JavaScript talking to the Python service over HTTP and Server-Sent Events.
- The web UI must remain interactive on documents of at least one hundred pages. To meet this on the lower end of the supported hardware, the page-row list **must use virtualized rendering** (only rows in or near the viewport mounted in the DOM); naïve append-every-row implementations are not acceptable. The choice of virtualization library is for the plan to make.
- The Electron shell is a thin Node.js wrapper that spawns the Python service as a child process; it does not embed a Python runtime. The shell is launched from the repository via a documented Node command (e.g. `npm run electron`) and assumes the Python project is already installed in the user's active environment. The exact command is fixed in the README during this work.
- The Electron shell launches the Python service on a **fresh, unused localhost port chosen by the operating system** (port 0 binding pattern); the chosen port is read from the child's standard streams and passed to the BrowserWindow via a known URL parameter. Hard-coded ports are not acceptable. If the child fails to start or fails to print its port within a documented timeout, the shell surfaces an explicit error to the user — it does not silently load an empty window.
- The Electron shell only loads the BrowserWindow contents **after** the Python service has answered an HTTP readiness probe; the shell's main process implements the readiness handshake.
- Local model caches and HF Hub credentials remain unchanged. Adding a new frontend must not change how models are downloaded or where they live.
- The streaming pipeline must not change the final Markdown output for a given input and backend selection beyond what is already produced today, where "must not change" is verified by the canonical-normalization equivalence test in Success Criteria.
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

## Job Directory Layout

Every job has exactly one directory on disk; nothing about an in-flight or completed job lives outside it. The layout below is fixed by this spec; the implementation plan may add files inside the directory but may not move the existing ones.

```
<jobs_root>/<job_id>/
├── manifest.json        # job-level metadata (immutable except status fields)
├── input.pdf            # copy or hardlink of the submitted PDF
├── events.jsonl         # append-only event log, one JSON event per line
├── pages/
│   ├── 0001.png         # rasterized page image, zero-padded to 4 digits
│   ├── 0001.md          # finalized per-page Markdown
│   ├── 0002.png
│   ├── 0002.md
│   └── …
├── result.md            # written when the job reaches the done state
└── result.docx          # built lazily on first docx download; cached for subsequent requests
```

Constraints on the layout:

- `<jobs_root>` defaults to `~/.cache/arabic-pdf-transcribe/jobs/`. The path is configurable via a single environment variable; the spec does not name the variable, the plan picks it.
- `<job_id>` is a server-generated opaque token (UUIDv4 hex, no slashes); inputs from the client never set it.
- Page indices are **1-based** and zero-padded to four digits in filenames; this matches the `--pages` CLI syntax users already see.
- A page is considered complete on disk only when both `pages/NNNN.png` and `pages/NNNN.md` exist and `events.jsonl` carries a matching `page_done` event. Writes happen in this order, with `events.jsonl` appended last; readers can use the events log as the source of truth.
- `result.md` is the concatenation of `pages/NNNN.md` files in numeric order, written in one shot at job completion. While the job is running it does not exist; clients that want partial output read the per-page files.
- `manifest.json` carries: job_id, created_at, input_pdf_filename, page_selection, layout_backend, ocr_backend, status (`running` / `completed` / `cancelled` / `failed`), completed_pages (sorted list of integers), total_pages.

## Job Lifecycle and State Machine

A job moves through these states inside the service process. State transitions are durable in `events.jsonl` and reflected in `manifest.json.status`.

```
        (POST /jobs)
           │
           ▼
        accepted ─── tier-1 input check fails ──▶ failed (terminal)
           │
        warming ─── tier-2 backend check fails ──▶ failed (terminal)
           │
        running ─────── all pages emitted ──▶ finalizing ──▶ completed (terminal)
           │
       (cancel)
           │
           ▼
        stopping ─── in-flight page committed ──▶ cancelled (terminal)
                                                     │
                                                  (resume)
                                                     │
                                                     ▼
                                                  running
```

State definitions:

- **accepted** — the request is parsed, tier-1 input checks pass, the job directory is created, the job_id is returned to the client. No engine work yet.
- **warming** — the engine resolves backends and ensures the model adapters are loaded (warm hit, or load if cold). Tier-2 errors surface here.
- **running** — pages are being processed. Each page that finishes appends a `page_started` and a `page_done` event and writes its PNG and Markdown.
- **stopping** — the user cancelled. The currently in-flight page is allowed to commit; no further pages are scheduled. The state transitions to **cancelled** as soon as the in-flight page is on disk.
- **cancelled** — terminal. The Resume action transitions back to **running** with the pending-page set recomputed from `manifest.completed_pages` and the original `page_selection`.
- **finalizing** — every page in `page_selection` has a `page_done`; the engine concatenates `pages/*.md` into `result.md` and emits the `pipeline_done` event.
- **completed** — terminal. The Resume button is hidden in the UI. `manifest.json.status = "completed"`.
- **failed** — terminal. `manifest.json.status = "failed"` with a `reason` field. No Resume.

Resume invariants (in-process only):

- Resume requires the service process is the same one that started the job (in-memory job state still exists).
- Resume re-enters at the first page of `page_selection` not in `manifest.completed_pages`, in order, until done.
- Resume does not re-rasterize or re-OCR completed pages.
- Resume is hidden from the UI in any state other than **cancelled**.

## API and Event Contract (Skeletal)

The plan is free to add fields, query params, and side endpoints, but the routes and event types listed below are part of this spec and may not be removed or renamed.

### HTTP routes

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/jobs` | Submit a new job. Multipart body: the PDF file plus a JSON metadata field carrying the full CLI flag namespace (`layout_backend`, `ocr_backend`, `pages`, `strict`, etc.). Returns `201 Created` with `{job_id, status}`. Returns `409 Conflict` if another job is already active. Tier-1 input failures return `400` with `{error_code, message}`. |
| `GET` | `/jobs/{job_id}` | Job status snapshot — same fields as `manifest.json`. |
| `GET` | `/jobs/{job_id}/events` | Server-Sent Events stream of all events for this job, ordered by emission. Reconnecting clients pass `Last-Event-ID` to resume mid-stream from the events log. |
| `GET` | `/jobs/{job_id}/pages/{n}/image` | Rendered PNG for page `n` (1-based). 404 until the page's `page_done` event has been emitted. |
| `GET` | `/jobs/{job_id}/pages/{n}/markdown` | Per-page Markdown for page `n`. Same 404 rule. |
| `GET` | `/jobs/{job_id}/result.md` | Final concatenated Markdown. 404 until `pipeline_done`. |
| `GET` | `/jobs/{job_id}/result.docx` | Final Word document with hard page breaks between source-PDF pages and RTL paragraph direction on every paragraph. 404 until `pipeline_done`. Built lazily on first request and cached in the job directory thereafter. |
| `POST` | `/jobs/{job_id}/cancel` | Request cancellation. The body carries `{action: "stop" | "resume" | "delete"}`. The default action on first invocation is `"stop"`; subsequent invocations carry `"resume"` or `"delete"`. |
| `DELETE` | `/jobs/{job_id}` | Equivalent to `POST .../cancel` with `action: "delete"` plus removal of the job directory. Idempotent. |
| `GET` | `/healthz` | Service health and engine readiness; used by the Electron shell's readiness handshake. |

### SSE event types

Every SSE message has an `id` (monotonic per job, persisted in `events.jsonl`) and a `event:` line naming one of the types below. The `data:` line is JSON.

| Event type | Required fields | When emitted |
|-----------|-----------------|--------------|
| `job_accepted` | `job_id`, `total_pages`, `page_selection` | First event after `POST /jobs`. |
| `warming` | `backend_layout`, `backend_ocr`, `phase ∈ {"loading", "ready"}` | At entry to and exit from the **warming** state. The `loading` event is what the UI shows as "loading models…". |
| `page_started` | `page_index`, `total_pages` | Just before page `n` enters layout detection. |
| `page_done` | `page_index`, `total_pages`, `image_url`, `markdown_url`, `duration_seconds` | After PNG and Markdown for page `n` are on disk. `image_url` and `markdown_url` are server-relative paths the SPA can fetch directly (`pages/0007/image`, `pages/0007/markdown`). |
| `page_failed` | `page_index`, `total_pages`, `error_code`, `message` | When a single page fails (only emitted if the engine continues; with `--strict` the engine emits `pipeline_failed` instead). |
| `state_change` | `from`, `to` | On every state-machine transition listed in the Job Lifecycle section. |
| `pipeline_done` | `total_pages`, `result_url` | Job reached **completed**. `result_url` is the server-relative path to `result.md`. |
| `pipeline_failed` | `error_code`, `message`, `phase` | Terminal failure. The `phase` distinguishes tier-1, tier-2, and per-page failures from a hard pipeline abort. |

The SPA constructs page asset URLs only by following the `image_url` / `markdown_url` strings emitted by `page_done` events; it does not synthesize them from the `page_index`. This keeps the URL scheme an implementation detail of the service.

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
- [ ] Cross-process resume — surviving a service restart — is deferred. A follow-up spec should decide what subset of job state must be persisted on disk (job manifest, completed-page list, backend selection, input-PDF digest) and how the service detects a stale or moved input PDF on re-attach.
- [ ] Image storage format — the spec says PNG; if disk usage on long documents becomes a problem the plan may switch to a smaller format (lossy WebP, JPEG, or a thumbnail variant alongside the full-resolution PNG). This is left to the plan because it does not change the API contract.
- [ ] Docx generation library: `python-docx` is the obvious default, but the plan may pick another if there is a concrete reason; the choice does not change the API or the success criterion.

### Nice-to-Know (Optimization)
- [ ] Should the service expose a `/cancel` endpoint, or is cancellation triggered only via the UI? (Either is fine; the plan picks.)
- [ ] Should ETA smoothing be moving-average, median over a window, or something else? (The plan picks.)
- [ ] When the user runs the documented `clean` command, what is the cutoff (delete only fully completed jobs older than N days, or wipe everything)? Per-job retention is mandated by the Technical Constraints; this question is purely about the cleanup ergonomics.

## Performance Requirements

- **Time-to-first-page (web mode, default backends, warm process)**: ≤ 5 seconds from submit to the first `page_done` event for a multi-page PDF, on the documented hardware baseline.
- **End-to-end runtime improvement**: ≥ 30 percent reduction versus the current CLI baseline on the reference fixture, with a stretch target of 50 percent. Measurement methodology is pinned by the implementation plan; the fixture and harness are committed to the repository.
- **Model load cost amortization (service mode)**: a second job submitted against the same backends as the previous job pays no model-load cost.
- **Memory footprint of a warm service**: bounded by the loaded model sizes plus a small per-job working set; no unbounded growth across repeated jobs.
- **UI responsiveness on long documents**: with one hundred pages of `page_done` events delivered in succession, the SPA's main-thread frame budget stays under the browser's interactive threshold (the planning phase pins the exact metric and instrument); the virtualization mandate in Technical Constraints is the structural guarantee behind this.

## Security Considerations

- The service binds to localhost only and is not authenticated. The threat model assumes a single trusted user on a single machine.
- The service must not accept absolute or path-traversing job identifiers; job IDs are server-generated and validated.
- File uploads must reject non-PDF inputs early and must not write outside the per-job directory.
- The Electron shell must not load remote URLs in the BrowserWindow; only the local SPA build is loaded.
- The Electron shell must shut down its Python child cleanly on quit so a stale service does not linger.

## Test Scenarios

### Functional Tests
1. CLI run on the reference fixture with default backends produces Markdown that, after the canonical normalization (NFC, trailing whitespace stripped, multi-blank-lines collapsed, terminal newline), is byte-identical to the committed expected-output file.
2. Web (service-mode) run on the reference fixture produces the same normalized Markdown after `pipeline_done`.
3. **Docx export equivalence and page boundaries.** A docx export of a multi-page run contains every page's transcribed text and only that page's text; no text from one source PDF page appears in the docx page corresponding to a different source page. The test inspects the generated docx structure, not just its plaintext.
4. Submitting a job against an unwritable job directory returns a tier-1 error in the API response before any page is rasterized.
5. Submitting a job against a missing PDF returns a tier-1 error with an actionable message.
6. Submitting a job whose backend requires a GPU on a machine without one emits a tier-2 `pipeline_failed` event before the first `page_started`, with an actionable message.
7. The same submission on a machine with a working GPU progresses past `warming` to the first `page_done` within the time-to-first-page budget.
8. Submitting a second job with the same backends as the previous job emits a `warming` event with `phase: "ready"` and no preceding load, demonstrating model reuse.
9. Submitting a second job with different backends emits a `warming` event with `phase: "loading"` for the new adapter only.
10. Cancelling a job mid-run with `action: "stop"` allows the in-flight page to commit, transitions to `cancelled`, and leaves the job directory in the consistent state defined in Desired State item 5.
11. After cancellation, `action: "resume"` re-enters `running` and completes the document without rasterizing or OCR-ing any page already in `manifest.completed_pages`.
12. The Electron shell launches the Python service on a fresh OS-chosen port, waits for the readiness probe, loads the SPA, processes a small document end-to-end, and shuts down without leaving a Python process behind.
13. Two pages of identical layout but different text produce two distinct rows in the SPA, with each row's Markdown coming from the matching `page_done` event's `markdown_url` (no cross-contamination between pages).

### Non-Functional Tests
1. End-to-end runtime on the reference fixture meets the ≥ 30 percent reduction target.
2. Time-to-first-page on the same fixture is ≤ 5 seconds on a warm process.
3. With a one-hundred-page synthetic document, the SPA stays under the browser's interactive frame-budget threshold (instrument pinned by the plan).
4. Submitting two jobs in rapid succession returns a 409 Conflict on the second; the first job is unaffected.

## Dependencies

- **External Services**: None.
- **Internal Systems**: Existing Surya, DocLayout-YOLO, EasyOCR adapters; existing pipeline post-processing; existing Region / BBox / RegionRole types.
- **Libraries/Frameworks**: A Python ASGI server (e.g. uvicorn) and a Python web framework already in the ecosystem (e.g. FastAPI). SvelteKit with the static adapter on the frontend. Electron for the desktop shell. A Python docx-generation library (e.g. `python-docx`) is added for the .docx export; this is the only new Python dependency. No new ML dependencies.

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
| Docx export merges or skips text across page boundaries (a page's content bleeds into the previous or next docx page) | Medium | High | Build the docx from the per-page Markdown files, never from `result.md`; insert a hard page break between every pair of consecutive transcribed pages; cover the invariant with a structural test that inspects the docx XML, not just its plaintext. |
| Right-to-left rendering of Arabic in the docx looks wrong when opened in Word / LibreOffice | Medium | Medium | Set RTL paragraph direction on every paragraph at generation time; verify with a fixture-based test that opens the docx and asserts the property is set on every paragraph. |
| SPA mounting one DOM row per page bogs down on hundred-plus-page documents | Medium | Medium | Mandate virtualized rendering at the spec level; instrument the frame budget in CI as a non-functional test. |
| ASGI event loop blocks during long OCR calls, breaking SSE delivery and cancellation | Medium | High | Mandate off-loop inference at the spec level; add a service test that submits a cancellation while a page is in OCR and asserts the in-flight page commits and the state transitions to `cancelled` within a documented bound. |
| Electron BrowserWindow loads before the Python service is ready, showing a blank window | Medium | Low | Mandate the readiness handshake at the spec level; the shell does not navigate the BrowserWindow until `/healthz` returns ok. |

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

**Third pass — ChatGPT and Qwen, 2026-05-03 (user-driven, manually run by the repository owner)**:
- Both reviewers, independently, asked for a **skeletal API and SSE event contract** in the spec rather than deferring it to the plan. A new top-level **API and Event Contract (Skeletal)** section was added covering routes, request/response shapes, SSE event types, and the rule that the SPA constructs page asset URLs only by following the URLs emitted in `page_done` events (Qwen's "event-to-image mapping" point).
- Both reviewers asked for an **explicit job state machine**, especially for cancel and resume correctness. A new top-level **Job Lifecycle and State Machine** section was added with named states, transitions, and resume invariants. The "consistent on-disk state" definition was tightened in Desired State item 5.
- Both reviewers asked for the **per-job directory structure** to be defined at spec level. A new top-level **Job Directory Layout** section was added. The plan may add files inside the directory but may not move the existing ones.
- ChatGPT flagged that a **30 percent speedup target with no committed lever** was a gamble. Desired State item 8 now commits to page-render / OCR overlap as a non-optional speedup lever; the plan may add additional levers but cannot drop the overlap one.
- Qwen flagged that **"semantic equivalence" was not testable**. The Success Criteria entry now defines a canonical normalization (NFC, trailing whitespace stripped, multi-blank-lines collapsed, terminal newline) and mandates a fixture-based byte-identity test under that normalization. The "representative document" is now a checked-in **reference fixture** with documented page count, layout mix, and a hardware baseline.
- Qwen flagged the **ASGI event-loop blocking risk**. Technical Constraints now mandate that all model inference runs off the ASGI event loop. A risk row and a service test were added.
- Qwen flagged that the **model-reuse claim** silently included the CLI, where it cannot apply (each CLI invocation is a fresh process). Desired State item 7 and the Success Criteria entry now scope model reuse to **service mode only**.
- ChatGPT flagged the **fail-early vs lazy-import tension**. Desired State item 6 now defines two tiers: tier-1 (input checks, ≤ 1 second, synchronous) and tier-2 (backend warm-up, before the first page event, reported via `warming` SSE events). The "first second" promise is preserved for things that can be checked synchronously and is honestly relaxed for model loading.
- ChatGPT flagged **DOM bloat for long documents**. Technical Constraints now mandate virtualized rendering of the page-row list, and a Risk row covers it.
- ChatGPT flagged **Electron sidecar lifecycle gaps** (port collision, readiness race). Technical Constraints now mandate ephemeral-port binding (port 0), reading the chosen port from the child's stdio, and a readiness handshake against `/healthz` before the BrowserWindow loads. Risk rows cover both.
- ChatGPT flagged the **cancel-and-in-flight-page semantics**. Desired State item 5 now states that a page already inside the OCR call is allowed to finish and commit; only pages after it are skipped.

**Fourth, user-driven addition — docx export**:
- The repository owner added a hard requirement that the system export to `.docx` with **page boundaries preserved** (every PDF page becomes its own Word page; no text from one source page appears on a different docx page). Desired State gained a new item 9; Success Criteria, the API contract (`GET /jobs/{job_id}/result.docx`), the Job Directory layout (`result.docx`), Test Scenarios, Dependencies, and Risks were all updated to cover docx with hard page breaks, RTL paragraph direction, and a structural test that inspects the docx XML.

**Note on consultation coverage**: SPIR's default policy is two consultants. Across the four passes documented above, four distinct external reviewers contributed: Claude Opus, Gemini 3 Flash Preview, ChatGPT (user-run), and Qwen (user-run). Codex (GPT-5.4) was unavailable through this checkpoint due to a usage limit lasting until 2026-05-08; that gap was more than filled by the user-driven third-pass reviewers. The user is aware of and accepted this substitution.

## Approval
- [ ] Technical Lead Review
- [ ] Product Owner Review
- [ ] Stakeholder Sign-off
- [ ] Expert AI Consultation Complete

## Notes

- This spec deliberately keeps Surya, DocLayout-YOLO, and EasyOCR as the only OCR/layout choices. New models are out of scope; they belong in a follow-up survey and a separate spec.
- This spec deliberately keeps "SaaS" out of scope but does not preclude it. The plan must document any architectural decisions that would block a future SaaS layer.
- The fail-early discipline applies project-wide, not only to the new entrypoints; the plan may include a small refactor of the existing CLI to apply the same checks.
