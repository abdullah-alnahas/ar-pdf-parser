# Plan: Streaming Engine + Web/Electron Frontends

## Metadata
- **ID**: plan-2026-05-03-streaming-engine-and-frontends
- **Status**: draft
- **Specification**: [`codev/specs/2-streaming-engine-and-frontends.md`](../specs/2-streaming-engine-and-frontends.md)
- **Created**: 2026-05-03

## Executive Summary

Implement Approach 1 from the spec — a single long-running Python engine driving three frontends (existing CLI, new local web service, new Electron shell), with streaming page-level events, per-job on-disk artifacts, an in-process cancel/resume state machine, render/OCR overlap as the committed speedup lever, and a Word (.docx) exporter that preserves PDF page boundaries.

Work is decomposed into eight independently shippable phases. Each phase produces a single atomic commit and is gated on the spec's success criteria for the slice it covers. The phases are ordered so that the docx exporter (a hard user requirement) ships via the CLI before the web service is built — every phase except Phase 6 (SPA) and Phase 7 (Electron shell) is usable from the existing CLI on its own.

## Success Metrics

Inherited from `codev/specs/2-streaming-engine-and-frontends.md` (Success Criteria, Performance Requirements, Test Scenarios):

- [ ] All spec acceptance criteria green at end of Phase 8.
- [ ] Streaming output equivalence test (canonical normalization, byte-identical) gated in CI from Phase 2 onwards.
- [ ] End-to-end runtime on the reference fixture ≥ 30 percent shorter than the current CLI baseline by end of Phase 4.
- [ ] Time-to-first-page ≤ 5 seconds on a warm process by end of Phase 5.
- [ ] Docx export tests (page-break invariant, RTL on every paragraph, no cross-page text leakage) green by end of Phase 3.
- [ ] No reduction in overall test coverage at any phase commit.
- [ ] CLI test suite stays 100 percent green at every phase commit.

## Phases (Machine Readable)

```json
{
  "phases": [
    {"id": "phase_1", "title": "Engine refactor: singleton, events, per-page artifacts"},
    {"id": "phase_2", "title": "Job state machine, cancel/resume in-process, reference fixture, equivalence test"},
    {"id": "phase_3", "title": "Docx export with page boundaries (CLI flag)"},
    {"id": "phase_4", "title": "Speedup: render/OCR overlap and benchmark"},
    {"id": "phase_5", "title": "FastAPI service: routes, SSE, healthz, off-loop inference"},
    {"id": "phase_6", "title": "Svelte SPA: submit, side-by-side strip, virtualized rows"},
    {"id": "phase_7", "title": "Electron shell with port-0 sidecar and readiness handshake"},
    {"id": "phase_8", "title": "Documentation, cleanup, lessons learned"}
  ]
}
```

## Phase Breakdown

### Phase 1: Engine refactor — singleton, events, per-page artifacts
**Dependencies**: None

#### Objectives
- Lift adapter construction out of `cli.main()` and into a long-lived `Engine` object that can be reused across jobs.
- Introduce a typed event channel so the pipeline can emit `page_started`, `page_done`, `warming`, `pipeline_done`, etc., to any consumer.
- Make every page produce its own on-disk artifacts (PNG raster + Markdown chunk) inside a per-job directory before the pipeline finishes, so future phases can stream and resume.
- Preserve current CLI behavior bit-for-bit; this phase is engine-internal plumbing only.

#### Deliverables
- [ ] New module `arabic_pdf_transcribe.engine` defining `Engine` (holds the layout + OCR adapters) and `Job` (per-invocation state + job directory handle).
- [ ] New module `arabic_pdf_transcribe.events` defining `PipelineEvent` dataclasses matching the SSE event types in the spec (job_accepted, warming, page_started, page_done, page_failed, state_change, pipeline_done, pipeline_failed) and an `EventEmitter` protocol.
- [ ] Modified `pipeline.transcribe()` accepts a `job_dir: Path` and an `EventEmitter`, writes `pages/NNNN.png` (the page raster) and `pages/NNNN.md` (the per-page Markdown) for every page that completes, writes `manifest.json` + `events.jsonl` continuously, and writes `result.md` at completion.
- [ ] `cli.main()` rewritten to construct an `Engine`, build a `Job` against a default per-invocation directory under `~/.cache/arabic-pdf-transcribe/jobs/<uuid>/`, attach a stderr `EventEmitter`, and copy `result.md` to the user's `-o` path on success.
- [ ] CLI prints a one-line status per `page_done` event when stdout is a TTY (mirroring today's progress bar behavior), JSON Lines events when `--json-logs` is set.
- [ ] Unit tests for `Engine` construction (no model load until first `transcribe`), for the `EventEmitter` contract, for `manifest.json` schema, for `events.jsonl` append-only invariant.
- [ ] Existing CLI test suite still passes unchanged.

#### Implementation Details

- `Engine` holds `layout_detector: LayoutDetector | None`, `ocr_transcriber: OCRTranscriber | None`, plus a small in-memory cache keyed by `(layout_backend_name, ocr_backend_name)` so a second job with the same backends reuses adapters. Adapters are constructed lazily on first `Engine.warm(...)` call.
- `Job` carries `job_id: str` (uuid4 hex), `job_dir: Path`, `manifest: Manifest`, an open file handle on `events.jsonl`, and the full set of validated CLI options. The job directory is created eagerly so tier-1 errors surface immediately.
- `PipelineEvent` is a `@dataclass(frozen=True)` discriminated by a `kind` field. JSON serialization uses a single `Event.to_dict()` method that produces the SSE payload shape directly (job_id always present, page_index 1-based).
- `EventEmitter` is a `Protocol` with `emit(event: PipelineEvent) -> None`. The pipeline depends only on the protocol. Two concrete emitters ship in this phase: `FileEmitter` (writes `events.jsonl`) and `StderrEmitter` (used by the CLI). A `MultiEmitter` fans events out to several listeners (Phase 5's service emitter will join here).
- The `pipeline.transcribe()` rewrite still drives pages serially; render/OCR overlap is deferred to Phase 4. Per-page write order is **PNG → Markdown → events.jsonl append**, matching the spec's source-of-truth rule for the events log.
- `manifest.json` is rewritten in place atomically (temp file + rename) on every `page_done`. `events.jsonl` is append-only, never rewritten.
- The CLI keeps its current `-o` flag semantics by copying `result.md` from the job dir; the job dir itself is retained per the spec's retention rule.

#### Acceptance Criteria
- [ ] Running `arabic-pdf-transcribe <pdf> -o out.md` on the reference fixture (added in Phase 2) produces an `out.md` whose canonical-normalization-byte-equivalent matches the same fixture's expected output committed in this phase.
- [ ] On the same run, `~/.cache/arabic-pdf-transcribe/jobs/<uuid>/pages/` contains one PNG and one Markdown file per processed page, all 1-based and zero-padded to four digits.
- [ ] `events.jsonl` contains exactly one `page_done` event per processed page, in monotonic page-index order, with absolute `image_url` and `markdown_url` paths matching the spec's API URL shape (even though no HTTP server runs yet).
- [ ] Submitting two jobs back-to-back inside the same Python process via the public `Engine` API loads adapters once (verified by mocking the adapter constructors and asserting call count).
- [ ] `pytest` is fully green; no test that passed before this phase fails.
- [ ] No public CLI flag added or removed.

#### Test Plan
- **Unit Tests**: Engine warm-cache hit/miss; FileEmitter append-only invariant; PipelineEvent → SSE-payload round-trip; manifest.json schema; per-page write ordering.
- **Integration Tests**: end-to-end CLI run on a small fixture verifying `pages/`, `manifest.json`, `events.jsonl`, and `result.md` shape.
- **Manual Testing**: run on the existing real-world Arabic fixture used in the survey notebook; eyeball that the per-page Markdown files match what the user already accepted as good output.

#### Rollback Strategy
Single-commit phase; revert with `git revert <hash>`. The CLI surface is unchanged, so reverting cleanly restores the prior behavior with no callers affected.

#### Risks
- **Risk**: per-page IO during the run measurably slows the CLI.
  - **Mitigation**: writes are buffered, `events.jsonl` append uses a single open file handle, and `manifest.json` rewrite is atomic but cheap. Phase 4's benchmark catches any regression.
- **Risk**: existing CLI users rely on the absence of a job directory.
  - **Mitigation**: directory lives under `~/.cache/`, not under the user's CWD; documented in the CLI `--help`.

---

### Phase 2: Job state machine, cancel/resume in-process, reference fixture, equivalence test
**Dependencies**: Phase 1

#### Objectives
- Introduce the `JobState` enum + transitions exactly as defined in the spec's Job Lifecycle section.
- Make cancel and resume work in-process for the CLI today (the same primitives will drive the service in Phase 5).
- Commit the reference fixture and the canonical-normalization equivalence harness so every later phase has a single objective output gate.

#### Deliverables
- [ ] `arabic_pdf_transcribe.engine.state` defining `JobState` enum (`accepted`, `warming`, `running`, `stopping`, `cancelled`, `finalizing`, `completed`, `failed`) and a transition table that rejects illegal moves.
- [ ] `Job` gains a `cancellation_token: threading.Event` (or `asyncio.Event`, depending on Phase 5's needs — the API is `is_set()`/`set()`/`wait()` either way) checked at every safe point: between adapter loads in `warming`, between pages in `running`, and inside the per-page validator/ML loop at coarse boundaries.
- [ ] `Job.resume()` recomputes pending pages from `manifest.completed_pages` minus the original `page_selection` and re-enters the loop.
- [ ] CLI gains `--cancel-on-signal` (default on): SIGINT triggers cancel and prints "Cancelled. Resume with: `arabic-pdf-transcribe --resume <job_id>`" rather than killing the process. `--resume <job_id>` is a CLI flag that re-attaches to a job dir and continues.
- [ ] Reference fixture committed at `tests/fixtures/pdfs/reference.pdf` (a small Arabic PDF with documented page count, layout mix, and licensing) plus an expected-output checked in alongside.
- [ ] `arabic_pdf_transcribe.testing.equivalence` module implementing the canonical normalization (line endings → `\n`, NFC, trailing-whitespace strip, multi-blank-line collapse, terminal newline) and a `assert_markdown_equivalent(actual, expected)` helper.
- [ ] CI test running the equivalence check on every commit; this becomes the regression gate for Phases 3–8.

#### Implementation Details

- The `accepted → warming → running` happy path runs synchronously today (no asyncio yet). Phase 5 wraps the same primitives in `asyncio.to_thread`.
- Cancel-during-warming requires checking the token between adapter loads (typically two adapters: layout + OCR). If the cancel arrives during a single adapter's `__init__` it is honored after that adapter finishes loading; the in-flight load is not interrupted.
- Cancel-during-running is checked between pages, not inside an OCR call. The in-flight page commits to disk before the state transitions to `cancelled`. This matches the spec.
- Resume is implemented by reading `manifest.completed_pages`, computing `pending = page_selection - completed`, and re-running `transcribe()` with `pending` as the page selection. The existing `page_selection` stays in `manifest` unchanged so Resume produces deterministic output.
- The reference fixture must be small enough to keep CI runtime acceptable but large enough to cover layout heterogeneity (body text, headings, at least one figure). A 5–8 page extract from a public-domain Arabic source meets both. Licensing recorded in `tests/fixtures/pdfs/LICENSES.md`.
- The expected-output Markdown is generated **once** by running the current CLI on the fixture immediately before this phase starts, captured under the canonical normalization, and committed. The Phase 1 implementation must already produce a normalization-equivalent file (it does; Phase 1's pipeline change is purely IO-shape, not transcription content).

#### Acceptance Criteria
- [ ] `pytest tests/test_equivalence.py` runs the new pipeline on `tests/fixtures/pdfs/reference.pdf`, normalizes both outputs, and asserts byte-identity.
- [ ] `pytest tests/test_state_machine.py` exhaustively covers every legal and illegal transition in the table; illegal moves raise a typed `InvalidStateTransition`.
- [ ] Cancel during warming on the reference fixture (simulated by a slow adapter constructor) transitions to `cancelled` without ever entering `running`.
- [ ] Cancel during running on the reference fixture leaves `events.jsonl` ending in `state_change(running → stopping)` then `state_change(stopping → cancelled)`, with the in-flight page's PNG + Markdown present and `manifest.completed_pages` reflecting it.
- [ ] `--resume <job_id>` against the cancelled job directory completes the document and `result.md` matches the expected output under canonical normalization.
- [ ] All previous tests still pass.

#### Test Plan
- **Unit Tests**: state machine transitions; cancellation token at warming and running boundaries; manifest.completed_pages updates; pending-page computation.
- **Integration Tests**: SIGINT during a CLI run produces a cancellable job; `--resume` completes it; equivalence test green.
- **Manual Testing**: run, Ctrl-C mid-page, observe message; rerun with `--resume`, verify final output.

#### Rollback Strategy
Revert the commit. The state machine is internal; the only public surface added is `--resume <job_id>` and the SIGINT trapping behavior. Removing them does not break anything that worked before Phase 2.

#### Risks
- **Risk**: SIGINT trapping interferes with users who currently rely on Ctrl-C to hard-kill the CLI.
  - **Mitigation**: Two consecutive SIGINTs within one second escalate to hard exit (matches modern CLI conventions like `kubectl`, `gh`).
- **Risk**: the reference fixture's Markdown changes if the underlying models update.
  - **Mitigation**: pin the model versions used to generate the expected output in `tests/fixtures/pdfs/LICENSES.md`; flag in CI whenever the equivalence test fails so we explicitly approve every regeneration.

---

### Phase 3: Docx export with page boundaries (CLI flag)
**Dependencies**: Phase 1

#### Objectives
- Ship the docx export the spec mandates as a hard user requirement, accessible from the CLI today (the service in Phase 5 picks it up for free).
- Guarantee the spec's invariants: hard page break between every pair of consecutive PDF pages; RTL paragraph direction on every paragraph; no cross-page text leakage; figure regions rendered as the literal text placeholder `[Figure: page N, region M]`.

#### Deliverables
- [ ] New module `arabic_pdf_transcribe.export.docx` providing `markdown_pages_to_docx(per_page_md_paths: Sequence[Path]) -> bytes` and `Job.write_docx(out_path: Path) -> None`.
- [ ] CLI flag `--docx <path>` (alongside the existing `-o` for Markdown). Specifying both is allowed; specifying only `--docx` skips Markdown copy.
- [ ] New dependency: `python-docx` (added to `pyproject.toml` core deps; the spec already approved it).
- [ ] Heading levels (`#`, `##`, `###`) → Word `Heading 1` / `Heading 2` / `Heading 3` styles; bullet/numbered lists → Word list styles; tables → Word tables; figures → literal `[Figure: page N, region M]` paragraph; everything else → `Normal` style.
- [ ] RTL is set on every paragraph via `<w:bidi/>` in `<w:pPr>` and on every table via `<w:bidiVisual/>` in `<w:tblPr>`.
- [ ] Hard page break inserted between consecutive per-page Markdown files (one `<w:br w:type="page"/>` between every adjacent pair, never before the first page or after the last).
- [ ] Tests under `tests/test_docx_export.py` cover: page-break count = N − 1; every `<w:p>` has `<w:bidi/>`; per-page text presence (run page N's known unique token through the docx and assert it appears in the section after the (N−1)-th page break and not after the N-th).

#### Implementation Details

- `markdown_pages_to_docx` parses each per-page Markdown with a small, well-tested Markdown-AST library (the plan suggests `markdown-it-py` because it produces a token stream that is easy to walk; we are not rendering HTML so no extra deps are needed).
- The token-walker maps each top-level token to a `python-docx` operation. Heading tokens become `document.add_heading(text, level)`; paragraphs become `document.add_paragraph()` with text runs added per inline token; lists become numbered/bulleted paragraphs; tables map straightforwardly via `document.add_table()`.
- For every paragraph and every table, an XML helper sets `<w:bidi/>` / `<w:bidiVisual/>`. This is the cleanest way to set RTL with `python-docx`; see `python-docx` README discussions for the canonical pattern.
- Between pages, `document.add_paragraph()` followed by `run.add_break(WD_BREAK.PAGE)` inserts the hard page break. The page-break paragraph itself is excluded from the "every paragraph has RTL" invariant by construction (it has no text), but we set RTL on it anyway for consistency.
- Figure handling: when the per-page Markdown contains a figure marker (today the pipeline emits an HTML comment `<!-- figure: page=N region=M -->` for figure regions whose text is empty), the exporter emits `[Figure: page N, region M]` as a paragraph. If the figure marker convention is not present today, Phase 3 also adds it to the per-page Markdown writer in Phase 1's pipeline so the exporter has something to detect.
- Cancelled jobs do not produce a docx; the CLI `--docx` flag returns a clear error if `manifest.status != "completed"`.

#### Acceptance Criteria
- [ ] `arabic-pdf-transcribe <pdf> --docx out.docx` on the reference fixture produces a docx whose page-break count = page count − 1.
- [ ] Inspecting `out.docx` with `unzip -p out.docx word/document.xml` shows `<w:bidi/>` inside every `<w:p>/<w:pPr>` element.
- [ ] A test file containing a unique sentinel token on each fixture page is processed; the sentinel for page N is found in the docx text body in the segment between page break (N−1) and page break N (or before the first break for N=1, or after the last break for N=total).
- [ ] `arabic-pdf-transcribe --resume <id> --docx ...` works on a Resume that ends in `completed`.
- [ ] `arabic-pdf-transcribe --docx ...` against a `cancelled` job returns a non-zero exit code and prints an actionable error.
- [ ] Markdown export remains the default and is unchanged.

#### Test Plan
- **Unit Tests**: per-page-MD-token → docx-element mapping for headings, paragraphs, lists, tables, figures; RTL XML attribute presence; page-break count.
- **Integration Tests**: full pipeline → docx on the reference fixture; sentinel-per-page boundary test; corrupt-Markdown handling returns a typed exception.
- **Manual Testing**: open generated docx in LibreOffice and visually confirm RTL rendering of Arabic.

#### Rollback Strategy
Revert the commit. The `--docx` flag is purely additive; removing it does not affect the Markdown path.

#### Risks
- **Risk**: a Markdown construct the parser cannot represent in docx silently produces wrong output.
  - **Mitigation**: parser failures raise; the CLI surfaces the failure with the offending page number; integration tests cover heading/list/table/figure constructs from the reference fixture explicitly.
- **Risk**: `python-docx` API changes between versions.
  - **Mitigation**: pin to a tested minor version in `pyproject.toml`.
- **Risk**: RTL still renders LTR in some Word/LibreOffice versions despite `<w:bidi/>`.
  - **Mitigation**: also set `<w:rtl/>` on every run inside the paragraph (belt-and-braces); document the tested-against versions in the docx export module's docstring.

---

### Phase 4: Speedup — render/OCR overlap and benchmark
**Dependencies**: Phase 1

#### Objectives
- Ship the spec's non-optional speedup lever: page N + 1 is rasterized while page N is in OCR.
- Commit the benchmark harness and the `tests/fixtures/pdfs/reference.pdf` baseline measurement so the ≥ 30 percent target is testable in CI.

#### Deliverables
- [ ] `pipeline.transcribe()` rewritten around an `asyncio.Queue(maxsize=4)` of pre-rasterized pages; rasterization runs in `asyncio.to_thread` (pdfium2 releases the GIL, so threads are real parallelism for the CPU-side rendering).
- [ ] OCR consumer task pulls from the queue and runs the existing OCR adapter call inside `asyncio.to_thread` — every blocking inference call lives off the asyncio event loop, satisfying the spec's ASGI mandate ahead of Phase 5.
- [ ] CLI keeps its synchronous-looking surface; under the hood it runs an asyncio loop.
- [ ] Benchmark harness `scripts/benchmark.py` (or `pytest tests/benchmark/test_speedup.py` if pytest-benchmark fits cleanly) runs the current CLI baseline (pre-Phase-1) vs the new pipeline on `tests/fixtures/pdfs/reference.pdf`, three runs each, reports median wall-clock and the reduction ratio.
- [ ] CI gate: median reduction must be ≥ 30 percent on the documented hardware baseline. If the gate runs on machines that don't match the baseline (e.g. CPU-only CI), the gate is informational, not blocking, and the binding measurement is run locally during release.
- [ ] Brief note in the commit's body recording the measured reduction.

#### Implementation Details

- The asyncio loop owns the queue, the rasterizer task, and the OCR consumer. The rasterizer runs ahead by up to `maxsize=4` pages (the spec's documented page-render-ahead concurrency). When the queue is full, rasterization back-pressures — memory stays bounded.
- Cancellation propagates by setting the cancellation token; both producer and consumer check it between iterations.
- Per-page event emission moves to the OCR consumer: `page_done` is emitted after both layout-detection and OCR finish for that page.
- `BENCHMARK.md` documents the methodology (warm process, cold process, GPU on/off matrix).
- Optional Phase-4 levers (batched OCR, fp16) are scoped here as **opportunistic**: if the overlap-only run already clears the 30 percent gate by a healthy margin, those levers are deferred to a follow-up. If the overlap run is below the gate, the plan adds them in this phase.

#### Acceptance Criteria
- [ ] On the reference fixture with default backends, warm process, GPU available, the new pipeline is ≥ 30 percent faster than the pre-Phase-1 CLI baseline.
- [ ] Equivalence test (Phase 2) stays green — the speedup must not change the output.
- [ ] No model inference call blocks the asyncio event loop (verified by a test that mocks the adapter to sleep and asserts SSE-style event emission stays responsive — proxied via the in-process EventEmitter latency in this phase, exercised end-to-end against the real service in Phase 5).
- [ ] Memory footprint during a long-document run does not grow unboundedly (verified by running on a 50+ page document and asserting RSS stays within a small multiple of model size).

#### Test Plan
- **Unit Tests**: queue back-pressure semantics; cancellation propagation; OCR-consumer event ordering.
- **Integration Tests**: benchmark green on reference fixture; equivalence test still green.
- **Manual Testing**: run the survey-style real-world fixture, confirm the wall-clock improvement matches the benchmark.

#### Rollback Strategy
Revert. The async pipeline replaces the synchronous one; reverting restores Phase 1's serial implementation and the ≥ 30 percent target is not met until the revert is reversed.

#### Risks
- **Risk**: 30 percent target is not met by overlap alone.
  - **Mitigation**: this phase has authority to add fp16 (where supported) and batched OCR within the same commit if needed; the plan is willing to expand its own scope inside this phase if the benchmark forces it.
- **Risk**: asyncio + GPU model interaction surfaces a subtle race (e.g. CUDA context confusion).
  - **Mitigation**: model inference stays inside `to_thread` calls (no new thread per page; bounded executor); add a stress-test that runs the pipeline 10 times back-to-back in the same Python process and asserts no resource leak.

---

### Phase 5: FastAPI service — routes, SSE, healthz, off-loop inference
**Dependencies**: Phase 1, Phase 2, Phase 4

#### Objectives
- Expose the engine over the HTTP + SSE surface the spec defines, on localhost only, single-job-at-a-time.
- Wire the existing in-process state machine and event emitter to the network: SSE clients get the same events the CLI sees in `events.jsonl`; cancel/resume/delete are HTTP-driven instead of SIGINT-driven.

#### Deliverables
- [ ] New package `arabic_pdf_transcribe.service` containing:
  - `app.py` — FastAPI application factory.
  - `routes/jobs.py` — POST /jobs, GET /jobs/{id}, GET /jobs/{id}/events, GET /jobs/{id}/pages/{n}/image, GET /jobs/{id}/pages/{n}/markdown, GET /jobs/{id}/result.md, GET /jobs/{id}/result.docx, POST /jobs/{id}/cancel, DELETE /jobs/{id}.
  - `routes/health.py` — GET /healthz.
  - `single_job.py` — an `asyncio.Lock`-style guard that returns 409 when a second submission arrives while a job is in flight.
  - `sse.py` — SSE helper backed by `events.jsonl` (replays from `Last-Event-ID`).
- [ ] CLI subcommand: `arabic-pdf-transcribe serve [--port 0] [--host 127.0.0.1]`. Port 0 binds to an ephemeral port, the chosen port is printed to stdout in a single line `LISTENING http://127.0.0.1:<port>` so Phase 7's Electron shell can parse it.
- [ ] New runtime dependencies: `fastapi`, `uvicorn`, `python-multipart` (multipart upload), `sse-starlette` or hand-rolled SSE.
- [ ] Engine inference calls all wrapped in `asyncio.to_thread` at the route boundary so the event loop stays responsive while a page is in OCR — Phase 4 already moved the bulk of the work off-loop; this phase ensures route handlers do not reintroduce blocking calls.
- [ ] Service tests using FastAPI `TestClient`: 201 on submit, 409 on second submit, 404 on unknown job, SSE replay with `Last-Event-ID`, cancel-during-warming honored, cancel-during-running drains the in-flight page, resume from cancelled, delete is idempotent, /healthz returns 200 with engine ready boolean.
- [ ] An integration test that submits a job, sets a small artificial delay in the OCR mock to keep a page "in flight", sends `POST /cancel {action: "stop"}`, and asserts the SSE stream closes after the in-flight page commits — proves the off-loop guarantee.

#### Implementation Details

- Single-job semaphore: a single `Job | None` slot inside the app. POST /jobs sets it; the slot is cleared on terminal state. While set, POST /jobs returns 409 with the active job's ID and status.
- SSE replay: GET /jobs/{id}/events accepts `Last-Event-ID`. Implementation streams `events.jsonl` from that line forward, then tails the file via an `asyncio.Event` triggered by the engine on every emission. The same `events.jsonl` the CLI writes is the source of truth — no parallel in-memory ring buffer.
- File uploads write to `<jobs_root>/<id>/input.pdf` directly; the service never holds the PDF in memory. Tier-1 input checks (writable dir, valid PDF magic bytes, valid page selection) run synchronously inside the route handler before returning 201.
- POST /jobs/{id}/cancel routes to the in-process Job's cancellation token; the handler returns immediately with 200, the actual state transition happens asynchronously and is observable via SSE.
- /healthz returns `{status: "ok", engine_ready: bool, active_job_id: str | None}` — used by Phase 7's Electron shell readiness handshake.
- Logging: every route handler logs structured JSON (route, status, job_id, latency).

#### Acceptance Criteria
- [ ] All HTTP routes from the spec's API table return spec-conformant responses (verified by route tests).
- [ ] All SSE event types from the spec are emitted with their required fields, including the `job_id` on every event.
- [ ] Cancellation during warming and during running both work end-to-end, with the same on-disk consistency invariants the CLI Phase-2 tests already cover.
- [ ] Time-to-first-page on the reference fixture (warm process) is ≤ 5 seconds when measured against the service.
- [ ] Service test suite is green; CLI test suite stays green.

#### Test Plan
- **Unit Tests**: SSE replay parser; single-job lock; multipart upload validation.
- **Integration Tests**: full route table; cancel-during-warming; cancel-during-running with off-loop assertion; resume from cancelled; delete idempotency; /healthz when warm vs cold.
- **Manual Testing**: `curl -F file=@reference.pdf -F meta='{"layout_backend":"full-page","ocr_backend":"surya"}' http://127.0.0.1:8080/jobs` then `curl http://127.0.0.1:8080/jobs/<id>/events` to watch the stream live.

#### Rollback Strategy
Revert. The service package is additive; the CLI continues to work without it.

#### Risks
- **Risk**: SSE clients miss events between connect and replay window.
  - **Mitigation**: `events.jsonl` is the source of truth; clients can always replay from byte 0 via Last-Event-ID = 0 to recover.
- **Risk**: a route handler re-introduces blocking in the loop.
  - **Mitigation**: a lint rule (or a manual pre-commit checklist documented here) forbids `time.sleep`, `requests.get`, and direct `engine.run_*` calls in route bodies. The Phase-4 latency assertion test catches regressions.

---

### Phase 6: Svelte SPA — submit, side-by-side strip, virtualized rows
**Dependencies**: Phase 5

#### Objectives
- Build the local web UI the spec describes: a single SvelteKit static-adapter SPA that submits jobs, streams page events live, renders side-by-side per-page rows in a virtualized vertical strip, and exposes the cancel/resume/delete and Markdown/docx download actions.
- Be the same artifact the Electron shell will load in Phase 7.

#### Deliverables
- [ ] New `frontend/` directory with a fresh SvelteKit project using the static adapter, `vite` for dev, and `pnpm` (or `npm`) for package management — the project picks the same toolchain the rest of the JS code in this repo already uses.
- [ ] Pages:
  - `/` — Submit form with file picker + drag-drop, backend selectors (layout dropdown, OCR dropdown), pages input, strict toggle. POSTs to `/jobs`, navigates to `/jobs/{id}`.
  - `/jobs/[id]` — Top bar with progress (current/total, percentage, smoothed ETA based on a moving median of the last 5 pages), Cancel button. Below: vertical strip of page rows.
- [ ] Components:
  - `ProgressBar.svelte` — derived from the SSE event stream.
  - `PageRow.svelte` — left: `<img src={image_url}>`, right: rendered Markdown (via `marked` or `svelte-markdown`).
  - `CancelMenu.svelte` — Resume / Keep / Delete. Visible only in the right state per the spec.
  - `DownloadButtons.svelte` — Markdown + docx, visible only when `pipeline_done` has fired.
- [ ] Virtualization: page-row list uses `svelte-virtual-list` (or comparable). Mounted DOM nodes proportional to viewport, not document length.
- [ ] SSE client connects to `/jobs/{id}/events` via `EventSource`; reconnection uses `Last-Event-ID`.
- [ ] Build artifact lives at `frontend/build/` (SvelteKit static adapter default). The FastAPI service from Phase 5 mounts this directory at `/` so production runs do not require a separate web server.
- [ ] `npm run dev` proxies `/jobs`, `/healthz`, etc. to FastAPI for hot-reload development.
- [ ] Vitest unit tests for ProgressBar derivation logic and ETA smoother. One Playwright end-to-end test that submits the reference fixture, waits for `pipeline_done`, asserts the docx download works.
- [ ] README section in `frontend/README.md` covering dev and build commands.

#### Implementation Details

- The SPA constructs page asset URLs **only** from the URLs present in the `page_done` event payload, per the spec's requirement.
- Cancel menu logic: visible if `state ∈ {running, stopping}`. Clicking shows three actions; each maps to `POST /jobs/{id}/cancel {action: …}`.
- Resume button: visible if `state = cancelled`.
- Delete button: visible in any non-running state.
- Progress derivation: total pages from the `job_accepted` event's `total_pages` field; completed from the count of `page_done` events received. ETA = median(last 5 `page_done.duration_seconds`) × remaining.
- Right-to-left text rendering: every `PageRow` Markdown container has `dir="rtl"` so Arabic text renders correctly. The original-page image needs no special handling.

#### Acceptance Criteria
- [ ] Submitting the reference fixture from `/` lands on `/jobs/{id}` and shows the first page row within ≤ 5 seconds (warm process).
- [ ] Subsequent pages stream in as additional rows without a full document re-render.
- [ ] Submitting a 100-page synthetic fixture stays interactive (frame budget green; the metric and instrument are pinned during the work and reported in the commit body).
- [ ] Cancel + Resume + Delete actions match the spec's state-machine rules end-to-end (driven through the UI).
- [ ] Markdown and docx download buttons appear after `pipeline_done` and produce the same files the CLI produces.
- [ ] `npm run build` produces a static `frontend/build/` directory; serving it from the FastAPI route does not require Node.

#### Test Plan
- **Unit Tests**: Vitest on ProgressBar derivation, ETA smoother, state-aware button visibility.
- **Integration Tests**: Playwright e2e against a running service on the reference fixture.
- **Manual Testing**: drive cancel/resume/delete by hand on a real Arabic PDF; eyeball RTL rendering and the side-by-side alignment on different viewport widths.

#### Rollback Strategy
The SPA lives in `frontend/`; reverting the commit removes it. The service from Phase 5 keeps working with no UI.

#### Risks
- **Risk**: virtualization library does not survive future Svelte version bumps.
  - **Mitigation**: pick a maintained library; pin its version; the SvelteKit code path that depends on it is small and replaceable.
- **Risk**: SSE reconnect storms when the user opens many tabs.
  - **Mitigation**: the service is single-job; multiple tabs read the same `events.jsonl` so the fan-out is cheap.
- **Risk**: docx download confuses the browser when content-disposition is missing.
  - **Mitigation**: service sets `Content-Disposition: attachment; filename="result.docx"` (covered by the Phase-5 test).

---

### Phase 7: Electron shell with port-0 sidecar and readiness handshake
**Dependencies**: Phase 5, Phase 6

#### Objectives
- Wrap the Phase-6 SPA in a thin Electron BrowserWindow that, on launch, spawns the Phase-5 FastAPI service as a child process, waits for `/healthz`, and points the window at the local URL.
- Keep the shell minimal: no Python bundling, no persistent state outside what the service already writes, no auto-update infrastructure.

#### Deliverables
- [ ] New `electron/` directory with `main.js`, `preload.js`, `package.json`.
- [ ] `npm run electron` (declared in the root or in `electron/package.json` per the project's existing convention) launches the shell.
- [ ] `main.js` spawns `python -m arabic_pdf_transcribe.service serve --port 0 --host 127.0.0.1`, parses the `LISTENING http://127.0.0.1:<port>` line from stdout, and stores the port.
- [ ] Readiness probe: poll `GET http://127.0.0.1:<port>/healthz` with a 30-second deadline; refuse to load the BrowserWindow if the probe fails and surface a native error dialog.
- [ ] BrowserWindow loads `http://127.0.0.1:<port>/` (the FastAPI service serves the SPA at `/`).
- [ ] Native file-dialog integration via Electron's `dialog.showOpenDialog`; the renderer sends the chosen path to the main process via `preload.js` and the main process forwards a multipart upload to the service. Drag-drop in the BrowserWindow keeps working without changes (it uses the same form post the SPA uses in browser mode).
- [ ] Quit lifecycle: on `before-quit`, send SIGTERM to the Python child, await exit with a 5-second timeout, then SIGKILL.
- [ ] Integration test (Node + spawn + pgrep): launch the shell against the reference fixture, run a small job to completion, quit, assert no Python process named `arabic_pdf_transcribe.service` remains.
- [ ] README updates documenting prerequisites (Python env active, dependencies installed) and the launch command.

#### Implementation Details

- Port discovery: the service prints exactly one line `LISTENING http://127.0.0.1:<port>` to stdout when the bind succeeds. The Electron main process reads stdout line-by-line; the first matching line yields the URL. Other stdout lines are forwarded to the Electron main-process console for debugging.
- Readiness handshake: poll every 200 ms with a 30 s deadline. If the deadline expires, kill the child and show a native dialog: "The transcription service did not start. Check that Python is installed and dependencies are present."
- Renderer lockdown: `main.js` disables remote URL navigation; only `http://127.0.0.1:<port>/*` is allowed.
- Single-instance: Electron's `app.requestSingleInstanceLock()` ensures a second launch focuses the existing window rather than spawning another sidecar.

#### Acceptance Criteria
- [ ] `npm run electron` opens a window that fully loads the Phase-6 SPA without manual intervention on a machine with the Python environment active.
- [ ] Killing the Electron process via the OS leaves no orphan Python service running (verified by the integration test).
- [ ] The shell refuses to load the window if the service does not become ready within the deadline; the user sees an actionable error.
- [ ] Drag-drop a PDF onto the window works identically to the browser mode.
- [ ] Two `npm run electron` invocations focus the existing window rather than spawning a second sidecar.

#### Test Plan
- **Unit Tests**: port-line parser; readiness probe with mocked HTTP.
- **Integration Tests**: full launch + small job + quit, assert no orphan; second-launch single-instance behavior.
- **Manual Testing**: drive the Electron UI by hand on the reference fixture; verify Markdown and docx downloads land in the OS-native Downloads folder.

#### Rollback Strategy
Revert. The shell is purely additive and lives in its own directory; the service and SPA continue to work in browser mode.

#### Risks
- **Risk**: the Python child outlives the Electron parent on hard kills (kernel `SIGKILL` to Electron).
  - **Mitigation**: write the child PID to a small file under `<jobs_root>/` at start; on next Electron launch, check for an existing service responding on its old port and either reuse it or kill it. (Stretch — if the integration test reveals this is a real problem in practice, the plan will add it; otherwise it stays out of scope.)
- **Risk**: stdout race — Electron reads the port line after the bind but before the bind happens (line buffering on Python).
  - **Mitigation**: the service's `serve` subcommand calls `print(..., flush=True)` immediately after the uvicorn server reports `Application startup complete`.

---

### Phase 8: Documentation, cleanup, lessons learned
**Dependencies**: Phases 1–7

#### Objectives
- Make the repository legible to a new developer landing fresh: how to run each frontend, what each command does, what the architecture looks like.
- Capture what was learned in a `codev/reviews/2-streaming-engine-and-frontends.md` per SPIR.

#### Deliverables
- [ ] README updates: a single section explaining the three entrypoints (CLI, web service, Electron shell), prerequisites for each, and the canonical commands.
- [ ] `docs/architecture.md` (or equivalent) updated with the new engine/service/SPA/shell topology and the per-job directory layout.
- [ ] `codev/resources/arch.md` updated to reflect the new modules.
- [ ] `codev/reviews/2-streaming-engine-and-frontends.md` written per SPIR's review template: what went well, what was hard, lessons learned, and any methodology improvements proposed.
- [ ] Removal of any dead code paths the new entrypoints made obsolete (only if truly unused — verified by `grep` and tests).
- [ ] CHANGELOG entry summarizing the work since the last release.

#### Implementation Details
This phase is documentation-heavy; no production code changes beyond dead-code removal. The lessons-learned document is the SPIR Review-phase artifact.

#### Acceptance Criteria
- [ ] A reader who has never seen the project can clone, install, and run each of the three frontends following only the README.
- [ ] `codev/reviews/2-streaming-engine-and-frontends.md` exists, follows the SPIR template, and references the phases by name.
- [ ] No dead code introduced by Phases 1–7 remains (verified by a manual sweep + the existing test suite still passing).

#### Test Plan
- **Unit Tests**: none for documentation; the existing tests must stay green.
- **Manual Testing**: a "fresh-eyes" walkthrough of the README on a clean machine, ideally via a git clone in a temporary directory.

#### Rollback Strategy
Documentation-only revert is safe; no functional impact.

#### Risks
- **Risk**: documentation drifts from the implementation as small follow-ups land.
  - **Mitigation**: documentation is part of the Phase 8 commit; subsequent changes that touch the entrypoints update the README in the same commit.

---

## Dependency Map

```
Phase 1 ─┬─▶ Phase 2 ─┬─▶ Phase 5 ─┬─▶ Phase 6 ─▶ Phase 7 ─▶ Phase 8
         │            │            │
         ├─▶ Phase 3 ─┘            │
         │                         │
         └─▶ Phase 4 ──────────────┘
```

Phase 3 (docx) depends only on Phase 1's per-page Markdown writer. Phase 4 (speedup) depends only on Phase 1's pipeline shape. Phase 5 (service) depends on Phase 1, the state machine from Phase 2, and the off-loop pipeline from Phase 4. Phases 6 and 7 are sequential after Phase 5. Phase 8 is the final wrap.

Phase 3 and Phase 4 are independent and can be reordered if practical reasons emerge during implementation; the plan keeps the docx-first order to honor the user's stated priority.

## Resource Requirements

### Development Resources
- One developer (the repository owner) plus AI agents collaborating through codev protocols.
- A development machine matching the documented hardware baseline for Phase 4's benchmark gate.

### Infrastructure
- No production infrastructure changes; everything is local-first.
- Disk: per-job artifacts under `~/.cache/arabic-pdf-transcribe/jobs/`; no quota imposed.
- Configuration additions: a single environment variable for the jobs root (default `~/.cache/arabic-pdf-transcribe/jobs/`); a single environment variable for the service port (default 0 / OS-chosen).

## Integration Points

### External Systems
- **HuggingFace Hub** (existing): model downloads. No change in this work.
- **Python-docx ecosystem** (new in Phase 3): docx generation. Vendored via PyPI dep.

### Internal Systems
- **Existing CLI** is preserved unchanged in surface; under the hood it switches to the new engine in Phase 1.
- **Existing pipeline modules** (`pipeline.py`, `roles/classify.py`, layout/OCR adapters) are wrapped, not rewritten.

## Risk Analysis

### Technical Risks

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| 30% speedup target missed by overlap-only | M | M | Phase 4 has authority to add fp16 + batched OCR within the same commit; if both fail to clear the gate, the plan is renegotiated with the user. | repo owner |
| Equivalence test flickers because of model-version drift | L | M | Pin model versions in `tests/fixtures/pdfs/LICENSES.md`; require explicit approval to regenerate the expected output. | repo owner |
| Off-loop guarantee regresses silently in a future PR | M | M | Phase 5 ships a latency-assertion service test; Phase 8 documents the rule in the README. | repo owner |
| Electron child orphaning under hard kill | L | M | PID file + reuse-or-kill on next launch; documented as a follow-up if the basic SIGTERM path proves insufficient. | repo owner |
| python-docx fails on a Markdown construct in the wild | M | M | Integration test covers the reference-fixture constructs; the docx path raises a typed error so the user sees a clear failure rather than silent corruption. | repo owner |

### Schedule Risks
The SPIR protocol does not measure schedule in calendar time; phases are gated by "done" or "not done." The only schedule-shaped risk is that Phase 4's benchmark gate fails repeatedly and forces multiple iterations; this is bounded by re-running the same harness with an additional lever, so the failure mode is convergent.

## Validation Checkpoints

1. **After Phase 1**: existing CLI tests green; per-page artifacts visible in the new job directory; equivalence-bit-pattern matches today's CLI on a quick smoke test.
2. **After Phase 2**: equivalence test in CI; cancel/resume manual smoke through the CLI.
3. **After Phase 3**: docx page-break + RTL invariants asserted; manual eyeball in LibreOffice.
4. **After Phase 4**: benchmark on the documented hardware shows ≥ 30 percent reduction; equivalence still green.
5. **After Phase 5**: full HTTP route + SSE replay tests green; cancellation drains the in-flight page; /healthz reflects engine state.
6. **After Phase 6**: SPA submits + streams + downloads; 100-page synthetic interactivity within the frame budget.
7. **After Phase 7**: shell launches, processes a job, quits without orphaning; second launch focuses existing window.
8. **After Phase 8**: README walkthrough on a clean machine succeeds end-to-end.

## Monitoring and Observability

This is a local-first project with no production deployment, so traditional metrics/alerting do not apply. What does:

- The service emits structured JSON logs on every route and every state transition; users can `tail -f` them for debugging.
- `events.jsonl` per job is itself a per-job event log, durable and grep-able.
- The CLI has a `--json-logs` flag that mirrors the same event shape on stderr.

## Documentation Updates Required
- [ ] Top-level `README.md` rewrite covering CLI, web, Electron entrypoints.
- [ ] `docs/architecture.md` covering engine/service/SPA/shell and the per-job directory.
- [ ] `codev/resources/arch.md` updates per the MAINTAIN protocol.
- [ ] `frontend/README.md` for the SPA dev workflow.
- [ ] `electron/README.md` for the desktop shell.

## Post-Implementation Tasks
- [ ] Re-run the survey notebook against the new pipeline to confirm the eyeball-quality match users already accepted.
- [ ] Capture release notes and tag a new project version.

## Expert Review

**Date**: pending
**Models Consulted**: pending — to be filled after the multi-agent review pass.
**Key Feedback**: pending.
**Plan Adjustments**: pending.

## Approval
- [ ] Technical Lead Review
- [ ] Engineering Manager Approval
- [ ] Resource Allocation Confirmed
- [ ] Expert AI Consultation Complete

## Change Log
| Date | Change | Reason | Author |
|------|--------|--------|--------|
| 2026-05-03 | Initial plan draft | SPIR Plan phase first artifact | repo owner via Claude |

## Notes

- Each phase is a single atomic commit per SPIR. Within a phase, the implementation may iterate locally before the commit.
- The plan deliberately schedules the docx exporter (Phase 3) before the web service (Phase 5) so the user gets the docx capability from the CLI as soon as Phase 3 lands.
- Render/OCR overlap is the only structural speedup lever the plan commits to up front. fp16 inference and batched OCR remain candidate levers for Phase 4 only if the overlap-only run misses the 30 percent gate.
- Cross-process resume remains out of scope per the spec; the in-process resume implemented in Phase 2 is sufficient for the work's success criteria.
