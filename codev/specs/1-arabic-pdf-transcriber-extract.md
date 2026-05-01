# Specification: Arabic PDF Transcriber — Extract → Layout → ML Fallback → MD/Word Export

## Metadata
- **ID**: spec-2026-05-01-arabic-pdf-transcriber-extract
- **Status**: draft
- **Created**: 2026-05-01

## Clarifying Questions Asked

The issue (#1) and project brief together provide most answers. Remaining clarifying questions are tracked under **Open Questions** below for user/architect resolution during the spec review gate. The following questions were answered up front by the issue text:

| Question | Answer (from issue) |
|---|---|
| What is the primary language target? | Arabic, with multi-language as later nice-to-have. |
| Which output formats are first-class? | Markdown and Word (`.docx`). Other formats deferred. |
| Should ML always run? | No — native (text-layer) extraction is tried first; ML is a fallback when native is missing or insufficient. |
| What ML steps are required? | (a) layout detection per page, (b) per-region OCR/vision transcription. |
| Where do models come from? | Hugging Face / open-source preferred. Specific model selection (Qari-OCR, AIN, Surya, etc.) is part of plan/research. |
| Must reading order be reconstructed? | Yes, RTL-aware logical reading order matching the source. |
| Real-time / streaming / GUI? | Out of scope. CLI/library only. |

## Problem Statement

Arabic PDFs in the wild come in three broad flavours: (1) born-digital with a clean text layer, (2) born-digital but with broken/encoded text layers (ligatures shuffled, glyph IDs not Unicode, no spaces), and (3) image-only scans. General-purpose extractors (`pdftotext`, naive `pdfminer` use) frequently produce mojibake or wrong reading order on Arabic, and end users have no single tool that:

1. Tries the cheap, faithful, deterministic native path **first**,
2. Falls back to ML **only when needed**,
3. Reconstructs **logical, RTL-aware reading order** across multi-column / mixed-content pages,
4. Emits **clean Markdown and Word** that preserve headings, paragraphs, lists, and tables.

The result for users today is brittle pipelines, garbled text, and manual cleanup. This project produces a single CLI/library that handles all three flavours in one ordered pipeline with deterministic behaviour and predictable output quality.

## Current State

- Pure-text PDFs: `pdftotext`, `pdfplumber`, `pypdfium2` work for Latin scripts but routinely break on Arabic (wrong direction, glyph-not-Unicode, missing diacritics, broken word boundaries).
- Image PDFs: Tesseract supports Arabic but has weak layout understanding, no native RTL reading-order reconstruction across columns, and produces noisy output on dense or stylised Arabic typography.
- ML-only tools: Recent OCR / vision-language models (Qari-OCR, AIN, Surya, Marker, Docling) produce strong per-region results but are expensive when used on text-layer PDFs, and most do not expose a clean, RTL-aware Markdown/Word emitter.
- Layout-aware document understanding stacks exist (LayoutLMv3, DocLayout-YOLO, PubLayNet-class detectors) but are not wired into an Arabic-first pipeline with a deterministic native-first short-circuit.

Users today either pick one of the above and accept its weaknesses, or hand-stitch a pipeline. There is no single, opinionated, Arabic-first tool with `try-native → detect-layout → OCR-regions → reorder → export` as a contract.

## Desired State

A single Python package + CLI:

```
arabic-pdf-transcribe input.pdf -o output.md   # Markdown
arabic-pdf-transcribe input.pdf -o output.docx # Word
```

Behavioural contract:

1. Open the PDF. For each page:
   - Attempt **native text extraction**. Run a quality validator (encoding sanity, Arabic-character ratio, junk-glyph detection, word-boundary heuristics).
   - If the page passes validation, take the native text and the native layout (text blocks with bounding boxes) directly.
   - If the page fails validation **or** has no text layer, run **layout detection** (region segmentation), then **per-region OCR/vision transcription** with an Arabic-strong HF model.
2. Reconstruct **logical reading order** across regions: RTL-aware, column-aware, header/footer/caption-aware.
3. Emit **structured Markdown** (headings, paragraphs, lists, tables, figure-captions) and **structured Word** (matching styles).
4. Mixed pages (some pages native-clean, some image-only) are handled correctly within a single document.
5. The pipeline is deterministic where the native path is taken; ML stages are reproducible given fixed model weights and decoding parameters.
6. The library exposes the same pipeline programmatically with hooks for swapping the layout detector, the OCR model, the validator, and the emitter.

## Stakeholders

- **Primary Users**: Researchers, archivists, journalists, and developers who need clean Arabic text from PDFs (Quranic studies, historical archives, legal documents, scanned books).
- **Secondary Users**: Pipeline authors who want to embed the library in larger document-processing systems (RAG, search, translation).
- **Technical Team**: Maintainer (project owner). Implementation by SPIR builder agents.
- **Business Owners**: Project owner (`abdullah.nahass@gmail.com`).

## Success Criteria

- [ ] CLI accepts a single PDF path and emits Markdown to stdout or a file.
- [ ] CLI accepts a single PDF path and emits a `.docx` file.
- [ ] Library API exposes the same pipeline with explicit stages: `extract_native`, `validate_native`, `detect_layout`, `transcribe_regions`, `reorder`, `emit_markdown`, `emit_docx`.
- [ ] Native-first short-circuit: a born-digital Arabic PDF with a clean text layer produces output **without invoking any ML model** (verified by no model load + no GPU/CPU inference time on that path).
- [ ] On a benchmark set of Arabic PDFs covering each of the three flavours (digital-clean, digital-broken, image-scan), the tool produces non-empty, RTL-correct output for each.
- [ ] Output preserves logical reading order on a multi-column Arabic test page (verified by ordered-region snapshot test).
- [ ] Diacritics, hamza forms, and ligatures are preserved through to Markdown/Word output (no mojibake).
- [ ] Headings, paragraphs, and lists detected by the layout/native stage map to corresponding Markdown / Word styles.
- [ ] Tables (when detected) round-trip into Markdown tables and Word tables with correct cell content (basic grid; complex merged cells deferred — see Non-goals).
- [ ] Output is reproducible on the **native path**: same input + pinned PDF-library versions → byte-identical Markdown.
- [ ] Output is reproducible on the **ML path** within a Character Error Rate (CER) tolerance ≤ 0.05 across runs with pinned model revisions, decoding parameters, and seeds (CPU path; GPU floating-point determinism not guaranteed).
- [ ] All Hugging Face model dependencies are pinned by revision (commit hash), not just name; PDF-extraction libraries are pinned to exact versions.
- [ ] License audit passes: every model and code dependency has a license compatible with the project license.
- [ ] All tests pass with >=80% line coverage on the pipeline modules (validator, reorder, emitters mandatory; ML adapters can be lower).
- [ ] Documentation: README with install/usage, architecture diagram of the pipeline, and a model-card section listing chosen models, revisions, and licenses.

## Constraints

### Technical Constraints
- **Native-first ordering is non-negotiable** — ML stages may not run if native extraction is sufficient, for cost, speed, and fidelity reasons.
- **Hugging Face preferred** for both layout detection and OCR; non-HF ML allowed only when no comparable HF option exists, and must be justified in the plan.
- **RTL-aware reading order is mandatory** — left-to-right column emission for an Arabic document is a defect.
- **Python ecosystem** is assumed (Hugging Face, common PDF libraries, python-docx).
- **Offline-capable** — once models are cached, no network is required to transcribe.
- **CPU must work** — GPU is a speed optimisation, not a hard requirement. Smaller fallback models acceptable on CPU.
- **No proprietary cloud APIs** in the default pipeline (Azure/Google/OpenAI optional plugins only, off by default).
- **License-compatibility** — all default-pipeline model weights and code dependencies must have licenses compatible with the project license.

### Business Constraints
- Solo maintainer; complexity must be justified.
- No paid-API costs in the default pipeline.

## Assumptions

- Most target documents are page-based PDFs (not form/AcroForm-heavy). Forms are out of scope this iteration.
- Arabic content dominates targeted documents; mixed Arabic/Latin pages occur and must not break the pipeline.
- Model weights can be downloaded once and cached locally.
- Python 3.10+ is acceptable as the minimum supported runtime.
- Modern desktop hardware (CPU + 16 GB RAM) is the floor; GPU-only models are optional.
- The project license is permissive (MIT/Apache-class); model license compatibility is checked against this.

## Solution Approaches

### Approach 1: Layered pipeline with explicit gate per page (RECOMMENDED)

**Description**: Each page is processed independently through an ordered pipeline. Page-level gating decides whether ML is needed.

```
PDF
  └─ for each page:
       ├─ native_extract()          # text + native blocks + bboxes
       ├─ validate_native()         # quality gate
       │     ├─ pass → use native blocks as regions
       │     └─ fail → ML branch:
       │              ├─ layout_detect()       # regions per page
       │              └─ transcribe_regions()  # per-region OCR/VLM
       └─ reorder(regions)          # RTL-aware logical order
                                    # (heading/list/table classification)
  └─ emit_markdown() / emit_docx()
```

**Pros**:
- Clean separation of concerns; each stage is independently testable.
- Per-page gating means a mostly-clean PDF with a few scanned pages doesn't pay full ML cost.
- Plugin points are obvious (validator, layout detector, OCR model, emitter).
- Native and ML branches converge on a uniform internal representation (regions with text + bbox + role), so reorder/emit logic is shared.

**Cons**:
- Requires designing the unified internal region representation carefully so both branches produce identical-shaped output.
- Validator quality is critical; a bad validator either runs ML too often (slow) or accepts garbage (wrong output).
- Reading-order reconstruction is non-trivial for complex multi-column Arabic layouts with footnotes and marginalia.

**Estimated Complexity**: Medium-High.
**Risk Level**: Medium.

### Approach 2: ML-only — always run layout + OCR

**Description**: Skip native extraction entirely; treat every page as if it were image-only.

**Pros**:
- Simpler pipeline (one branch).
- More uniform output quality.

**Cons**:
- Violates the issue's explicit ordering constraint (native first).
- Slow and expensive on clean digital PDFs.
- Lower fidelity on text-layer PDFs that have authoritative Unicode (re-OCR'ing perfect text introduces noise).
- Requires GPU for acceptable speed on long documents.

**Estimated Complexity**: Low-Medium.
**Risk Level**: High (rejects a constraint).

### Approach 3: Native-only with heuristic fallbacks (no ML)

**Description**: Combine multiple text-layer extractors with heuristic post-processing to fix broken Arabic encodings.

**Pros**:
- No ML dependency; small footprint, fast.

**Cons**:
- Cannot handle image-only PDFs at all.
- Heuristic Arabic fix-ups are fragile and produce silent corruption on edge cases.
- Layout reconstruction without ML is weak on multi-column scans.

**Estimated Complexity**: Medium.
**Risk Level**: High (does not meet image-PDF success criterion).

### Approach 4: End-to-end VLM (single-model, page-image-in / Markdown-out)

**Description**: Use a single document-level vision-language model (e.g. a Marker-class or Qari-class model) and let it emit Markdown directly per page; bypass explicit layout/OCR separation.

**Pros**:
- Simple integration if the chosen model is good.
- Some VLMs already emit reading-ordered Markdown.

**Cons**:
- Black-box: no plugin points; quality is bound to the chosen model.
- Cost/speed identical to Approach 2 — always-on ML — violates native-first rule.
- Hard to debug regressions (no intermediate representation).
- Word/`.docx` emission still needs a separate styled-emitter pass.

**Estimated Complexity**: Low.
**Risk Level**: Medium-High (rejects native-first).

**Recommendation**: **Approach 1**. It satisfies all constraints, provides clear plugin points, and lets the plan stage choose specific models for the layout and OCR slots without changing the rest of the pipeline.

## Resolved Decisions

The following decisions are made up-front in this spec to unblock planning. Each can be revisited during the spec-approval gate.

- **Project license**: **MIT**. All default-pipeline model weights and code dependencies must be license-compatible (i.e. permissively redistributable for use, with attribution). Models with non-commercial-only or copyleft-derivative licenses are excluded from the default pipeline; users may opt them in at their own discretion.
- **Validator shape (accepted)**: a multi-signal validator that combines (a) Arabic-codepoint ratio (within the expected band for pages whose rendered glyphs are Arabic), (b) replacement / private-use / glyph-ID-class character count below a ceiling, and (c) word-boundary plausibility (whitespace and Arabic word-shape transitions within an expected distribution). All three signals must agree the page is "clean" for the native path to be taken; if any signal flags the page as suspicious, the ML branch runs. Concrete thresholds and the exact word-boundary statistic are tuned in the plan/research step.
- **Model distribution**: **download-on-first-use via the Hugging Face cache** (standard `transformers` / `huggingface_hub` behaviour). The package does not bundle weights. Users who need offline-only installs run `huggingface-cli download <pinned-revision>` ahead of time; this workflow is documented in the README.
- **Reproducibility scope**: byte-identical output is required only on the **native path** with pinned PDF-library versions. The **ML path** is reproducible up to model floating-point determinism — tests on the ML path use a Character Error Rate (CER) tolerance against a reference transcription rather than byte-equality. Both paths pin: (i) Python version, (ii) PDF-extraction library versions, (iii) HF model commit-hash revisions, (iv) decoding parameters and seeds.
- **Semantic output contract (v1 minimum)** — the unified region representation classifies each region as one of: `heading`, `paragraph`, `list-item`, `table`, `figure`, `caption`, `header-footer` (the last is dropped from output by default). Heading **level** in v1 is inferred from the layout-detector class plus a font-size / region-height heuristic, mapping to Markdown `#`/`##`/`###` (cap at H3 in v1; deeper levels collapse to H3). On the **native path**, heading detection uses native font-size signals; absence of font-size info downgrades all detected headings to a single H2 level rather than emitting plain paragraphs. Lists are detected by bullet/number prefix patterns (Arabic and Western) on both paths. Tables are basic grids only (per Non-Goals).
- **Failure behaviour and CLI exit codes**:
  - Default mode is **best-effort per page**: a single page that fails (parser error, OOM in OCR, model decode failure) produces a placeholder region in the output marked with a `<!-- transcription-failed: page N reason: ... -->` HTML-comment in Markdown (and a styled "Transcription failed (page N)" paragraph in Word), and the pipeline continues. The CLI exits with code `0` if at least one page transcribed, `2` if all pages failed.
  - `--strict` mode (opt-in) aborts on the first per-page failure with exit code `2` and a clear stderr message.
  - Encrypted / password-protected PDFs: refuse with exit code `3` and a message stating the PDF is encrypted; no `--password` flag in v1 (deferred).
  - Unsupported / corrupted PDF (cannot open at all): exit code `4` with a clear error.
  - Model download/cache miss in offline mode: exit code `5` with a message naming the missing model and the `huggingface-cli download` command to run.
  - OOM during ML inference: caught, treated as per-page failure (or `--strict` abort), with a hint to reduce page rasterisation DPI or use the smaller fallback model.
- **CLI format selection**: format is selected by **`--format md|docx`** when the output is stdout or has no extension; when `-o PATH` is given with a recognised extension (`.md`, `.markdown`, `.docx`), the extension drives the format and an explicit `--format` that disagrees is an error. Stdout defaults to Markdown.
- **Progress reporting**: a progress indicator (page X of N) is written to **stderr** by default; `--quiet` suppresses it; `--json-logs` emits structured per-page log lines on stderr instead.
- **Temp file lifecycle**: page rasterisation uses an in-process buffer where possible; when a temp file is unavoidable, it is created in a process-scoped temp directory (`tempfile.TemporaryDirectory`) that is removed on exit (success or failure). No artifacts are left in `/tmp` on shared systems.

## Open Questions

### Critical (Blocks Progress)

(All previously-critical items are resolved above. Reopen during spec review if any default is wrong for this project.)

### Important (Affects Design)

- [ ] **Target Arabic varieties** — Modern Standard Arabic only, or also Classical (Quranic, with full diacritics) and dialectal handwriting? Drives OCR-model selection.
- [ ] **Handwritten Arabic in scope?** — the issue does not mention it; default proposal is **out of scope** for this iteration (printed Arabic only).
- [ ] **Tables** — what fidelity? Proposal: detect-and-emit basic grid tables; complex merged cells / nested tables → out of scope.
- [ ] **Figures and equations** — Proposal: detect figures, embed as image references in the Markdown/Word output; do not OCR equations as text in v1.
- [ ] **Page-range / batch / single-file CLI surface** — proposal: `arabic-pdf-transcribe FILE [--pages 1-10] [--out PATH] [--format md|docx]`. Batch over a directory deferred.
- [ ] **Confidence reporting** — should the output include per-region confidence (as JSON sidecar)? Proposal: optional `--debug-json` flag emits sidecar; primary outputs stay clean.

### Nice-to-Know (Optimization)

- [ ] Prefer Qari-OCR vs Surya vs AIN as the **default** OCR — to be benchmarked in the plan's research step, not here.
- [ ] Layout detector: HF `cmarkea/dit-base-layout-detection`-class vs DocLayout-YOLO vs Surya layout — to be benchmarked.
- [ ] Multi-language deferral path: design the pipeline so a future `--lang` flag can swap the OCR adapter without rewriting the pipeline.

## Performance Requirements

These are **target floors**, not hard SLAs (this is a CLI/library, not a service):

- **Native-only path**: transcription of a 10-page born-digital Arabic PDF in under 5 s on a modern CPU laptop, no ML model loads.
- **ML path on CPU**: transcription of a 10-page image-only Arabic PDF in under 5 minutes on a modern 8-core CPU laptop (16 GB RAM), using the chosen default OCR model.
- **ML path on GPU**: same workload in under 1 minute on a single mid-range consumer GPU (8 GB VRAM).
- **Resource ceiling**: peak RSS under 8 GB on the CPU path with the default models.
- **Reproducibility**: same input + pinned model revisions + fixed seeds → byte-identical Markdown output.

## Security Considerations

- **Untrusted PDF handling**: PDFs are a known attack surface. The pipeline must use libraries that do not execute embedded JavaScript or follow embedded URIs. PDF parsing happens in-process; document-level sandboxing is out of scope but the chosen libraries must be vetted for safe parsing defaults.
- **Model supply chain**: HF model weights must be pinned by revision hash, not floating tags. Loaded model files should be verified against the pinned revision.
- **No telemetry / no network at inference**: after first download, the pipeline runs offline; no automatic phone-home.
- **No execution of model-emitted content**: emitter-side outputs (Markdown/Word) must escape any sequences that downstream renderers could interpret as active content (HTML in MD, embedded macros in Word — Word output uses python-docx with no macro support, so VBA injection is not a vector here).
- **License compliance** — every model/dep has a license recorded; the pipeline refuses to load a model whose recorded license is incompatible with the project license (configurable but defaults strict).

## Test Scenarios

### Functional Tests

1. **Happy path — clean digital Arabic PDF** → native extraction passes the validator → Markdown matches a byte-identical snapshot. No model loaded.
2. **Happy path — clean digital mixed Arabic/Latin PDF** → native extraction passes; both directions handled correctly.
3. **Image-only Arabic scan** → native fails the validator → layout + OCR run → Markdown is non-empty and matches reference within CER ≤ 0.10 on the test page; reading order RTL-correct.
4. **Mixed PDF** — some pages digital-clean, some image-only → per-page gating produces correct output for both kinds in one document.
5. **Broken text layer** — digital PDF with mojibake / non-Unicode glyphs → validator rejects it → ML branch runs → output is correct.
6. **Multi-column Arabic page** → reading order across columns is RTL-correct (region-order snapshot is the canonical assertion; "RTL-correct" operationally means: for any two regions A and B sharing a row band, the right-most one appears first; column-major order otherwise).
7. **Headings / paragraphs / lists** preserved as Markdown headings / paragraphs / list items, and as Word styled paragraphs.
8. **Basic table** detected and emitted as a Markdown table and a Word table with matching cell content.
9. **CLI usage**: `arabic-pdf-transcribe FILE -o out.md` and `... -o out.docx` produce non-empty, well-formed outputs; `arabic-pdf-transcribe FILE --format md` to stdout produces well-formed Markdown.
10. **Library usage**: pipeline stages are independently invokable and a custom adapter can be injected (verified by replacing the OCR adapter with a stub).
11. **Empty PDF** (0 pages) — produces empty (or single-blank-line) output and exit code `0` for `.md`, a valid empty `.docx` for Word; does not crash.
12. **Password-protected PDF** — exits with code `3` and a clear stderr message; no partial output written.
13. **Truncated / corrupted PDF** — exits with code `4` and a clear error; temp directory is cleaned up.
14. **Per-page partial failure (default mode)** — a 5-page PDF where page 3 triggers an OCR failure produces 4 transcribed pages and a placeholder for page 3, exit code `0`.
15. **Per-page failure under `--strict`** — same input as above aborts on page 3 with exit code `2`.
16. **Intra-region bidi mixing** — an Arabic paragraph containing an English citation preserves the citation's LTR run inside the RTL paragraph in both Markdown and Word output (Unicode bidi class assertions, not visual rendering).
17. **Huge single page** (e.g., A2 poster scan) — pipeline either transcribes within memory ceiling or fails cleanly with the OOM exit path (`--strict`) / per-page placeholder (default).
18. **Format-extension mismatch** — `--format docx -o out.md` exits with code `4` and a clear error; matching extension or no-extension cases work.

### Non-Functional Tests

1. **Performance** — native-only path within target floor on a representative 10-page digital PDF.
2. **Performance** — ML path within target floor on a representative 10-page scan (CPU and, if available, GPU).
3. **Reproducibility** — repeated runs on the same input with pinned models produce byte-identical Markdown.
4. **License audit** — automated check that every model and dependency in the resolved environment has a recorded compatible license.
5. **Encoding-safety** — outputs are valid UTF-8; characters outside the Arabic + ASCII ranges in test corpora are preserved.
6. **PDF-safety** — feeding a malformed/truncated PDF produces a clean error, not a crash or arbitrary execution.

## Dependencies

- **Internal**: none (greenfield project).
- **Libraries (candidates, final selection in plan)**:
  - PDF parsing: one of `pypdfium2`, `pdfplumber`, `pdfminer.six` for text-layer extraction; `pdf2image` or `pypdfium2` for page rasterisation.
  - Layout detection: a Hugging Face document-layout model (DiT-based, DocLayout-YOLO, or Surya).
  - OCR / VLM transcription: a Hugging Face Arabic-strong model (Qari-OCR, AIN, Surya, or comparable).
  - Output: `python-docx` for Word; built-in string formatting (or `markdown-it-py` for round-trip validation) for Markdown.
- **Hugging Face Hub**: required for first-time model download (transformers + huggingface_hub).

## References

- GitHub issue #1 — Arabic PDF transcriber — extract → layout → ML fallback → MD/Word export.
- Hugging Face model hub: `https://huggingface.co/models?language=ar&pipeline_tag=image-to-text` (model selection during plan/research phase).
- SPIR protocol: `codev/protocols/spir/protocol.md`.

## Risks and Mitigation

| Risk | Probability | Impact | Mitigation Strategy |
|------|------------|--------|-------------------|
| Validator wrongly accepts broken native text → silent garbage output | Medium | High | Multi-signal validator (codepoint ratio + glyph-class check + word-boundary entropy); snapshot tests on broken-text-layer fixtures; default to ML when validator is uncertain. |
| RTL reading order incorrect on multi-column pages | Medium | High | Region-graph reading-order algorithm with explicit RTL pass; snapshot tests on a curated multi-column corpus. |
| Default OCR model has weak coverage of Quranic / classical Arabic with diacritics | Medium | Medium | Pluggable OCR adapter; document model coverage in the README model card; allow user to swap to a different HF model via config. |
| HF model license becomes incompatible after upgrade | Low | High | Pin model revisions by commit hash; license-audit test runs in CI on every dependency change. |
| Tables in Arabic PDFs are too varied to extract reliably | Medium | Medium | Scope to "basic grid tables" in v1; complex tables → out of scope and documented; emit a placeholder + figure capture as fallback. |
| ML stages too slow on CPU to be useful | Medium | Medium | Document GPU expectations clearly; allow a smaller / faster default model on CPU; per-page parallelism. |
| Untrusted PDFs trigger parser bugs | Low | Medium | Use libraries with safe-by-default parsing; never enable JS / external resources; fuzz-test with malformed PDF fixtures. |
| Hugging Face Hub unavailable at install time | Low | Low | Cache-first design; document offline workflow (`huggingface-cli download` ahead of time); pipeline runs offline once cached. |
| Snapshot tests are fragile across model versions | Medium | Low | Pin model revisions; separate "native path" snapshots (deterministic) from "ML path" snapshots (looser, character-error-rate threshold instead of byte equality). |

## Non-Goals (this iteration)

- Full multi-language coverage (Arabic-first; pipeline must not preclude later expansion).
- Real-time / streaming transcription.
- GUI / web app.
- Handwritten Arabic.
- PDF AcroForms / interactive form-field extraction.
- Equation OCR (figures captured as images, not transcribed).
- Complex/merged/nested tables.
- Output formats beyond Markdown and Word (HTML, plain text, JSON-with-positions deferred).
- Cloud / paid-API providers in the default pipeline.

## Benchmark Corpus

Tests reference a curated benchmark corpus stored under `tests/fixtures/pdfs/`, organised by flavour:

- `tests/fixtures/pdfs/digital-clean/` — born-digital Arabic PDFs with clean text layers (≥ 3 documents, mixing single- and multi-column layouts).
- `tests/fixtures/pdfs/digital-broken/` — born-digital with mojibake / non-Unicode glyph IDs (≥ 2 documents).
- `tests/fixtures/pdfs/image-scan/` — image-only Arabic scans (≥ 3 documents, including at least one multi-column page).
- `tests/fixtures/pdfs/mixed/` — pages of more than one flavour in the same document (≥ 1 document).
- `tests/fixtures/pdfs/edge/` — empty, password-protected, truncated, A2-poster, and intra-region bidi-mixing examples.

For each fixture, a sibling `*.expected.md` (and selectively `*.expected.docx`) reference is checked in. Native-path fixtures use byte-equality; ML-path fixtures use CER tolerance. License of each fixture must be permissive (or owned by the project) and recorded in `tests/fixtures/pdfs/LICENSES.md`.

## Expert Consultation
<!-- Filled by porch after 3-way consultation -->
**Date**: 2026-05-01
**Models Consulted**: GPT-5 Codex, Claude (Gemini Pro was unavailable due to upstream quota exhaustion at consultation time; architect approved 2/3 consultation per `[ARCHITECT INSTRUCTION | 2026-05-01T18:32:24.195Z]`).

### Iteration 1 — feedback summary

**Codex (REQUEST_CHANGES)** — addressed:
- Resolved all "Critical (Blocks Progress)" open questions (license, validator shape, model distribution) via the new "Resolved Decisions" section.
- Reconciled reproducibility inconsistency: native path is byte-identical; ML path uses CER tolerance ≤ 0.05.
- Added v1 semantic-output contract for the unified region representation (heading / paragraph / list-item / table / figure / caption / header-footer), heading-level inference rules, and list/table detection rules.
- Added explicit failure-mode behaviour and CLI exit codes (`0`/`2`/`3`/`4`/`5`), `--strict` mode, encrypted-PDF handling, OOM behaviour, and offline-cache-miss handling.
- Added the benchmark corpus section defining fixture ownership, shape, and what "RTL-correct" means operationally.

**Claude (COMMENT)** — addressed:
- Added per-page partial-failure error contract (best-effort default + `--strict`).
- Added progress reporting contract (stderr, `--quiet`, `--json-logs`).
- Added explicit CLI format-selection rule (`--format` flag vs. extension; conflict is an error).
- Added missing edge-case tests: empty PDF, password-protected, truncated, intra-region bidi, huge single page.
- Added temp-file cleanup constraint (process-scoped `tempfile.TemporaryDirectory`, removed on exit).
- Pinned PDF-extraction libraries by exact version in the reproducibility constraints.

**Gemini Pro** — unavailable (upstream API returned `429 You have exhausted your capacity on this model.` for ten retry attempts spanning ~4 minutes; consult command exited code 1). Per architect instruction, proceeding with 2/3 consultations and re-attempting once before final approval.

## Approval
- [ ] Technical Lead Review
- [ ] Product Owner Review
- [ ] Stakeholder Sign-off
- [ ] Expert AI Consultation Complete

## Notes

- Spec defines **WHAT** and **WHY**. Phase breakdown, file paths, and code structure belong in the plan.
- The unified per-page **region** representation (text + bbox + role) is identified here as the integration point between native and ML branches; its exact schema is a plan-phase decision.
- The pipeline is designed so a future multi-language extension swaps the OCR adapter (and possibly the validator) without changing the reorder/emit stages.
