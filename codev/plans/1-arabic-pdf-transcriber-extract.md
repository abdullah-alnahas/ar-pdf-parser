# Plan: Arabic PDF Transcriber — Extract → Layout → ML Fallback → MD/Word Export

## Metadata
- **ID**: plan-2026-05-01-arabic-pdf-transcriber-extract
- **Status**: draft
- **Specification**: [codev/specs/1-arabic-pdf-transcriber-extract.md](../specs/1-arabic-pdf-transcriber-extract.md)
- **Created**: 2026-05-01

## Plan-Level Assumptions (Closing Spec's Open Questions)

The spec leaves several "Important (Affects Design)" open questions for plan time. This plan adopts the following defaults; each is the spec's proposed default and is documented here so phases 4–5 do not re-litigate during model selection:

- **Target Arabic varieties**: printed **Modern Standard Arabic** primary; **Classical** Arabic with full diacritics is a stretch goal evaluated during phase 5 model selection (model card records actual diacritics performance). Dialectal handwriting → out of scope.
- **Handwritten Arabic**: out of scope, per spec non-goal. Phase 5 model selection prioritises printed-Arabic OCR; handwriting capability, if present, is not tested.
- **Tables**: basic grid only (single-cell content, regular row/column structure). Merged cells, nested tables, complex headers → out of scope.
- **Figures / equations**: detect figures, embed as image-reference placeholders in Markdown / Word (e.g. `![figure on page N]()` / Word "Figure on page N" paragraph); equations not OCR'd as text.
- **Page-range CLI**: `--pages 1-N` accepted (zero-pad-tolerant, comma-separated ranges allowed: `1-3,7,10-12`).
- **Confidence reporting**: `--debug-json PATH` emits a JSON sidecar with per-region `confidence` and validator signal values.

## Executive Summary

This plan implements **Approach 1** from the specification: a layered, page-gated pipeline where each page is processed independently. Native text extraction is attempted first; a multi-signal validator decides whether the native result is trustworthy; if not, the ML branch (layout detection → per-region OCR) runs. Both branches converge on a unified `Region` representation, which is then RTL-reordered, role-classified, and emitted to Markdown or Word.

The work is broken into nine phases. The first three phases establish the deterministic, ML-free core (skeleton, native extraction, validator). The next three add the ML branch (layout detector, OCR adapter, reading-order reconstruction — the last is shared between branches but is implemented after both upstream sources exist). Phase 7 produces both emitters from the unified `Region` stream. Phase 8 wires everything together via the CLI and pipeline orchestrator with full failure-mode coverage. Phase 9 ships the benchmark corpus, end-to-end tests, license audit, and documentation.

Phases are intentionally ordered so that a useful intermediate artifact exists at every commit: by the end of phase 3 the package can already transcribe clean digital Arabic PDFs to plain text via a small driver script; by the end of phase 7 it can emit Markdown and Word from a hand-built `Region` list; phase 8 plugs the validator-gated pipeline behind both.

The package targets Python 3.10+, MIT-licensed. The default OCR and layout models are Hugging Face hosted, pinned by commit revision; selection is finalised in phase 4 / phase 5 as a documented research step inside each phase, not before.

## Success Metrics

- [ ] All specification success criteria met (re-validated at the end of phase 9 against the spec checklist).
- [ ] Test coverage ≥ 80 % on `validator`, `reorder`, `emit_markdown`, `emit_docx`, and `cli` modules; ≥ 60 % on ML-adapter modules (network/model loading is mocked at the adapter boundary).
- [ ] Native-only-path performance floor met on a 10-page born-digital Arabic PDF (CPU laptop, < 5 s, no model load).
- [ ] ML-path performance floor met on a 10-page image-only Arabic PDF (CPU < 5 min on 8-core, GPU < 1 min on mid-range).
- [ ] License-audit CI check is green: every model and runtime dependency has a recorded license compatible with MIT.
- [ ] Reproducibility: native-path snapshot tests are byte-identical across two CI runs; ML-path snapshot tests pass within CER ≤ 0.05 on the curated corpus.
- [ ] Documentation: README install/usage, architecture diagram, model card (chosen models, revisions, licenses), CLI exit-code table.
- [ ] No critical security findings: PDF parsing uses safe-by-default settings; no embedded JS / external resource resolution; emitter output is escape-safe.

## Phases (Machine Readable)

```json
{
  "phases": [
    {"id": "phase_1_skeleton_and_license_audit", "title": "Project skeleton, tooling, and license-audit harness"},
    {"id": "phase_2_region_model_and_native_extraction", "title": "Unified Region representation and native PDF extraction adapter"},
    {"id": "phase_3_native_text_validator", "title": "Multi-signal native-text quality validator"},
    {"id": "phase_4_layout_detection_adapter", "title": "Layout detection adapter (Hugging Face)"},
    {"id": "phase_5_ocr_transcription_adapter", "title": "Per-region OCR / VLM transcription adapter (Hugging Face)"},
    {"id": "phase_6_reading_order_and_roles", "title": "RTL-aware reading-order reconstruction and role classification"},
    {"id": "phase_7_emitters_markdown_and_docx", "title": "Markdown and Word (.docx) emitters"},
    {"id": "phase_8_cli_and_pipeline_orchestrator", "title": "CLI and pipeline orchestrator with full failure-mode coverage"},
    {"id": "phase_9_benchmark_corpus_and_docs", "title": "Benchmark corpus, end-to-end tests, license-audit CI, and documentation"}
  ]
}
```

## Phase Breakdown

### Phase 1: Project skeleton, tooling, and license-audit harness
**Dependencies**: None

#### Objectives
- Establish a Python 3.10+ package layout and the deterministic-by-default tooling chain.
- Land the license-audit harness now so every dependency added in later phases is automatically vetted.
- Pin every runtime dependency by exact version; pin every dev tool likewise.

#### Deliverables
- [ ] `pyproject.toml` declaring the package `arabic_pdf_transcribe`, exact pinned runtime deps (initially: `pypdfium2`, `python-docx`, `huggingface_hub`, `transformers`, `torch` listed with extras for CPU/GPU; final ML deps land in phases 4–5).
- [ ] Source tree:
  - `src/arabic_pdf_transcribe/__init__.py`
  - `src/arabic_pdf_transcribe/regions.py` (placeholder for the `Region` dataclass — schema lands in phase 2)
  - `src/arabic_pdf_transcribe/errors.py` — typed exception hierarchy (full set listed in phase 8; module created here so phases 4–5 can raise `ModelDownloadError` / `OCRTranscriptionError` without forward references)
  - `src/arabic_pdf_transcribe/_version.py`
- [ ] `LICENSE` (MIT, copyright the project owner).
- [ ] `README.md` stub (project name, one-paragraph summary, status badge, "see codev/specs/1-…").
- [ ] Tooling configs: `ruff` (lint + format), `pytest`, `coverage` config (`tool.coverage.run.source = ["arabic_pdf_transcribe"]`).
- [ ] `tools/license_audit.py` — reads the resolved `pip` environment + a project-local `models.toml` allow-list and fails on any non-permissive license. Depends only on the standard library + `tomllib` + `importlib.metadata`.
- [ ] `models.toml` (initially empty `[[models]]` array; populated in phases 4–5).
- [ ] `Makefile` (or `taskfile.yml`) targets: `lint`, `test`, `audit`, `all`.
- [ ] CI workflow `.github/workflows/ci.yml` running `lint`, `test`, `audit` on Python 3.10 and 3.11 (Linux only in v1).
- [ ] Smoke unit tests: `tests/test_skeleton.py` asserting the package imports and the version is a non-empty string.
- [ ] `tests/fixtures/pdfs/.gitkeep` and `tests/fixtures/pdfs/LICENSES.md` stub (corpus populated in phase 9).

#### Implementation Details
- `src/`-layout (not flat), so test imports go through the installed package and CI catches packaging mistakes.
- License-audit logic: read every distribution in the active environment via `importlib.metadata.distributions()`, extract the `License` / `License-Expression` / classifier fields, normalise (e.g. `MIT License` ≡ `MIT`), and compare against an allow-list (`MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `MPL-2.0` permitted; copyleft-derivative and "non-commercial only" excluded). The harness exits non-zero if anything fails.
- For ML model licenses (`models.toml`) the harness validates each entry has `name`, `revision`, `license`, and that `license` is in the allow-list. Models entries are added in phases 4 and 5.
- `Region` placeholder in phase 1 is a `TYPE_CHECKING`-only stub so the module imports; the real schema is the **first** task of phase 2.
- CI runs on `ubuntu-latest`; matrix `python-version: ["3.10", "3.11"]`. macOS / Windows deferred to a follow-up issue (out of scope this iteration).

#### Acceptance Criteria
- [ ] `pip install -e .[dev]` succeeds in a clean venv.
- [ ] `make lint test audit` is green.
- [ ] `tools/license_audit.py` fails as expected when the allow-list is artificially narrowed (test for the failure path is included).
- [ ] CI on the PR is green.

#### Test Plan
- **Unit Tests**: `test_skeleton.py` (import + version), `test_license_audit.py` (positive: allow-listed deps pass; negative: a fabricated `models.toml` entry with a forbidden license fails the audit; negative: a fabricated distribution metadata path fails the audit).
- **Integration Tests**: deferred to phase 9 (no behaviour exists yet).
- **Manual Testing**: run `make all` locally in a fresh venv; inspect `models.toml` schema.

#### Rollback Strategy
- Pure-additive phase. Revert the PR; nothing else depends on it yet.

#### Risks
- **Risk**: Some otherwise-acceptable upstream packages declare licenses inconsistently in metadata → audit false-positives.
  - **Mitigation**: maintain a small `tools/license_audit_overrides.toml` keyed by `<distribution-name>==<version>` with documented justification; PR reviewers are required to scrutinise additions.
- **Risk**: Pinning `torch` exactly fights downstream platforms (CPU vs CUDA wheels).
  - **Mitigation**: pin to a version that has both CPU and CUDA wheels on PyPI; document the install matrix in the README.

---

### Phase 2: Unified Region representation and native PDF extraction adapter
**Dependencies**: Phase 1

#### Objectives
- Define the `Region` schema both branches converge on.
- Implement the native-extraction adapter that produces a list of `Region`s per page from text-layer PDFs without invoking any ML.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/regions.py` — unified document model:
  - `Region` `@dataclass(frozen=True, slots=True)` carrying:
    - `page_index: int`, `bbox: BBox` (`x0, y0, x1, y1`, origin top-left, PDF user units),
    - `text: str` (empty for `figure` / `failure_placeholder`; cell text lives in `table_grid` for `table`),
    - `role: RegionRole` (enum: `heading`, `paragraph`, `list_item`, `table`, `figure`, `caption`, `header_footer`, `failure_placeholder`, `unknown`),
    - `source: RegionSource` (enum: `native`, `ocr`),
    - `confidence: float | None`,
    - `heading_level: int | None` — `1`/`2`/`3` when `role == heading`; `None` otherwise (per the spec's "cap at H3" rule),
    - `list_marker: ListMarker | None` — `ListMarker(kind: Literal["bullet","ordered"], ordinal: int | None, raw_marker: str | None)` when `role == list_item`,
    - `table_grid: TableGrid | None` — `TableGrid(rows: tuple[TableRow, ...])` where `TableRow = tuple[TableCell, ...]` and `TableCell(text: str, confidence: float | None, bbox: BBox)`. Set when `role == table`. Merged cells: not supported in v1; cells with row/colspan > 1 are flattened to per-row text and a `meta["v1_table_simplification"]=True` flag.
    - `group_id: str | None` — stable identifier shared across a `figure` and its surrounding `caption` regions to express figure-caption linkage. Generated as `f"p{page_index}-g{counter}"` per page during native extraction or by the layout adapter.
    - `failure_reason: str | None` — populated only when `role == failure_placeholder` (e.g. `"ocr_failed: model out-of-memory on page N"`). The Markdown / Word emitters render placeholders from this.
    - `meta: Mapping[str, Any]` — frozen `MappingProxyType`, used for adapter-specific signals (font-size histogram, layout class label, OCR decoding params).
  - Helpers: `iter_pages(regions)`, `Region.with_text(...)`, `Region.with_role(...)`, `Region.with_heading_level(...)`, `Region.as_failure_placeholder(reason: str, *, page_index, bbox) -> Region`.
  - Schema versioning: `REGION_SCHEMA_VERSION = "1"` exported; included in `Region.to_json()` to allow backward-compatible reads later.
- [ ] `src/arabic_pdf_transcribe/extract/native.py` — `extract_native(pdf_path: Path) -> Iterator[NativePage]`, where `NativePage` carries: `page_index`, `page_width`, `page_height`, `regions: list[Region]` (all `source=native`, `role=unknown`), `font_size_hist: Counter[float]`, `has_text_layer: bool`. The native extractor includes a **basic native-table detector** (`extract/_native_tables.py`): when `pypdfium2` exposes ruling lines or a regular grid of text bboxes, the corresponding regions are coalesced into a single `Region(role=table, table_grid=TableGrid(...))`; otherwise no table region is emitted (the page falls through to the layout-detector / heuristic table path in phases 4 / 6).
- [ ] One mixed-script fixture: `tests/fixtures/pdfs/digital-clean/lorem-ar-en-mixed.pdf` (one page mixing Arabic paragraphs and English citations; covers spec test scenario 2 from phase 2 onward, not deferred to phase 9).
- [ ] `src/arabic_pdf_transcribe/pdf/_pypdfium2_loader.py` — thin wrapper over `pypdfium2` that exposes only the calls native extraction uses; safe-defaults (no JS exec, no external resource resolution).
- [ ] Unit tests: `tests/test_regions.py`, `tests/test_extract_native.py` with synthetic PDFs (built in-process via `pypdfium2`'s render-from-string equivalents, or via a tiny pre-built fixture).
- [ ] In-tree text-layer fixtures:
  - `tests/fixtures/pdfs/digital-clean/lorem-ar-2col.pdf` — 2-page MIT-clearable Arabic-text PDF.
  - `tests/fixtures/pdfs/digital-clean/lorem-ar-en-mixed.pdf` — mixed Arabic/Latin (per Deliverables above).
  - `tests/fixtures/pdfs/digital-clean/lorem-ar-table.pdf` — a single-page document with one regular grid table (3×3) for native-table detection.
  - All licenses recorded in `tests/fixtures/pdfs/LICENSES.md`.

#### Implementation Details
- The PDF library choice is **`pypdfium2`** (Apache-2.0, actively maintained, exposes text + bbox per character/word/block, fast, and safe-by-default — does not execute JS).
- Native extraction strategy: walk text objects, group by line geometry, then by paragraph proximity. Bounding boxes are emitted at paragraph granularity. Font-size histogram per page is captured for later heading-level inference (phase 6).
- `has_text_layer` is `False` when no text objects exist; this short-circuits the validator to "fail" without running statistics.
- `Region.bbox` uses top-left origin (not PDF native bottom-left) — convert at the boundary so all downstream stages share one coordinate system.
- No reading-order logic in this phase; regions come out in extraction order. Reordering is phase 6.

#### Acceptance Criteria
- [ ] `extract_native` on the in-tree clean PDF returns regions whose concatenated text equals the PDF's underlying text content (modulo whitespace normalisation).
- [ ] `Region` is hashable, frozen, and serialisable to JSON via a stable schema (test asserts a round-trip).
- [ ] Coverage on `regions.py` and `extract/native.py` ≥ 90 %.

#### Test Plan
- **Unit Tests**: `Region` immutability, `with_*` helpers, JSON round-trip, `extract_native` happy path, no-text-layer short-circuit.
- **Integration Tests**: fixture PDF → expected region count + text content.

#### Rollback Strategy
- The new modules are isolated. Revert the PR; phase 1 still stands.

#### Risks
- **Risk**: `pypdfium2`'s paragraph grouping is too coarse / too fine on real Arabic PDFs.
  - **Mitigation**: paragraph grouping is a pluggable function; phase 6 reorders blocks to logical order regardless of extraction granularity, so over-fragmentation is recoverable.
- **Risk**: Coordinate-system bugs (BL vs TL origin) silently invert reading order.
  - **Mitigation**: a unit test asserts a known-position region lands in the expected TL-origin bbox on the fixture.

---

### Phase 3: Multi-signal native-text quality validator
**Dependencies**: Phase 2

#### Objectives
- Decide, per page, whether to trust the native extraction or fall back to the ML branch.
- Make the decision rule explicit, testable, and tuneable.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/validate/native_validator.py` — `validate_page(page: NativePage) -> ValidationResult` where `ValidationResult` carries `accept: bool`, `signals: dict[str, float]`, `reasons: list[str]`.
- [ ] Three independent signal functions:
  - `arabic_codepoint_ratio(text: str) -> float`
  - `replacement_glyph_ratio(text: str) -> float` — counts U+FFFD, private-use, `.notdef`-class glyph IDs, and high-frequency ligature placeholders.
  - `word_boundary_plausibility(text: str) -> float` — token-length distribution KL-divergence against a small bundled reference distribution for printed Arabic.
- [ ] `src/arabic_pdf_transcribe/validate/_reference_dist.json` — small reference distribution for the word-boundary plausibility signal; license-clean (derived from a permissive Arabic Wikipedia sample described in the file's header).
- [ ] Threshold defaults in code (with documented per-signal rationale); thresholds are configurable via `ValidatorConfig`.
- [ ] Unit tests covering: clean-Arabic-passes, mojibake-fails, mixed-script-passes-when-Arabic-portion-clean, no-text-layer-fails, replacement-glyph-storm-fails.
- [ ] Two new fixtures in `tests/fixtures/pdfs/digital-broken/` exhibiting mojibake and replacement-glyph patterns (license documented).

#### Implementation Details
- Decision rule: a page is `accept=True` iff **all three** signals fall inside their accept band. If any signal is outside, the page is rejected and the ML branch will run for it. This is the "all three must agree" rule from the spec's Resolved Decisions.
- Threshold tuning is a deliberate plan deliverable — initial values are derived from the in-tree fixtures and recorded in the module's docstring with the methodology.
- The reference word-boundary distribution is stored as a small JSON file (token-length frequencies). It is generated offline once by a one-off helper script in `tools/` and checked in; the script is included for reproducibility but is not part of the package.

#### Acceptance Criteria
- [ ] All clean fixtures pass; all broken fixtures fail.
- [ ] Replacing any single signal threshold with a flagrantly wrong value causes at least one fixture to flip to the wrong verdict (sanity check that no signal is dead weight).
- [ ] `ValidatorConfig` round-trips to / from a TOML file.

#### Test Plan
- **Unit Tests**: per-signal tests with hand-crafted strings; full validator tests with fixtures.
- **Integration Tests**: deferred (per-page validator runs against `NativePage`, exercised in phase 8 e2e).

#### Rollback Strategy
- Validator is pure (no I/O beyond reading the bundled JSON). Revert the PR; the upstream native extraction continues to work; downstream code that would consume the validator does not yet exist.

#### Risks
- **Risk**: Threshold defaults overfit the in-tree fixtures and misbehave on real PDFs.
  - **Mitigation**: phase 9 expands the corpus and runs the validator over it; thresholds are revisited then; rule **shape** is locked in this phase, but numbers are not.
- **Risk**: Reference word-boundary distribution is fragile (small sample).
  - **Mitigation**: signal contributes one of three votes; a single weak signal cannot wrongly reject by itself (all three must flag for accept; conversely, if *any* signal rejects, page is rejected — but then ML branch runs, which is the safer direction).

---

### Phase 4: Layout detection adapter (Hugging Face)
**Dependencies**: Phase 2 (Region schema)

#### Objectives
- Implement the ML-branch's layout-detection step as a swappable adapter.
- Pin the chosen model by commit revision in `models.toml`.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/layout/__init__.py` — `LayoutDetector` Protocol with `detect(page_image: PILImage, page_index: int) -> list[Region]` (regions carry bbox + tentative `role`; `text=""`, `source=ocr`). When the detector emits a `table` class, the adapter performs **structural cell detection** within the table bbox (using ruled-line detection on the page image via OpenCV-style morphological ops in `layout/_table_cells.py`; OpenCV is *not* a runtime dep — implementation uses `Pillow`-only morphology; complex layouts fall back to one-cell-per-row coalescing). The resulting `Region` carries `table_grid: TableGrid`, populated with empty `TableCell.text` strings; per-cell text is filled in by the OCR adapter in phase 5 (which is invoked per-cell).
- [ ] `src/arabic_pdf_transcribe/layout/hf_detector.py` — concrete adapter wrapping a Hugging Face document-layout model. Selection between {DiT-base layout, DocLayout-YOLO, Surya layout} is made inside this phase by benchmarking on `digital-broken` + (one synthetic) `image-scan` fixture; the chosen model + revision is recorded in `models.toml`. Decision is documented in the PR description and in the module's docstring with the criteria (Arabic robustness, layout class set, license, footprint, license).
- [ ] `src/arabic_pdf_transcribe/layout/_classes.py` — mapping from the chosen model's class labels to the project's `RegionRole` enum.
- [ ] `models.toml` updated with the layout-model entry: name, revision (commit hash), license, footprint MB, source URL.
- [ ] Unit tests using a tiny mocked detector (no network, no real model load); one optional `@pytest.mark.slow` test that exercises the real model on a single fixture page (skipped in CI by default; run nightly).
- [ ] One image-scan fixture: `tests/fixtures/pdfs/image-scan/scan-ar-1col.pdf` (license documented).

#### Implementation Details
- The adapter raises `ModelDownloadError` (cleanly mappable to CLI exit code 5) when the cache miss occurs in offline mode.
- `LayoutDetector` is a `runtime_checkable` `Protocol` to support stub injection.
- All `transformers` / `torch` imports are local to the adapter module (avoid heavy import on package load).
- Page rasterisation lives next to the adapter (`layout/_rasterise.py`) to keep the boundary clean: a page → PIL image at a configurable DPI (default 200).

#### Acceptance Criteria
- [ ] Mocked detector test exercises the path end-to-end with a fake model output.
- [ ] `models.toml` license-audit check passes for the chosen entry.
- [ ] Importing `arabic_pdf_transcribe` does **not** import `transformers` or `torch` (verified by a test that captures `sys.modules`).

#### Test Plan
- **Unit Tests**: protocol conformance, class-mapping correctness, mocked-detector flow, lazy-import assertion.
- **Integration Tests**: optional `@pytest.mark.slow` real-model test, off by default in PR CI.
- **Manual Testing**: PR author runs the slow test locally before merge; output is attached to the PR.

#### Rollback Strategy
- Adapter is isolated behind the `LayoutDetector` Protocol. Revert the PR; later phases will need stubs or noop detectors until phase 4 lands again.

#### Risks
- **Risk**: Chosen model is too large to load on a 16 GB-RAM CPU laptop alongside the OCR model from phase 5.
  - **Mitigation**: footprint is recorded in `models.toml`; the model + OCR combined memory budget is tracked; phase 5 may pick a smaller OCR model to fit.
- **Risk**: Model class set does not cover all `RegionRole` values; mapping has gaps.
  - **Mitigation**: unmapped classes fall through to `RegionRole.unknown` with a logged warning; phase 6 role classifier handles `unknown` paragraphs deterministically.

---

### Phase 5: Per-region OCR / VLM transcription adapter (Hugging Face)
**Dependencies**: Phase 4 (regions with bboxes; Region schema)

#### Objectives
- Implement the second ML step: take a `Region` (bbox only) on a page image and fill in `text`.
- Choose and pin the default Arabic-strong OCR/VLM model.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/ocr/__init__.py` — `OCRTranscriber` Protocol with `transcribe(region: Region, page_image: PILImage) -> Region` (returns the same region with `text` filled and `confidence` set). For `role == table` regions, the adapter walks `region.table_grid.rows[i].cells[j]`, OCRs each cell's `bbox`, and returns a new `Region` whose `table_grid` is fully populated (`TableCell.text` and `TableCell.confidence` filled). For `role == figure` regions, `text` stays empty; `caption` regions are OCR'd as plain paragraphs.
- [ ] `src/arabic_pdf_transcribe/ocr/hf_ocr.py` — concrete adapter wrapping the chosen Hugging Face Arabic OCR model. Selection between {Qari-OCR, AIN, Surya OCR, comparable} is benchmarked inside this phase on the `image-scan` fixture and one paragraph from each `digital-broken` fixture; choice is documented in the PR description and the module docstring with the criteria (CER on Arabic, diacritics handling, license, footprint).
- [ ] `models.toml` updated with the OCR-model entry.
- [ ] Confidence reporting: per-region confidence in `[0, 1]` if the model exposes it; `None` otherwise.
- [ ] Unit tests with a stub OCR (no real model) covering the `--strict` vs default error paths; one `@pytest.mark.slow` real-model test.

#### Implementation Details
- Region cropping: `_crop.py` extracts the bbox from the page image with a small configurable padding (default 4 px).
- The adapter declares its expected inputs (RGB PIL, normalised orientation) so callers convert once.
- OCR errors propagate as `OCRTranscriptionError`; the orchestrator (phase 8) maps these to per-page failures or `--strict` aborts.
- Decoding parameters (beam width, max-length, temperature) are surfaced in `OCRConfig` and recorded in the module docstring; defaults are deterministic (greedy or beam, no sampling).

#### Acceptance Criteria
- [ ] Stub-based unit tests cover happy path, error path, and confidence pass-through.
- [ ] `models.toml` license-audit passes.
- [ ] Lazy-import assertion still holds (heavy deps are not pulled in at package import).

#### Test Plan
- **Unit Tests**: Protocol conformance, stub-flow tests, error-path tests.
- **Integration Tests**: optional `@pytest.mark.slow` real-model test on the image-scan fixture.

#### Rollback Strategy
- Adapter isolated. Revert the PR; the package retains the native + layout phases without OCR.

#### Risks
- **Risk**: Best Arabic-OCR model has a non-permissive license.
  - **Mitigation**: the license audit will block. The fallback is Surya OCR (Apache-2.0). The PR author documents the comparison.
- **Risk**: OCR is the slowest stage; per-page time blows the CPU performance floor.
  - **Mitigation**: `--max-workers` parallelism on the orchestrator (phase 8); region-level batching where the model supports it; document GPU expectations in the README.

---

### Phase 6: RTL-aware reading-order reconstruction and role classification
**Dependencies**: Phase 2, Phase 4 (Regions exist from at least one branch)

#### Objectives
- Reorder a flat list of `Region`s on a page into logical reading order.
- Refine `RegionRole` assignments where layout signals + text signals together produce a better answer than the upstream model alone.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/order/reorder.py` — `reorder(regions: list[Region], page_width: float, page_height: float, *, rtl: bool = True) -> list[Region]`.
- [ ] `src/arabic_pdf_transcribe/order/_columns.py` — column detector using bbox histograms; supports 1, 2, and 3-column layouts.
- [ ] `src/arabic_pdf_transcribe/order/_rows.py` — row band assignment within a column.
- [ ] `src/arabic_pdf_transcribe/roles/classify.py` — heading-level inference using font-size histogram (native path) or relative region-height (ML path), capped at H3 with deeper levels collapsed to H3; **when the document has no font-size / region-height signal at all, every heading is emitted at H2** (matches spec's "Semantic output contract" rule); list-item detection from bullet/number prefixes (Arabic `-`, `•`, `–`, `*`, Arabic-Indic digits, Western digits + `.` / `)`); list-marker capture (`raw_marker` preserved on `Region.list_marker.raw_marker`); header/footer pruning by edge-band heuristic; **caption-figure linkage**: when a `caption` region appears within `≤ 0.05 × page_height` of a `figure` region's bottom edge and overlaps horizontally, both regions are assigned the same `Region.group_id`. Ungrouped captions stay as standalone `caption` paragraphs.
- [ ] Unit tests with curated synthetic region lists exercising 1/2/3-column RTL, mixed RTL/LTR within a region, captioned figure (caption beneath figure groups together), and footnote-band pruning.

#### Implementation Details
- Column detection works on bbox X-centres and X-edges; a column is a connected band in the X-projection histogram with a mass threshold.
- Within each column, regions are sorted top-to-bottom (Y-band first, then within band right-to-left when `rtl=True`).
- "Right-most first within a row band" is the operational definition from the spec; the row-band tolerance is `0.5 × median_region_height`.
- Role classifier is deterministic and pure; it modifies only `role`, `heading_level`, `list_marker`, and `group_id` — never `text`.
- Heading-level inference: bin the available size signal (native font sizes for native pages, region-height for ML pages) into 3 quantiles and map (largest quantile → H1, middle → H2, smallest → H3); when **no size signal exists** for a region (e.g. an ML page where the layout detector did not emit heights), the region's `heading_level` is set to **`2`** so the emitter renders it as `##` (H2). This is the single, consistent rule across this phase.

#### Acceptance Criteria
- [ ] Snapshot tests on synthetic 2-column and 3-column RTL pages match the expected ordered ID list.
- [ ] A page with no size signal at all produces all headings with `heading_level == 2` (i.e. all `##` in Markdown). Verified by a unit test on a synthetic region list.
- [ ] Header/footer regions in the top/bottom 5 % band are pruned by default; configurable.
- [ ] A figure-then-caption pair on the same page receives the same `group_id`; an ungrouped caption keeps `group_id=None`.

#### Test Plan
- **Unit Tests**: column detection, row banding, RTL ordering, role classification (each separately).
- **Integration Tests**: end-to-end reorder → ordered list snapshot for hand-built region lists representing each layout fixture.

#### Rollback Strategy
- Pure functions; revert the PR. Upstream extractors still produce regions in extraction order, which is sufficient for trivial single-column documents.

#### Risks
- **Risk**: Three-column Arabic pages with marginalia confuse the column detector.
  - **Mitigation**: marginalia detection is best-effort; documented limitation in v1; users can post-process if needed.
- **Risk**: Heading-level inference produces unstable levels across pages within the same document.
  - **Mitigation**: heading-level inference uses a document-wide font-size histogram (built across all native pages) when the document is uniformly native; ML-only documents use per-page height bands and accept some instability — documented.

---

### Phase 7: Markdown and Word (.docx) emitters
**Dependencies**: Phase 6

#### Objectives
- Convert an ordered, role-classified `Region` stream into Markdown and into Word.
- Preserve Unicode bidi correctness in both formats.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/emit/markdown.py` — `emit_markdown(regions: Iterable[Region]) -> str`. Mapping rules:
  - `heading` → `#` × `heading_level` (`heading_level` is always populated by phase 6, defaulting to 2).
  - `paragraph` → plain paragraph.
  - `list_item` → `- ` (bullet) or `1. ` (ordered, using `list_marker.ordinal`); consecutive list items render in the same list block.
  - `table` → Markdown pipe-table from `region.table_grid`; cell text is escape-safe; if `meta["v1_table_simplification"]=True` a leading `<!-- v1: merged cells flattened -->` HTML comment precedes the table.
  - `figure` → `![figure on page {page_index+1}](#)` (caption-less); when grouped with a `caption` via `group_id`, the caption text is used as the alt-text and the caption region is suppressed (it was emitted as part of the figure line).
  - `caption` (ungrouped) → italic paragraph (`*caption text*`).
  - `header_footer` → suppressed by default; surfaced only when the future `--include-running-heads` flag is set (defer flag to follow-up).
  - `failure_placeholder` → HTML comment `<!-- transcription-failed: page {N+1} reason: {failure_reason} -->`.
- [ ] `src/arabic_pdf_transcribe/emit/docx.py` — `emit_docx(regions: Iterable[Region], output_path: Path) -> None`. Uses `python-docx`; mapping:
  - `heading` → built-in style `Heading {heading_level}`.
  - `paragraph` → `Normal`.
  - `list_item` → `List Bullet` or `List Number`.
  - `table` → an actual Word table built from `region.table_grid`; one paragraph per cell; if v1-simplification flag is set, a comment paragraph precedes the table.
  - `figure` / `caption` → placeholder paragraph "Figure on page N" (image embedding deferred); grouped caption text is appended to the placeholder.
  - `failure_placeholder` → styled paragraph "Transcription failed (page N): {reason}" using the `Quote` built-in style.
- [ ] `src/arabic_pdf_transcribe/emit/_md_escape.py` — Markdown-safe escaping of region text (escape leading list markers, pipes inside table cells, raw HTML).
- [ ] `src/arabic_pdf_transcribe/emit/_bidi.py` — bidi helpers (insert U+200F where the spec's intra-region bidi-mixing test requires it; conservative — only for paragraphs whose dominant script is Arabic and that contain LTR runs).
- [ ] Unit tests: round-trip a hand-built region list to Markdown (byte-identical snapshot), to Word (open the produced `.docx`, walk paragraphs, assert styles and text), table emission, list emission, escape-safety of adversarial text (e.g. a region whose text starts with `# ` or contains `|`), bidi preservation tests using Unicode bidi class assertions.

#### Implementation Details
- Markdown is generated deterministically: stable ordering of attributes, no timestamps, no environment-dependent data.
- Word output uses `python-docx` with the default document template (no macros, no embedded code — explicit security guarantee per spec).
- Failure regions appear as Markdown HTML comments and as styled "Transcription failed (page N)" paragraphs in Word; the emitters render them from `Region(role=failure_placeholder, failure_reason=...)` (phase 8 owns synthesis; phase 7 owns rendering).
- Table rendering reads from `Region.table_grid`. The emitters never see partial / missing grids — phase 4's adapter and phase 6's role classifier guarantee `table_grid is not None` whenever `role == table`.

#### Acceptance Criteria
- [ ] Markdown snapshots are byte-identical across two runs.
- [ ] `.docx` output passes a structural test: open with `python-docx` and verify paragraph styles for each region role.
- [ ] No emitter call performs network I/O; verified by mocking `socket`.

#### Test Plan
- **Unit Tests**: per emitter, per role, with adversarial text inputs.
- **Integration Tests**: end-to-end region-list → Markdown / Word, tested in phase 9 against fixture PDFs.

#### Rollback Strategy
- Emitters are isolated behind module boundaries. Revert the PR; downstream CLI does not yet exist.

#### Risks
- **Risk**: Word styling differs across LibreOffice / Microsoft Word.
  - **Mitigation**: tests verify the underlying `python-docx` style attribute, not visual rendering. Limitation documented.
- **Risk**: Markdown bidi handling produces visually correct but byte-unstable output across runs.
  - **Mitigation**: bidi insertion is deterministic (rule-based); snapshot tests would catch instability.

---

### Phase 8: CLI and pipeline orchestrator with full failure-mode coverage
**Dependencies**: Phases 1–7 (production wiring requires the real adapters from phases 4 & 5; unit tests inject stubs that conform to the `LayoutDetector` and `OCRTranscriber` Protocols).

#### Objectives
- Wire native, validator, layout, OCR, reorder, and emitters into one pipeline.
- Implement the full CLI surface from the spec's "Resolved Decisions" section: format selection, exit codes, `--strict`, `--quiet`, `--json-logs`, progress reporting, encrypted/corrupted PDF handling, OOM/offline-cache handling.

#### Deliverables
- [ ] `src/arabic_pdf_transcribe/pipeline.py` — `transcribe(pdf_path, *, layout_detector, ocr_transcriber, validator, validator_config, markdown_emitter, docx_emitter, max_workers, ...) -> TranscribeResult` returning per-page outcomes plus the ordered region stream. **All four pluggable hooks** (`validator`, `layout_detector`, `ocr_transcriber`, `markdown_emitter` / `docx_emitter`) accept any object conforming to the relevant Protocol (`Validator`, `LayoutDetector`, `OCRTranscriber`, `MarkdownEmitter`, `DocxEmitter`); the package-level defaults are wired when the kwarg is omitted. This satisfies the spec's "library API exposes hooks for swapping the layout detector, the OCR model, the validator, and the emitter" requirement.
- [ ] `src/arabic_pdf_transcribe/cli.py` — `argparse`-based CLI implementing `arabic-pdf-transcribe FILE [-o PATH] [--format md|docx] [--pages RANGES] [--strict] [--quiet] [--json-logs] [--config PATH] [--debug-json PATH] [--max-workers N]`. Detail:
  - `--pages RANGES`: comma-separated ranges, e.g. `1-3,7,10-12`. Pages outside the range are skipped *before* extraction (no work done on them).
  - `--config PATH`: TOML file. Schema (documented in README): `[validator]` (overrides for `ValidatorConfig`), `[layout]` (HF model override + revision pin), `[ocr]` (HF model override + revision + decoding params), `[render]` (rasterisation DPI). The same config keys are also accepted as env vars (`AR_PDF_*`) for headless installs.
  - `--max-workers N`: per-page parallelism for the ML branch. Default `1` on CPU (avoid OOM); `--max-workers auto` selects `min(cpu_count, 4)` on CPU and `1` on GPU (GPU contention worse than CPU).
- [ ] `pyproject.toml` console-script entry: `arabic-pdf-transcribe = arabic_pdf_transcribe.cli:main`.
- [ ] **`errors.py` is extended** in this phase (created in phase 1) with the rest of the exception types: `EncryptedPDFError`, `CorruptedPDFError`, `OutOfMemoryDuringInference`, `FormatExtensionMismatch` (the phase-1 file already exposes `ModelDownloadError`, `OCRTranscriptionError`).
- [ ] `src/arabic_pdf_transcribe/_logging.py` — stderr progress (page X of N) by default; `--quiet` suppresses; `--json-logs` emits structured per-page log lines.
- [ ] Failure-region synthesis: when a page fails, an HTML-comment placeholder Markdown region or a styled Word paragraph is inserted in the output stream (consumed by phase 7's emitters).
- [ ] Process-scoped `tempfile.TemporaryDirectory` lifecycle: created on pipeline entry, removed on exit (success or failure).
- [ ] Unit tests for CLI argument parsing, format-selection logic, exit-code paths (each of `0/2/3/4/5`), and failure synthesis.

#### Implementation Details
- The pipeline is iterator-shaped: pages are processed one at a time, with progress events emitted per page, so a 500-page document does not hold all regions in memory.
- Format selection logic per spec: extension wins when `-o PATH` is given with a recognised extension; `--format` wins when no extension or stdout. Disagreement → `FormatExtensionMismatch` → exit 4.
- `--strict` is a single bool flag; default is best-effort.
- OOM handling: catch `MemoryError` and `torch.cuda.OutOfMemoryError`; map to per-page failure (or `--strict` abort).
- Encrypted-PDF detection happens before any extraction (`pypdfium2` exposes the bit); abort with exit 3.
- The orchestrator accepts `LayoutDetector` and `OCRTranscriber` instances by Protocol; tests inject stubs. Production wiring uses the HF adapters from phases 4 & 5.

#### Acceptance Criteria
- [ ] CLI produces the expected exit code for each failure scenario in spec test scenarios 11–18.
- [ ] Format-selection conflicts produce a clean error, not a crash.
- [ ] Long-running runs emit per-page progress.
- [ ] `--debug-json` produces a valid JSON sidecar containing per-region confidences.
- [ ] No leaked temp files after either success or failure (test asserts directory removal).

#### Test Plan
- **Unit Tests**: CLI parsing, format-selection logic, exit-code mapping, failure-region synthesis, temp-dir cleanup (verified via mock `TemporaryDirectory` or post-hoc assertion).
- **Integration Tests**: end-to-end via stub layout/OCR adapters on the in-tree fixtures (no real model load) — covers all native-path test scenarios, mixed PDFs (with the validator forcing some pages into the stub-OCR branch), and the failure scenarios.
- **Manual Testing**: run on the image-scan fixture with the real ML adapters once before PR merge.

#### Rollback Strategy
- The CLI and pipeline modules are isolated. Revert the PR; library API up to phase 7 still works.

#### Risks
- **Risk**: Iterator-shaped pipeline breaks composability with the emitters which currently expect `Iterable[Region]` for the whole document.
  - **Mitigation**: emitters from phase 7 are written to accept any `Iterable`; the orchestrator yields regions lazily so memory stays bounded.
- **Risk**: argparse is too rigid for the format/extension conflict logic.
  - **Mitigation**: validation happens after `parse_args`, in a separate `validate_args(ns) -> ValidatedArgs` function that is fully unit-tested.

---

### Phase 9: Benchmark corpus, end-to-end tests, license-audit CI, and documentation
**Dependencies**: Phases 1–8

#### Objectives
- Lock in the project's quality contract by populating the spec's benchmark corpus, running real end-to-end tests against it, validating the license-audit chain in CI, and writing the user-facing documentation.

#### Deliverables
- [ ] Corpus populated under `tests/fixtures/pdfs/` per spec — `digital-clean/` (≥ 3), `digital-broken/` (≥ 2), `image-scan/` (≥ 3), `mixed/` (≥ 1), `edge/` (empty, password-protected, truncated, A2-poster, intra-region bidi). Each fixture has a sibling `*.expected.md` reference. License recorded for each in `tests/fixtures/pdfs/LICENSES.md`.
- [ ] `tests/test_e2e.py` — end-to-end tests:
  - native-path fixtures: byte-identical Markdown snapshots
  - ML-path fixtures: CER ≤ 0.05 against `*.expected.md` (matches the spec's reproducibility tolerance; CER computed by a small in-tree implementation that matches WER/CER convention)
  - mixed PDFs: per-page outcome assertions
  - edge fixtures: exit-code assertions
- [ ] `.github/workflows/ci.yml` updated: `lint`, `test` (without `slow`), `audit`. A separate `nightly.yml` runs `slow` tests with cached models.
- [ ] Performance smoke (not a hard CI gate): `tests/test_perf_smoke.py` runs the native-only path on the digital-clean corpus and asserts wall-clock under the 5-second floor with a generous slack; ML-path perf is documented in the README, not gated in CI.
- [ ] `README.md` finalised: install, usage (CLI + library), exit-code table, model-card section listing each model + revision + license + footprint, architecture diagram (Mermaid), benchmark-corpus description, security notes, contributing.
- [ ] `docs/architecture.md` — pipeline diagram + module-by-module description.
- [ ] `docs/model-card.md` — per-model card following the format from each upstream HF page.
- [ ] `CHANGELOG.md` — `0.1.0` entry summarising the work.

#### Implementation Details
- Some fixtures must be either (a) MIT-licensable / public-domain Arabic PDFs from a well-attested source, or (b) hand-built by the maintainer specifically for this corpus. Hand-built fixtures are preferred for the `digital-broken` and `edge` cases because they're the hardest to source legally.
- The CER implementation is a tiny pure-Python Levenshtein on Unicode codepoints — no `editdistance` dependency, to keep the dep tree small and license-clean.

#### Acceptance Criteria
- [ ] All e2e tests pass on the corpus.
- [ ] License audit passes in CI on every PR.
- [ ] `arabic-pdf-transcribe --help` matches the documented surface.
- [ ] README's architecture diagram renders on GitHub.

#### Test Plan
- **Unit Tests**: CER computation correctness (parametrised, including edge cases — empty strings, identical strings, full-substitution).
- **Integration / E2E Tests**: full corpus run, per-flavour assertions.
- **Manual Testing**: `pip install dist/*.whl` in a fresh venv; run CLI on a real PDF; verify Markdown + Word output by eye on at least one document.

#### Rollback Strategy
- Documentation is purely additive; corpus + e2e tests can be relaxed in a follow-up if time pressure demands it (defaulting to a smaller subset). The phase is the natural release-cut moment, so rollback means cutting a smaller release.

#### Risks
- **Risk**: Sourcing licence-clean Arabic PDFs is slower than expected.
  - **Mitigation**: synthesise more fixtures via a small generator script (LaTeX → PDF) for `digital-clean` and `digital-broken`; image-scan corpus uses public-domain Wikisource Arabic scans where available, otherwise hand-scanned items released by the maintainer under MIT.
- **Risk**: Real-model E2E tests are flaky in CI.
  - **Mitigation**: real-model tests run in `nightly.yml`, not on every PR; PR CI uses stubs and snapshot fixtures.

## Dependency Map

```
                         ┌──────────────────────────┐
                         │ Phase 1: Skeleton + audit │
                         └────────────┬──────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │ Phase 2: Region + native      │
                         └────────────┬──────────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                                            ▼
┌──────────────────────────────┐              ┌──────────────────────────────┐
│ Phase 3: Validator            │              │ Phase 4: Layout adapter       │
└──────────────┬────────────────┘              └──────────────┬────────────────┘
               │                                              │
               │                                              ▼
               │                                ┌──────────────────────────────┐
               │                                │ Phase 5: OCR adapter          │
               │                                └──────────────┬────────────────┘
               │                                               │
               └───────────────────┬───────────────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ Phase 6: Reorder + role classification│
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ Phase 7: Emitters (Markdown + Word)   │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ Phase 8: CLI + orchestrator           │
                └──────────────────┬───────────────────┘
                                   ▼
                ┌──────────────────────────────────────┐
                │ Phase 9: Corpus + e2e + docs          │
                └──────────────────────────────────────┘
```

Phases 4 and 5 can proceed in parallel with phase 3 once phase 2 is merged (they only need the `Region` schema, not the validator). Phase 6 needs both the native source (phase 2) and at least the layout source (phase 4) to be exercised meaningfully; in practice it's authored after phase 5 to test against the full ML path.

## Resource Requirements

### Development Resources
- **Engineers**: One Python developer comfortable with Hugging Face, PDF tooling, and Arabic Unicode (NFC/NFKC, bidi, glyph classes). Solo maintainer for v1.
- **Environment**: Linux (Ubuntu 22.04+), Python 3.10/3.11, optional NVIDIA GPU with CUDA 12.x for the GPU performance floor; CPU-only path supported.

### Infrastructure
- **CI**: GitHub Actions, `ubuntu-latest`, no self-hosted runners.
- **Caches**: Hugging Face model cache directory in CI (`actions/cache`) for the `nightly.yml` workflow only.
- **Artifacts**: built wheel + sdist published as a CI artifact on `main`.

## Integration Points

### External Systems
- **Hugging Face Hub** — first-time model download.
  - **Integration Type**: HTTPS to `huggingface.co`.
  - **Phase**: 4, 5 (download); offline thereafter.
  - **Fallback**: clean error with instructions to run `huggingface-cli download <pinned-revision>` (CLI exit 5).

### Internal Systems
- None. This is a greenfield project.

## Risk Analysis

### Technical Risks
| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| Validator misclassifies broken native text → silent bad output | Medium | High | Three-signal rule + corpus tests; conservative default ("any signal flags → ML branch") | Builder |
| Reading-order wrong on multi-column RTL pages | Medium | High | Snapshot tests on hand-built layouts; documented v1 limitation on marginalia | Builder |
| Best Arabic OCR has non-permissive license | Medium | Medium | Picked second-best with permissive license; license-audit CI catches regressions | Builder |
| ML-path too slow on CPU for 10-page floor | Medium | Medium | Smaller default model on CPU; per-page parallelism; documented GPU expectations | Builder |
| Untrusted PDF triggers parser bug | Low | Medium | `pypdfium2` safe defaults; fuzz fixtures in `edge/`; no JS / external resource resolution | Builder |
| HF Hub unavailable at install time | Low | Low | Cache-first design; documented offline workflow; `nightly.yml` caches models | Builder |
| Snapshot tests too brittle across model versions | Medium | Low | Native-path snapshots are byte-identical; ML-path uses CER tolerance | Builder |
| Phase 4/5 model selection delays the PR | Medium | Low | Selection criteria are explicit; benchmarking on tiny fixtures keeps the loop fast | Builder |

### Schedule Risks
*Per the SPIR protocol's "no time estimates" rule, schedule risk is expressed as ordering / dependency risk only, not calendar risk.*

| Risk | Probability | Impact | Mitigation | Owner |
|------|-------------|--------|------------|-------|
| Phase 6 (reorder) blocks phases 7–8 if it under-delivers on three-column layouts | Medium | Medium | Phase 6's acceptance criteria are explicit about 1/2-column pass + 3-column "best effort"; 3-column improvements can land as a follow-up | Builder |
| Phase 9 corpus sourcing slips ahead of phase 8 | Low | Low | Begin sourcing fixtures during phase 1; add fixtures as each phase needs them | Builder |

## Validation Checkpoints
1. **After Phase 1**: skeleton imports cleanly; license audit harness fails-closed on a fabricated bad license.
2. **After Phase 3**: a clean Arabic PDF round-trips native → validator-accept → text on a `digital-clean` fixture (no ML loaded).
3. **After Phase 5**: the ML branch produces non-empty regions with reasonable text on the `image-scan` fixture (real-model nightly test).
4. **After Phase 6**: snapshot tests for 1/2/3-column RTL pages match expected order.
5. **After Phase 7**: Markdown + Word emit a hand-built region list correctly with adversarial text.
6. **After Phase 8**: CLI exit-code matrix matches spec for all edge fixtures.
7. **Before final approval**: full corpus E2E run is green; license audit green; README + model card complete.

## Monitoring and Observability

This is a CLI/library, not a service — observability is local-only.

### Metrics to Track (per-run, on stderr in `--json-logs` mode)
- `pages_total`, `pages_native_path`, `pages_ml_path`, `pages_failed`.
- Per-page wall-clock time for each stage.
- Validator signal values per page (when `--debug-json` is set).
- Region counts per role.

### Logging Requirements
- Default: human-readable progress on stderr (`page X of N`).
- `--quiet`: stderr silent.
- `--json-logs`: one JSON event per page on stderr; schema documented in the README.

### Alerting
- Not applicable for a CLI/library.

## Documentation Updates Required
- [ ] `README.md` — install, usage (CLI + library), exit codes, model card, architecture diagram.
- [ ] `docs/architecture.md` — pipeline + module structure.
- [ ] `docs/model-card.md` — per-model details.
- [ ] `CHANGELOG.md` — `0.1.0`.
- [ ] `tests/fixtures/pdfs/LICENSES.md` — per-fixture provenance and license.
- [ ] CLI exit-code table — in `README.md` and in the CLI's `--help`.

## Post-Implementation Tasks
- [ ] Performance validation on the documented hardware floors (CPU and GPU).
- [ ] Manual security review of `pypdfium2` configuration and emitter escape paths.
- [ ] User-acceptance test: maintainer runs the tool on a non-fixture, real-world Arabic PDF and inspects output.
- [ ] Confirm the `models.toml` allow-list is enforced in CI.

## Expert Review

**Date**: 2026-05-01
**Models Consulted**: GPT-5 Codex, Claude. Gemini Pro was unavailable (upstream HTTP 429 quota exhaustion, same condition as the spec phase); architect-approved 2/3 acceptance under `[ARCHITECT INSTRUCTION | 2026-05-01T18:32:24.195Z]`. Gemini will be retried once after this revision; if still blocked, the consultation log records the skip.

### Iteration 1 — feedback summary

**Codex (REQUEST_CHANGES)** — addressed:
- Region schema thinness: enriched with `heading_level`, `list_marker`, `table_grid` (rows × cells), `group_id` (caption-figure linkage), `failure_placeholder` role + `failure_reason`, plus `REGION_SCHEMA_VERSION`. Phase 2 deliverable rewritten.
- Table support end-to-end: native-table detection added to phase 2 (`extract/_native_tables.py`); ML-branch table-cell detection added to phase 4 (`layout/_table_cells.py`); per-cell OCR added to phase 5; emitters consume `table_grid` directly (phase 7).
- Phase 8 dependency inconsistency: `Dependencies` now reads "Phases 1–7" with explicit note that production wiring uses real adapters and unit tests use stubs.
- Validator + emitter hooks: pipeline now exposes `validator`, `markdown_emitter`, `docx_emitter` kwargs alongside `layout_detector` / `ocr_transcriber`. Spec's "library API hooks" requirement is met explicitly.
- Phase 6 H2/H3 contradiction: collapsed to a single rule — when no size signal exists, `heading_level == 2` everywhere. Acceptance criterion and implementation text now agree.

**Claude (COMMENT)** — addressed:
- `errors.py` timing: created in phase 1 (with `ModelDownloadError` and `OCRTranscriptionError`), extended in phase 8 with the rest. No more forward references.
- Phase 6 heading-level rule: same fix as above.
- CER tolerance: phase 9 e2e now uses ≤ 0.05 to match the spec.
- Mixed Arabic/Latin fixture: added to phase 2 (`lorem-ar-en-mixed.pdf`); spec test scenario 2 covered from phase 2 onward.
- Dependency diagram: redrawn to show phase 4 depending on phase 2, not phase 1.
- `--max-workers` and `--config`: spelled out in phase 8 deliverables, including default behaviour and config-file schema.
- Spec open questions (target Arabic varieties, handwriting, tables, figures, etc.): closed in the new "Plan-Level Assumptions" section at the top of this plan.

**Gemini Pro** — unavailable; retry attempt to follow this update.

### Plan Adjustments
- See edits above (search the file's git diff between commits `[Spec 1] Initial implementation plan` and `[Spec 1] Plan with multi-agent review` for the full set).

## Approval
- [ ] Technical Lead Review
- [ ] Engineering Manager Approval
- [ ] Resource Allocation Confirmed
- [ ] Expert AI Consultation Complete

## Change Log
| Date | Change | Reason | Author |
|------|--------|--------|--------|
| 2026-05-01 | Initial plan draft | New project; spec approved. | builder |

## Notes
- The plan deliberately keeps the **deterministic core (phases 1–3)** independent of any ML dependency. This means the project can ship a minimal release that handles clean digital Arabic PDFs without any HF download — useful as a fallback distribution.
- Phase 4 / phase 5 carry an embedded **research step** for model selection. This is allowed by the SPIR protocol — the spec deliberately deferred model choice to plan/research time. The selection criteria and the chosen revisions land in `models.toml` and in the relevant module docstrings as a permanent record.
- Phase 9 is the natural release-cut moment. After phase 9, the project enters **verify** per the SPIR builder role.
