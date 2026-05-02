# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-02

Initial release. Native-first Arabic PDF transcription pipeline with ML
fallback, RTL-aware reading order, and Markdown / Word output. Built across
nine SPIR phases over PRs #2 - #9.

### Added

- **Native text extraction** (`extract_native`) on top of `pypdfium2` with
  font-size histogram capture and `has_text_layer` short-circuit. Page
  filter pushed down so non-selected pages do zero work.
- **Multi-signal validator** (`validate_page`): Arabic-codepoint ratio,
  replacement-glyph ratio, and word-boundary KL divergence against a
  bundled Arabic reference distribution. Any signal flag → ML branch.
- **Layout-detection adapter** (`HFDiTLayoutDetector`) backed by
  `cmarkea/dit-base-layout-detection` (Apache-2.0). Lazy `transformers`
  / `torch` imports.
- **Per-region OCR adapter** (`HFGotOCRTranscriber`) backed by
  `stepfun-ai/GOT-OCR-2.0-hf` (Apache-2.0). Greedy decoding by default
  for reproducibility.
- **RTL-aware reading-order reconstruction** (`reorder`) — column
  detection, row banding, within-band right-to-left ordering.
- **Role classification** (`classify_page`) — heading-level inference,
  list-marker detection, header/footer pruning, caption-figure linkage.
- **Markdown emitter** (`emit_markdown`) — escape-safe, GFM-compatible,
  byte-identical across runs. NFC default; NFKC opt-in for Arabic
  presentation-form normalisation.
- **Word emitter** (`emit_docx`) — `python-docx` with built-in styles;
  byte-identical `word/document.xml` across runs.
- **Pipeline orchestrator** (`pipeline.transcribe`) — one shared
  `pypdfium2` document handle across native extraction, validation, and
  ML rasterisation. Best-effort default with `--strict` opt-in. CUDA
  OOM detection via class metadata (no `torch` import).
- **CLI** (`arabic-pdf-transcribe`) — `--format`, `--pages`, `--strict`,
  `--quiet`, `--json-logs`, `--config`, `--debug-json`, `--max-workers`,
  `--dpi`. Full TOML config schema (`[validator] [layout] [ocr]
  [render]`).
- **License-audit harness** (`tools/license_audit.py`) covering the
  runtime dep tree and the pinned model registry (`models.toml`).
- **Benchmark corpus** under `tests/fixtures/pdfs/` with sibling
  `*.expected.md` reference files for native-path snapshot tests.
- **CER implementation** (`tests/_cer.py`) — pure-Python Wagner-Fischer
  edit distance for ML-path acceptance tests.
- **CI** (`.github/workflows/ci.yml`) — lint + license audit + tests on
  Python 3.11/3.12. Nightly slow workflow (`nightly.yml`) with cached HF
  weights.

### Documentation

- `README.md` — install, CLI, library, exit codes, model summary,
  Mermaid architecture diagram, security notes.
- `docs/architecture.md` — pipeline + module-by-module description.
- `docs/model-card.md` — per-model card including selection rationale.
- `tests/fixtures/pdfs/LICENSES.md` — per-fixture provenance.

### Security

- `pypdfium2` opened with no JavaScript execution and no external
  resource resolution.
- All model weights pinned by revision hash, not floating tag.
- Markdown / Word emitters escape adversarial text (leading list
  markers, pipes inside table cells, raw HTML).
- No telemetry; no automatic phone-home after first cache fill.

### Known limitations / deferred

- DPI-tuning, `min_arabic_ratio` threshold tuning, and a real Arabic
  benchmark corpus are post-v1 follow-ups (the in-tree corpus is
  reproducibility-focused, not realism-focused).
- Distinct `RegionRole.FOOTNOTE` (currently collapsed into
  `HEADER_FOOTER`) is deferred — schema change with downstream emitter
  impact.
- Pure-Python morphology in `layout/_table_cells.py` is acceptable for
  v1; vectorised numpy / `scipy.ndimage.label` upgrade is a perf
  follow-up.
- `--max-workers > 1` is plumbed through but the v1 loop is sequential
  to bound peak RSS; parallel ML runs land post-v1 after benchmarking.
