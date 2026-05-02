# Architecture

## Subsystems

| Subsystem | Module path | Boundary |
|---|---|---|
| **PDF loader** | `arabic_pdf_transcribe.pdf._pypdfium2_loader` | Single entry point `open_pdf(path)`; translates `pypdfium2` errors to `EncryptedPDFError` / `CorruptedPDFError`. JS execution and external resource resolution disabled. |
| **Native extraction** | `arabic_pdf_transcribe.extract.native` | `extract_native(pdf_path, *, pages=None) -> Iterator[NativePage]`. Streaming + page-filter aware so excluded pages do zero work. `extract_native_from_document(document, *, pages=None)` is the document-handle variant the orchestrator uses for shared-handle calls. |
| **Validator** | `arabic_pdf_transcribe.validate.native_validator` | `validate_page(page, *, config) -> ValidationResult`. Three signals (Arabic ratio, replacement-glyph ratio, word-boundary KL). Any signal flag → ML branch. |
| **Layout adapter** | `arabic_pdf_transcribe.layout` | `LayoutDetector` Protocol + `HFDiTLayoutDetector` (Apache-2.0). Lazy `transformers` / `torch` imports. |
| **OCR adapter** | `arabic_pdf_transcribe.ocr` | `OCRTranscriber` Protocol + `HFGotOCRTranscriber` (Apache-2.0). Per-region transcription. Deterministic decoding by default. |
| **Reading order** | `arabic_pdf_transcribe.order.reorder` | RTL-aware: column detect → row band → within-band right-to-left. |
| **Role classifier** | `arabic_pdf_transcribe.roles.classify` | Heading levels, list markers, header/footer prune, caption-figure linkage. |
| **Emitters** | `arabic_pdf_transcribe.emit.{markdown,docx,_normalise,_md_escape,_bidi}` | `emit_markdown(regions) -> str` and `emit_docx(regions, path) -> None`. NFC default; NFKC opt-in. RLM injection only for Arabic-dominant paragraphs containing LTR runs. python-docx imported lazily by package proxy. |
| **Orchestrator** | `arabic_pdf_transcribe.pipeline` | `transcribe(pdf_path, *, layout_detector, ocr_transcriber, validator, validator_config, classify_config, reorder_fn, rtl, dpi, pages, strict, max_workers, progress) -> TranscribeResult`. Single shared `pypdfium2` document across the run. |
| **CLI** | `arabic_pdf_transcribe.cli` | argparse + `validate_args()` + exit-code mapping (0/2/3/4/5). |
| **Logging** | `arabic_pdf_transcribe._logging` | Per-page progress events; TEXT / QUIET / JSON modes; no timestamps in JSON for byte-stable stderr. |
| **Errors** | `arabic_pdf_transcribe.errors` | Typed exceptions mapped to CLI exit codes. |
| **License audit** | `tools/license_audit.py` + `models.toml` | Allow-list-driven license check on runtime deps and pinned model registry. |

## Data flow

```
Input PDF
  ↓
pdf/_pypdfium2_loader.open_pdf  (encrypted / corrupted boundary)
  ↓
extract/native.extract_native_from_document  (pages filter)
  ↓                                                 ↓ (rejected)
validate/native_validator.validate_page             layout/_rasterise.rasterise_page
  ↓ (accepted)                                      ↓
order/reorder.reorder                               layout/hf_detector.HFDiTLayoutDetector.detect
  ↓                                                 ↓
roles/classify.classify_page                        ocr/hf_ocr.HFGotOCRTranscriber.transcribe
  ↓                                                 ↓
                  pipeline.transcribe (merge per page)
                                ↓
                  emit/markdown.emit_markdown    emit/docx.emit_docx
                                ↓                       ↓
                           output.md            output.docx
```

## Cross-stage contract

The unified `Region` dataclass (`arabic_pdf_transcribe.regions`) carries every
piece of information across stages — bbox, role, source (NATIVE / OCR),
text, optional heading level, list marker, table grid, group id (for
caption-figure linkage), failure reason, and a frozen `meta` mapping for
flags like `v1_table_simplification`. The schema is versioned
(`REGION_SCHEMA_VERSION = "1"`) and JSON-serialisable for the
`--debug-json` sidecar.

## Invariants

- **Native-first short-circuit**: clean digital Arabic PDFs produce output
  with no model load. Verified by lazy-import regression tests in every
  ML-touching module.
- **Single document handle**: the orchestrator opens the PDF once and
  shares the handle across page-count, native extraction, and ML
  rasterisation. Rasterisation no longer reopens the file per page.
- **Failure-region source attribution**: the failure-placeholder region's
  `RegionSource` reflects the branch where the exception happened
  (NATIVE for validator/extraction failures; OCR for ML failures).
- **CUDA OOM detection without `torch` import**: `pipeline._is_cuda_oom`
  inspects the exception class's `__module__` / `__name__` so the
  orchestrator stays light when `[ml]` is not installed.
- **License compatibility enforced**: `tools/license_audit.py` runs in
  CI on every PR and asserts every runtime dep + every entry in
  `models.toml` carries an allow-listed license (MIT, Apache-2.0, BSD,
  ISC, MPL-2.0, PSF-2.0, Unlicense, 0BSD).
- **Determinism**: native path produces byte-identical Markdown across
  runs; ML path is reproducible up to model floating-point determinism
  on CPU (CER ≤ 0.05 acceptance contract).
- **Per-page best-effort**: a single page that fails synthesises a
  `FAILURE_PLACEHOLDER` region; the rest of the document still
  transcribes. `--strict` re-raises instead.

## Configuration

The CLI accepts a TOML config with four sections:

- `[validator]` — overrides for `ValidatorConfig` (Arabic ratio, replacement
  glyph ratio, word-boundary KL threshold).
- `[layout]` — overrides for `HFLayoutDetectorConfig` (model id, revision,
  pixel confidence, min region area).
- `[ocr]` — overrides for `OCRConfig` (model id, revision, decoding params).
- `[render]` — overrides for rasterisation (DPI). The `--dpi` CLI flag wins
  over `[render].dpi` when both are given.

## Phase 9 (release-cut) carry-overs

- ML-path CER reference texts for `image-scan/` and `mixed/` fixtures are a
  post-v1 follow-up (require running the real HF models on the in-tree
  fixtures and curating references).
- Real Arabic-content benchmark corpus (InDesign / LaTeX-bidi / Wikisource
  scans with documented licenses) is a post-v1 follow-up.
- Threshold-tuning carry-over from phase 3 (`min_arabic_ratio`,
  presentation-form handling) lands together with the corpus-realism
  follow-up so the data drives the thresholds.
- Distinct `RegionRole.FOOTNOTE` (currently collapsed into `HEADER_FOOTER`)
  is a deferred schema change with downstream emitter impact.
- Pure-Python morphology in `layout/_table_cells.py` is acceptable for v1;
  vectorised numpy / `scipy.ndimage.label` upgrade is a perf follow-up.
- `--max-workers > 1` is plumbed but the v1 loop is sequential to bound
  peak RSS; parallel ML runs land post-v1 after benchmarking.
