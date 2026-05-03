# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.7] - 2026-05-03

### Fixed

- **OCR produced 73 KB of garbage tokens despite ok=7 failed=0 (issue #24)**:
  the GOT-OCR-2.0 adapter called `model.generate(...)` without
  `tokenizer=processor.tokenizer` and `stop_strings="<|im_end|>"`. The
  model's chat template never emits a built-in EOS at the natural end
  of OCR output, so generation ran to `max_new_tokens` and the trailing
  tokens were sampled noise — math symbols, CJK ideographs, control
  bytes, runs of `■`/`●`. Now the kwargs are forwarded on every
  `generate` path (simple + CUDA-OOM CPU-fallback). The confidence
  aggregator is wrapped in try/except since `stop_strings` truncation
  can misalign the scores tuple on some transformers versions —
  degrades to `confidence=None` rather than raising.

## [0.1.6] - 2026-05-03

### Fixed

- **fp16/bf16 path on non-Ampere CUDA (issue #22)**: v0.1.5 loaded the
  layout + OCR models in fp16 on Turing GPUs (compute capability 7.5,
  e.g. GTX 1660 Ti) but the HF image processor still produced fp32
  `pixel_values`, causing every page to fail with `RuntimeError: Input
  type (float) and bias type (c10::Half) should be the same`. The
  shared `move_inputs_to_device` helper now optionally casts every
  floating-point tensor (`pixel_values` and friends) to the model
  dtype while leaving integer tensors (`input_ids`, `attention_mask`)
  alone. Both adapters store the resolved `torch_dtype` and pass it
  to the helper on every input-prep call, including CPU-fallback
  paths. Regression test exercises the path with real torch tensors
  + fp16 stub model.

## [0.1.5] - 2026-05-03

### Fixed

- **GPU OOM on 6 GB cards (issue #20)**: layout (DiT-base) and OCR
  (GOT-OCR-2.0) co-resident in fp32 exceeded VRAM on a GTX 1660 Ti,
  forcing a permanent CPU fallback for the entire run after a single
  `CUDA OOM during generate` on page 2. Four fixes:
  - **fp16 / bf16 by default on CUDA**. New `dtype` field on both
    adapter configs, propagated as `torch_dtype=` to `from_pretrained`.
    `auto` selects bf16 on Ampere+, fp16 on older CUDA, fp32 on CPU.
  - **Layout model evicts to CPU between pages on CUDA** (default-on
    via `HFLayoutDetectorConfig.evict_after_inference`). OCR no longer
    competes with layout for VRAM during decode.
  - **Single OOM retries once on GPU** before falling back to CPU
    permanently — handles transient KV-cache pressure without losing
    GPU acceleration for the whole document.
  - **`OCRConfig.max_new_tokens` lowered 1024 → 512** to cap KV-cache
    growth (still ~5× the 99th-percentile paragraph).

### Added

- **`--dtype {auto,float32,float16,bfloat16}` CLI flag and
  `[runtime].dtype` / per-section `[layout].dtype`, `[ocr].dtype`
  TOML entries (issue #20)**: precedence chain CLI > `[runtime]` >
  per-section > `auto`.

## [0.1.4] - 2026-05-03

### Fixed

- **ML inference no longer hangs on CPU when CUDA is available (issue #18)**:
  the OCR + layout adapters never moved their model or processed inputs to
  the GPU; CUDA-equipped hosts ran every region on CPU and pages with many
  regions appeared to hang for 30-60 minutes. New
  `arabic_pdf_transcribe._device.resolve_device("auto"|"cuda"|"cpu")` is
  shared by both adapters; both call `.to(device)` and switch to inference
  mode after `from_pretrained`, and move processor outputs via
  `BatchEncoding.to(device)`. CUDA OOM mid-run logs a one-line warning and
  downgrades the adapter to CPU for the remainder of the document.
- **Per-region progress logging (issue #18)**: pipeline now emits a
  `region` event before each OCR call (`{event:"region",region:i,of_regions:R,role:"PARAGRAPH"}`
  under `--json-logs`; text-mode equivalent in default output) plus a
  `layout` event before the layout-detect call. Long ML pages no longer
  appear frozen.
- **Bounded OCR decoding (issue #18)**: `OCRConfig.max_new_tokens` lowered
  from 4096 to 1024 (still > 3× a typical Arabic paragraph). Added
  `no_repeat_ngram_size=3` and `repetition_penalty=1.05` defaults to
  prevent the Qwen2 head from looping on adversarial Arabic crops.

### Added

- **`--device {auto,cuda,cpu}` CLI flag and `[runtime].device` TOML section
  (issue #18)**: precedence chain CLI > `[runtime]` > per-section
  `[layout]/[ocr].device` > `"auto"`.

## [0.1.3] - 2026-05-02

### Fixed

- **`transformers==4.46.3` predates GOT-OCR-2.0 (issue #16)**: the pinned
  version did not register the `got_ocr2` model type (added in
  transformers 4.49.0). Every ML-branch page failed with
  `ModelDownloadError` despite the model being correctly cached. Bumped
  pin to `transformers==4.49.0`. Added a CI smoke test that
  `MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES` registers `got_ocr2` —
  catches future regressions on day one.
- **DiT layout label `Formula` no longer drops Arabic body text (issue #16)**:
  `cmarkea/dit-base-layout-detection` was trained on English research
  papers and frequently mislabels justified Arabic body text as
  `Formula` (39 such regions on a single page in the bug report).
  Remapped `Formula → RegionRole.PARAGRAPH` so body text reaches the
  emitter and the spurious "label X mapped to UNKNOWN" warnings are
  silenced.
- **JSON-log failure events now include the exception message (issue #16)**:
  the `reason` field for typed pipeline errors was previously the
  exception class name only, hiding the actionable hint baked into
  the exception (e.g. the `--prefetch-models` recovery instruction
  on `ModelDownloadError`). Now `f"{type(exc).__name__}:{exc}"`.

## [0.1.2] - 2026-05-02

### Fixed

- **Validator false-accept on broken text layers (issue #14)**: real
  Arabic PDFs (Foulabook-class) often serialize font-specific glyph IDs
  as raw ASCII control codepoints (\x01..\x1F). The original three
  signals all abstained on this failure mode (no Arabic letters → arabic
  gate skipped; control bytes were treated as ASCII punctuation →
  replacement gate abstained; KL stayed under threshold). The
  `replacement_glyph_ratio` signal now counts ASCII control bytes
  (excluding the standard whitespace controls) as replacement glyphs,
  so such pages route to the ML branch.
- **Validator: visual-order / presentation-form text layers**: a 4th
  signal `presentation_form_ratio` flags pages whose Arabic body is
  overwhelmingly shaped glyphs (FB50–FDFF, FE70–FEFF) rather than
  logical-order base codepoints. Threshold tunable via
  `[validator].max_presentation_form_ratio`.
- Synthesised regression fixture
  `tests/fixtures/pdfs/digital-broken/broken-glyph-id-layer.pdf` mimics
  the Foulabook failure mode without committing copyrighted content.

### Added

- **`--prefetch-models` CLI flag (issue #14)**: downloads the layout +
  OCR weights into the local Hugging Face cache and exits, so offline
  / cache-miss runs no longer fail mid-pipeline. Honours `--config`.
  `ModelDownloadError` messages now reference the flag explicitly.

## [0.1.1] - 2026-05-02

### Fixed

- **ML extra: pin `torchvision==0.20.1`** to match `torch==2.5.1`'s C++
  ABI. Without the pin, `pip install -e '.[ml]'` resolved a torchvision
  wheel built for a different torch ABI, and any page routed through the
  ML branch crashed with
  ``operator torchvision::nms does not exist`` /
  ``partially initialized module 'torchvision'`` when `transformers`
  loaded the DiT layout image processor (issue #12).
- README install section documents the CPU-only install path for both
  `torch` and `torchvision`.
- CI / nightly workflows install `torchvision==0.20.1+cpu` from the
  PyTorch CPU wheel index alongside `torch==2.5.1+cpu`, so the CI matrix
  exercises the same wheels users install locally.
- Regression test (`tests/test_torchvision_abi.py`): in a fresh
  subprocess imports `torch` then `torchvision` and constructs the
  layout adapter — fails on the bad pin, passes on the correct one.

### Known limitations / deferred (carried over)

- **Native-branch validator false-accept on real Arabic PDFs**: the
  `min_arabic_ratio` / `max_replacement_ratio` / `max_word_boundary_kl`
  thresholds are still tuned only against synthetic in-tree fixtures and
  may accept broken text layers (mojibake / glyph-id-not-Unicode) on
  real corpora. Issue #12 part B; corpus-driven retune is the v0.2.0
  deliverable.

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
