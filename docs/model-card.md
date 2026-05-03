# Model card

`arabic-pdf-transcribe` v0.1.0 ships two ML models, both gated behind the
`[ml]` extra and pinned by revision hash. Selection followed the spec's
license-compatibility constraint (MIT-compatible permissive licenses
only) and the deferred decision in the spec's "Resolved Decisions"
section.

## Layout detector — `cmarkea/dit-base-layout-detection`

| Field | Value |
|---|---|
| Hugging Face id | `cmarkea/dit-base-layout-detection` |
| Pinned revision | `1995237326c8b53d93525b7b19e20bb363b4eb73` |
| Architecture | BeiT semantic segmentation head, 12 doc-AI classes |
| License | Apache-2.0 |
| Footprint | ~330 MB on disk |
| Inputs | 224×224 RGB image (the adapter resizes the rasterised page) |
| Outputs | Per-pixel class map (12 channels including Background) |
| Adapter | `arabic_pdf_transcribe.layout.hf_detector.HFDiTLayoutDetector` |
| Lazy load | `transformers` / `torch` import deferred to first `detect()` call |

### Selection rationale (vs alternatives)

| Candidate | License | Verdict |
|---|---|---|
| **DiT-base layout** | Apache-2.0 | **Chosen.** Permissive license, script-agnostic (operates on bitmaps so bidi/RTL is transparent), reasonable footprint. |
| DocLayout-YOLO | AGPL-3.0 derivative | Rejected on license — AGPL fails the project's allow-list. |
| Surya layout | GPL-3.0 (post-Surya v1) | Rejected on license — GPL fails the project's allow-list. |

The DiT model is segmentation, not detection; the adapter turns the
per-pixel argmax into bbox regions via connected-components (with a
configurable confidence threshold and minimum region area).

### Known limitations

- Trained on PubLayNet-class English document layouts; Arabic-specific
  layouts (e.g. footnotes in Quranic / classical typography) may
  mis-classify. The role classifier (`roles/classify.py`) re-labels
  via document-level signals (heading sizes, list markers) to mitigate.
- No native table-cell decomposition — `layout/_table_cells.py` runs a
  separate morphology pass on detected `TABLE` regions.

## OCR — `Qwen/Qwen2-VL-2B-Instruct`

| Field | Value |
|---|---|
| Hugging Face id | `Qwen/Qwen2-VL-2B-Instruct` |
| Pinned revision | `895c3a49bc3fa70a340399125c650a463535e71c` |
| Architecture | Qwen2-VL chat-style vision-language model (`Qwen2VLForConditionalGeneration`) |
| License | Apache-2.0 |
| Footprint | ~4.4 GB on disk; ~4 GB fp16 VRAM (per `models.toml`) |
| Inputs | RGB image of one cropped region + chat-template prompt |
| Outputs | UTF-8 text |
| Adapter | `arabic_pdf_transcribe.ocr.hf_ocr.HFQwen2VLOCRTranscriber` |
| Lazy load | `transformers` / `torch` import deferred to first `transcribe()` call |
| Decoding | Greedy (`do_sample=False`, `num_beams=1`) by default for reproducibility |

### Selection rationale

The original v0.1.x default, `stepfun-ai/GOT-OCR-2.0-hf`, was trained
predominantly on English + Chinese OCR corpora. On Arabic body text it
fell back to LaTeX math-italic substitutions (U+1D400 block) per
glyph and produced unusable output (issue #26). Qwen2-VL-2B-Instruct
is Apache-2.0 (clean for the license audit), has documented strong
multilingual coverage including Arabic, and at ~4 GB fp16 fits the 6
GB GPU budget alongside the DiT layout model with the eviction-
between-stages strategy from issue #20. Each OCR call wraps the
cropped region in Qwen2-VL's chat template with a deterministic
Arabic-aware prompt asking for verbatim text extraction.

### Known limitations

- Greedy decoding can mis-handle long ligatured Arabic words; the spec
  accepts CER ≤ 0.05 against reference texts.
- Per-region inference is one model call per region; pages with many
  small regions (dense magazines, ledger-style tables) are slower than
  page-level OCR. Phase-9 perf tuning lives in the post-v1 follow-up.
- The chat-style invocation adds a small fixed prompt overhead
  (~30-40 tokens) per region versus GOT-OCR-2.0's pure-image input.

## Reproducibility scope

- **Native path**: byte-identical Markdown across runs (no model load).
- **ML path**: deterministic decoding (greedy) + pinned revisions →
  reproducible up to model floating-point determinism on CPU. GPU
  floating-point determinism is not guaranteed and is not part of the
  contract.

## Threat / supply-chain notes

- Model weights are pinned by **revision hash**, not floating tag —
  the audit harness rejects updates that change the pinned hash without
  a corresponding `models.toml` change.
- No telemetry; no automatic phone-home; first run downloads from
  Hugging Face Hub, subsequent runs are offline-capable from the local
  cache.
- License compliance is enforced both at model-list level (`models.toml`)
  and at runtime-dep level (`tools/license_audit.py`).

## Model registry — `models.toml`

The full pin set (model id + revision + license + footprint) lives in
[`models.toml`](../models.toml). The license-audit harness reads it on
every run to assert that no model in the active pipeline carries a
license outside the project's allow-list.
