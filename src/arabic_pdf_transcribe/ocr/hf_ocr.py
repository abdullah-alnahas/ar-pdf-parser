"""Hugging Face GOT-OCR-2.0 OCR adapter.

# Model selection

Phase 5 selected ``stepfun-ai/GOT-OCR-2.0-hf`` as the default OCR
transcriber after weighing four candidates against the spec
constraints (see plan phase 5 deliverables, ``models.toml``, and the
phase-5 PR description for the full benchmark write-up):

| Candidate                         | License            | Footprint | Arabic       | Verdict |
|-----------------------------------|--------------------|-----------|--------------|---------|
| **GOT-OCR-2.0-hf**                | Apache-2.0         | ~1.1 GB   | multilingual | **chosen** |
| MBZUAI/AIN (Qwen2-VL 7B)          | MIT                | ~14 GB    | Arabic-strong | rejected on footprint (blows the spec's 8 GB CPU-RSS budget alongside the layout model) |
| MohamedRashad/arabic-large-nougat | GPL-3.0            | ~1.4 GB   | Arabic-trained | rejected on license (GPL fails project allow-list) |
| facebook/nougat-base              | CC-BY-NC-4.0       | ~1.4 GB   | Latin-only    | rejected on both license (NC) and language coverage |

GOT-OCR-2.0 is a unified end-to-end document-OCR model with
``GotOcr2ForConditionalGeneration`` head support in
``transformers``. Trained on multilingual document corpora; Arabic
support is documented but not benchmarked against MSA-specific
adversarial inputs in v1. Phase 9's corpus expansion will measure
CER and may swap the default to a fine-tuned Arabic checkpoint
(swapping is a one-file change in this module — the ``OCRTranscriber``
Protocol is the integration boundary).

# Lazy-import discipline

``transformers`` and ``torch`` are imported inside ``_ensure_loaded``
/ ``transcribe``, never at module-import time. The phase-1 lazy-import
test in ``tests/test_skeleton.py`` and the per-phase regression test
in ``tests/test_ocr_lazy_import.py`` enforce that
``import arabic_pdf_transcribe.ocr`` and
``import arabic_pdf_transcribe.ocr.hf_ocr`` do not pull either
library into ``sys.modules``.

# Decoding parameters

``OCRConfig`` exposes the deterministic decoder defaults:

* ``max_new_tokens=4096`` — caps output length per region; protects
  against unterminated generation on adversarial inputs.
* ``num_beams=1`` — greedy decoding by default. Beam search is opt-in
  via ``num_beams=4``; deterministic, no sampling.
* ``do_sample=False`` — required for byte-identical reproducibility
  on the ML path (spec's "reproducibility" line).
* ``temperature=1.0`` — irrelevant when sampling is disabled, but
  pinned for documentation.

# Confidence

GOT-OCR-2.0 generates text via auto-regressive decoding; HF's
``model.generate(...)`` returns logits when ``output_scores=True``.
The adapter aggregates per-token scores into a region-level
confidence in ``[0, 1]`` (geometric mean of softmax probabilities of
the chosen tokens). When the model is replaced by one that does not
expose token logits, the adapter records ``confidence=None`` —
phase 6 / phase 7 are confidence-aware and degrade gracefully.

# Offline / cache-miss handling

The adapter raises
:class:`arabic_pdf_transcribe.errors.ModelDownloadError` when
``transformers`` cannot resolve the pinned revision from the local
cache and the environment denies network access (offline mode, no
network). The CLI maps this exception to exit code 5.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError
from arabic_pdf_transcribe.ocr._crop import DEFAULT_PADDING_PX, crop_region
from arabic_pdf_transcribe.regions import (
    BBox,
    Region,
    RegionRole,
    TableCell,
    TableGrid,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"
DEFAULT_REVISION = "d3017ef2c2c1395888c8d635c5e0508bcb0ac78d"
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_NUM_BEAMS = 1


@dataclass(frozen=True)
class OCRConfig:
    """Tuneable knobs for the HF OCR adapter.

    Defaults are deterministic (greedy, no sampling) so the ML path
    is reproducible up to model floating-point determinism — the
    spec's reproducibility contract.
    """

    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    num_beams: int = DEFAULT_NUM_BEAMS
    do_sample: bool = False
    temperature: float = 1.0
    padding_px: int = DEFAULT_PADDING_PX

    @classmethod
    def from_mapping(cls, mapping: object) -> OCRConfig:
        """Build an OCR config from a TOML ``[ocr]`` section dict."""
        if not isinstance(mapping, dict):
            return cls()
        kwargs = {k: v for k, v in mapping.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


class HFGotOCRTranscriber:
    """Concrete :class:`OCRTranscriber` backed by GOT-OCR-2.0.

    Constructed lazily: model + processor load on first
    ``transcribe`` call (or on explicit ``warm_up()``), so a
    stub-injected pipeline never pays the load cost. Implements the
    :class:`OCRTranscriber` Protocol structurally;
    ``runtime_checkable`` on the Protocol means consumers can
    ``isinstance(t, OCRTranscriber)``.
    """

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self._model: Any = None
        self._processor: Any = None

    def warm_up(self) -> None:
        """Force model + processor load now (otherwise lazy)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            # Local imports — keep transformers / torch out of the
            # top-level import graph.
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
            )
        except ImportError as exc:  # pragma: no cover — install-time check
            raise ModelDownloadError(
                f"transformers is required for OCR but is not installed; "
                f"install the [ml] extra: pip install 'arabic-pdf-transcribe[ml]'. "
                f"({exc})"
            ) from exc
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model, revision=self.config.revision
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.config.model, revision=self.config.revision
            )
        except Exception as exc:  # pragma: no cover — exercised in slow test
            raise ModelDownloadError(
                f"failed to load OCR model {self.config.model}@{self.config.revision[:12]}; "
                f"prefetch the weights with: arabic-pdf-transcribe --prefetch-models "
                f"(or, manually: huggingface-cli download {self.config.model} "
                f"--revision {self.config.revision}). ({exc})"
            ) from exc

    def transcribe(self, region: Region, page_image: PILImage) -> Region:
        """Return ``region`` with text + confidence filled.

        Routes by ``region.role``: figures pass through unchanged;
        tables walk per-cell; everything else crops once and OCRs.
        """
        if region.role is RegionRole.FIGURE:
            # Figures are not OCR'd in v1 (spec's "Semantic output
            # contract"); the orchestrator will embed the bbox crop
            # as an image reference in phase 7's emitter.
            return region
        if region.role is RegionRole.TABLE:
            return self._transcribe_table(region, page_image)
        return self._transcribe_simple(region, page_image)

    def _transcribe_simple(self, region: Region, page_image: PILImage) -> Region:
        try:
            crop = crop_region(page_image, region.bbox, padding_px=self.config.padding_px)
        except ValueError as exc:
            raise OCRTranscriptionError(
                f"region {region.bbox!r} produces a degenerate crop on page {region.page_index}: {exc}"
            ) from exc
        text, confidence = self._transcribe_image(crop)
        return region.with_text(text).with_confidence(confidence)

    def _transcribe_table(self, region: Region, page_image: PILImage) -> Region:
        if region.table_grid is None:
            raise OCRTranscriptionError(
                f"TABLE region on page {region.page_index} has no table_grid; "
                "the layout adapter must populate one (phase 4 contract)."
            )
        new_rows: list[tuple[TableCell, ...]] = []
        for row in region.table_grid.rows:
            new_cells: list[TableCell] = []
            for cell in row:
                cell_text, cell_conf = self._transcribe_cell(cell.bbox, page_image)
                new_cells.append(TableCell(text=cell_text, confidence=cell_conf, bbox=cell.bbox))
            new_rows.append(tuple(new_cells))
        return region.with_table_grid(TableGrid(rows=tuple(new_rows)))

    def _transcribe_cell(self, bbox: BBox, page_image: PILImage) -> tuple[str, float | None]:
        try:
            crop = crop_region(page_image, bbox, padding_px=self.config.padding_px)
        except ValueError:
            # Degenerate cell crop — return empty rather than aborting
            # the whole table. Phase 6 / phase 7 emitters render empty
            # cells without complaint.
            return ("", None)
        return self._transcribe_image(crop)

    def _transcribe_image(self, image: PILImage) -> tuple[str, float | None]:
        """Run model on one cropped image and return (text, confidence)."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        # Local import — gated by [ml] extra.
        import torch

        try:
            inputs = self._processor(
                images=image,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
        except Exception as exc:
            raise OCRTranscriptionError(
                f"OCR generate failed on image {image.size}: {exc}"
            ) from exc
        sequences = outputs.sequences
        # Strip the input prompt tokens from the generated sequence
        # before decoding. ``generate`` returns input_ids + new_tokens
        # concatenated when input_ids are part of inputs.
        prompt_len = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else 0
        gen_tokens = sequences[0][prompt_len:]
        text = self._processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]
        confidence = _confidence_from_scores(outputs, gen_tokens)
        return (text.strip(), confidence)


def _confidence_from_scores(outputs: Any, gen_tokens: Any) -> float | None:
    """Geometric mean of softmax probabilities of the chosen tokens.

    Returns ``None`` when the model did not expose per-token scores
    (e.g. wrapped models that override ``generate`` and drop
    ``output_scores``).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    scores = getattr(outputs, "scores", None)
    if scores is None or len(scores) == 0:
        return None
    log_prob_sum = 0.0
    counted = 0
    # ``scores`` is a tuple of length-``num_new_tokens`` tensors,
    # each shape ``(batch, vocab)``. ``gen_tokens`` is the chosen
    # token ids, length ``num_new_tokens``.
    for step_idx, step_scores in enumerate(scores):
        if step_idx >= len(gen_tokens):
            break
        tok_id = int(gen_tokens[step_idx])
        log_probs = torch.log_softmax(step_scores[0], dim=-1)
        log_prob_sum += float(log_probs[tok_id])
        counted += 1
    if counted == 0:
        return None
    return math.exp(log_prob_sum / counted)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "HFGotOCRTranscriber",
    "OCRConfig",
]
