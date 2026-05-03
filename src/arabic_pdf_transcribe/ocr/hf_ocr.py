"""Hugging Face OCR adapter (Qwen2-VL-2B-Instruct).

# Model selection

Issue #26 swapped the default OCR transcriber from
``stepfun-ai/GOT-OCR-2.0-hf`` to ``Qwen/Qwen2-VL-2B-Instruct``.

GOT-OCR-2.0 was selected in phase 5 partly because Arabic-strong
alternatives blew the spec's CPU-RSS budget (MBZUAI/AIN at ~14 GB) or
had blocking licenses (Nougat-Arabic GPL-3.0). On the real Foulabook
Arabic corpus the model produced unusable output: every paragraph
region collapsed to 1-3 LaTeX math-italic codepoints (U+1D400 block,
Mathematical Alphanumeric Symbols), single Latin words, or replacement
glyphs. Root cause: the GOT-OCR-2.0 training corpus is dominated by
English + Chinese document OCR; Arabic appears only in trace amounts.
Faced with Arabic body text the model fell back to its closest learned
pattern — "this looks like rendered math, output LaTeX" — and emitted
math-italic substitutions per glyph.

| Candidate                         | License            | Footprint | Arabic       | Verdict |
|-----------------------------------|--------------------|-----------|--------------|---------|
| **Qwen/Qwen2-VL-2B-Instruct**     | Apache-2.0         | ~4 GB fp16 | strong multilingual incl. Arabic | **chosen** |
| stepfun-ai/GOT-OCR-2.0-hf         | Apache-2.0         | ~1.1 GB   | English/Chinese-only | issue #26: unusable on Arabic |
| MBZUAI/AIN (Qwen2-VL 7B)          | MIT                | ~14 GB    | Arabic-strong | rejected on footprint |
| MohamedRashad/arabic-large-nougat | GPL-3.0            | ~1.4 GB   | Arabic-trained | rejected on license |
| facebook/nougat-base              | CC-BY-NC-4.0       | ~1.4 GB   | Latin-only    | rejected on both license and language |
| OpenGVLab/InternVL2-2B            | MIT                | ~4 GB     | multilingual  | viable backup |

Qwen2-VL-2B-Instruct is a chat-style vision-language model with
``Qwen2VLForConditionalGeneration`` head support in
``transformers>=4.45``; the project pins ``transformers==4.49.0``.
The processor exposes a chat-template API: each call wraps the cropped
region in a single user-turn message that contains the image and a
short Arabic-aware instruction prompt, and decodes the assistant's
response back into plain text. The integration boundary
(:class:`OCRTranscriber` Protocol) is unchanged — only the
implementation in this module changed.

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

* ``max_new_tokens=512`` — caps output length per region.
* ``num_beams=1`` — greedy decoding by default.
* ``do_sample=False`` — required for byte-identical reproducibility
  on the ML path.
* ``temperature=1.0`` — irrelevant when sampling is disabled, but
  pinned for documentation.

Qwen2-VL emits a built-in EOS at the end of each assistant turn, so
no ``stop_strings`` plumbing is needed (unlike GOT-OCR-2.0, which
required forwarding ``<|im_end|>`` as a stop string — issue #24).

# Prompt

A short, deterministic prompt asks the model to extract Arabic body
text verbatim with no commentary. The prompt is multilingual-tolerant
(it keeps Latin words / digits intact) so mixed Arabic-Latin layouts
still survive.

# Confidence

Qwen2-VL-2B-Instruct generates text via auto-regressive decoding;
HF's ``model.generate(...)`` returns logits when
``output_scores=True``. The adapter aggregates per-token scores into a
region-level confidence in ``[0, 1]`` (geometric mean of softmax
probabilities of the chosen tokens). When the score tensors cannot be
aligned (e.g. early stopping on EOS misaligns the scores tuple on
some transformers versions), the adapter records ``confidence=None``.

# Offline / cache-miss handling

The adapter raises
:class:`arabic_pdf_transcribe.errors.ModelDownloadError` when
``transformers`` cannot resolve the pinned revision from the local
cache and the environment denies network access (offline mode, no
network). The CLI maps this exception to exit code 5.
"""

from __future__ import annotations

import contextlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arabic_pdf_transcribe._device import (
    is_cuda_oom,
    move_inputs_to_device,
    place_model,
    resolve_device,
    resolve_dtype,
)
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

DEFAULT_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
# Issue #18 RC#3 lowered the cap from 4096 → 1024. Issue #20 RC#4
# lowers it again 1024 → 512: with KV cache scaling linearly in
# seq_len (and SDPA attention quadratically), the 1024 ceiling
# inflated per-region peak VRAM unnecessarily on 6 GB cards. 512
# tokens is still ~5x the 99th-percentile paragraph length; users
# with very long regions can override via [ocr].max_new_tokens.
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_NUM_BEAMS = 1
# Belt-and-suspenders against repetition loops on adversarial crops.
DEFAULT_NO_REPEAT_NGRAM_SIZE = 3
DEFAULT_REPETITION_PENALTY = 1.05
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "auto"
# Issue #26: deterministic Arabic-aware OCR prompt. Wrapped in
# Qwen2-VL's chat-template format on every call. Kept short to bound
# the input-token count (KV cache pressure) and explicit about the
# script so the model does not transliterate or summarise.
OCR_PROMPT = (
    "Extract all visible text from this image, preserving the "
    "original script (Arabic, Latin, or digits). Output only the "
    "verbatim text — no translation, no commentary, no formatting."
)


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
    no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    prompt: str = OCR_PROMPT

    @classmethod
    def from_mapping(cls, mapping: object) -> OCRConfig:
        """Build an OCR config from a TOML ``[ocr]`` section dict."""
        if not isinstance(mapping, dict):
            return cls()
        kwargs = {k: v for k, v in mapping.items() if k in cls.__dataclass_fields__}
        return cls(**kwargs)  # type: ignore[arg-type]


class HFQwen2VLOCRTranscriber:
    """Concrete :class:`OCRTranscriber` backed by Qwen2-VL-2B-Instruct.

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
        self._device: str | None = None
        # Issue #22 RC#1: cast fp32 image inputs to model dtype before
        # forward when running fp16/bf16, otherwise the forward raises
        # "Input type (float) and bias type (c10::Half) should be the same".
        self._dtype: Any = None
        # Issue #20 RC#3: a CUDA OOM gets one in-place retry on GPU
        # (after empty_cache) before we permanently fall back to CPU.
        # The flag stays set across regions so a subsequent OOM falls
        # back immediately rather than thrashing.
        self._oom_retry_used: bool = False

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
        # Resolve the target device first so dtype "auto" can pick
        # bf16 / fp16 only when actually heading to CUDA.
        self._device = resolve_device(self.config.device)
        torch_dtype = resolve_dtype(self.config.dtype, self._device)
        self._dtype = torch_dtype
        load_kwargs: dict[str, Any] = {"revision": self.config.revision}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        try:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model, revision=self.config.revision
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.config.model, **load_kwargs
            )
        except Exception as exc:  # pragma: no cover — exercised in slow test
            raise ModelDownloadError(
                f"failed to load OCR model {self.config.model}@{self.config.revision[:12]}; "
                f"prefetch the weights with: arabic-pdf-transcribe --prefetch-models "
                f"(or, manually: huggingface-cli download {self.config.model} "
                f"--revision {self.config.revision}). ({exc})"
            ) from exc
        self._place_model()

    def _place_model(self) -> None:
        """Move model + switch to inference mode on the resolved device.

        Issue #18 RC#1: previously the model was left on whichever
        device ``from_pretrained`` defaulted to (CPU), so a user with
        CUDA available still ran inference on CPU — orders of
        magnitude slower. ``self._device`` is sticky so a CUDA OOM
        mid-run can permanently downgrade us to CPU. Issue #20: the
        device is now resolved up-front in ``_ensure_loaded`` so
        ``torch_dtype`` selection can depend on it.
        """
        if self._device is None:
            self._device = resolve_device(self.config.device)
        place_model(self._model, self._device)

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

    def _build_inputs(self, image: PILImage) -> Any:
        """Wrap ``image`` in Qwen2-VL's chat-template prompt and tokenize.

        Issue #26: Qwen2-VL is a chat-style VLM. Each OCR call sends a
        single user turn whose content is ``[image, text-prompt]``.
        The processor renders that into the model's chat template
        (with ``add_generation_prompt=True`` so the assistant turn is
        primed), then tokenises image + text in one call.
        """
        assert self._processor is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.config.prompt},
                ],
            }
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
        )

    def _transcribe_image(self, image: PILImage) -> tuple[str, float | None]:
        """Run model on one cropped image and return (text, confidence)."""
        self._ensure_loaded()
        assert self._model is not None
        assert self._processor is not None
        # Local import — gated by [ml] extra.
        import torch

        try:
            inputs = self._build_inputs(image)
            inputs = move_inputs_to_device(inputs, self._device or "cpu", dtype=self._dtype)
            outputs = self._run_generate(inputs, torch)
        except OCRTranscriptionError:
            raise
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

    def _run_generate(self, inputs: Any, torch: Any) -> Any:
        """Call ``model.generate`` with the configured decoding kwargs.

        On a CUDA OOM:
        * Issue #20 RC#3 — first OOM gets a single in-place retry on GPU
          after ``torch.cuda.empty_cache()``; transient pressure (e.g. a
          peer adapter that just released activations) can clear up
          enough to let the retry succeed without abandoning the GPU.
        * If the retry also OOMs (or a later region OOMs again), we
          fall back to CPU permanently for the rest of the run: move
          the OCR model to CPU, downgrade ``self._device``, and retry
          on CPU. Subsequent regions stay on CPU.
        """
        kwargs = self._generate_kwargs()
        try:
            with torch.no_grad():
                return self._model.generate(**inputs, **kwargs)
        except RuntimeError as exc:
            if not is_cuda_oom(exc) or self._device != "cuda":
                raise
            if not self._oom_retry_used:
                self._oom_retry_used = True
                LOGGER.warning(
                    "OCR: CUDA OOM during generate; freeing cache and retrying on GPU once"
                )
                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
                try:
                    with torch.no_grad():
                        return self._model.generate(**inputs, **kwargs)
                except RuntimeError as retry_exc:
                    if not is_cuda_oom(retry_exc):
                        raise
                    # fall through to permanent CPU fallback
            LOGGER.warning(
                "OCR: CUDA OOM during generate; falling back to CPU for the remainder of this run"
            )
            with contextlib.suppress(Exception):  # pragma: no cover — defensive
                torch.cuda.empty_cache()
            self._model.to("cpu")
            self._device = "cpu"
            cpu_inputs = move_inputs_to_device(inputs, "cpu", dtype=self._dtype)
            with torch.no_grad():
                return self._model.generate(**cpu_inputs, **kwargs)

    def _generate_kwargs(self) -> dict[str, Any]:
        """Decoding kwargs forwarded to ``model.generate``.

        Qwen2-VL emits a built-in EOS at the end of each assistant
        turn, so generation terminates cleanly without the
        ``stop_strings`` plumbing GOT-OCR-2.0 needed (issue #24).
        """
        return {
            "max_new_tokens": self.config.max_new_tokens,
            "num_beams": self.config.num_beams,
            "do_sample": self.config.do_sample,
            "temperature": self.config.temperature,
            "no_repeat_ngram_size": self.config.no_repeat_ngram_size,
            "repetition_penalty": self.config.repetition_penalty,
            "return_dict_in_generate": True,
            "output_scores": True,
        }


def _confidence_from_scores(outputs: Any, gen_tokens: Any) -> float | None:
    """Geometric mean of softmax probabilities of the chosen tokens.

    Returns ``None`` when the model did not expose per-token scores
    (e.g. wrapped models that override ``generate`` and drop
    ``output_scores``) or when the score tensors cannot be aligned
    with ``gen_tokens`` (early EOS truncates the output mid-step,
    which on some transformers versions misaligns the scores tuple —
    degrade to ``None`` rather than raising).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    scores = getattr(outputs, "scores", None)
    if scores is None or len(scores) == 0:
        return None
    try:
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
    except Exception as exc:
        LOGGER.debug("OCR: confidence aggregation skipped (%s)", exc)
        return None
    if counted == 0:
        return None
    return math.exp(log_prob_sum / counted)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_REVISION",
    "OCR_PROMPT",
    "HFQwen2VLOCRTranscriber",
    "OCRConfig",
]
