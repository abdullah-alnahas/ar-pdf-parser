"""Regression tests for issue #26.

Pre-fix symptom: GOT-OCR-2.0-hf produced LaTeX-math-italic output for
Arabic body text. Every paragraph region collapsed to 1-3 codepoints
in the U+1D400 Mathematical Alphanumeric Symbols block, single Latin
words, or replacement glyphs. Root cause: the GOT-OCR-2.0 training
corpus is dominated by English + Chinese OCR; faced with Arabic the
model fell back to its closest learned pattern — "this looks like
math, output LaTeX".

Fix: swapped the default OCR model to ``Qwen/Qwen2-VL-2B-Instruct``
(Apache-2.0, multilingual incl. Arabic). The :class:`OCRTranscriber`
Protocol is unchanged; only the implementation in
``arabic_pdf_transcribe.ocr.hf_ocr`` swapped.

These tests pin the new default and check the integration boundary —
the chat-template prompt is built and forwarded to the processor on
every OCR call. They do not depend on real model weights.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arabic_pdf_transcribe.ocr.hf_ocr import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    OCR_PROMPT,
    HFQwen2VLOCRTranscriber,
    OCRConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_TOML = REPO_ROOT / "models.toml"


# ---------------------------------------------------------------------------
# Pinned default — Qwen2-VL-2B-Instruct, Apache-2.0
# ---------------------------------------------------------------------------


def test_default_ocr_model_is_qwen2_vl_2b_instruct() -> None:
    """Issue #26 RC: GOT-OCR-2.0 had no real Arabic coverage. The
    default OCR model is now ``Qwen/Qwen2-VL-2B-Instruct``. If this
    drifts back to GOT-OCR-2.0 (or anything English/Chinese-only),
    Arabic output regresses to LaTeX-math-italic garbage."""
    cfg = OCRConfig()
    assert cfg.model == "Qwen/Qwen2-VL-2B-Instruct"
    assert DEFAULT_MODEL == "Qwen/Qwen2-VL-2B-Instruct"
    # The pinned commit hash must be a 40-char hex SHA — never a
    # floating tag like ``main``.
    assert len(DEFAULT_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_REVISION)


def test_models_toml_pins_qwen2_vl_with_apache_license() -> None:
    """The pinned-models registry must list the new OCR default with
    a permissive license that passes the project allow-list."""
    data = tomllib.loads(MODELS_TOML.read_text(encoding="utf-8"))
    ocr_entries = [m for m in data.get("models", []) if m.get("stage") == "ocr"]
    assert (
        len(ocr_entries) == 1
    ), f"expected exactly one [ocr] model entry in models.toml, got {len(ocr_entries)}"
    entry = ocr_entries[0]
    assert entry["name"] == "Qwen/Qwen2-VL-2B-Instruct"
    assert entry["revision"] == DEFAULT_REVISION
    # Apache-2.0 is on the project allow-list (tools/license_audit.py).
    assert entry["license"] == "Apache-2.0"


# ---------------------------------------------------------------------------
# Chat-template invocation — the integration boundary that changed
# ---------------------------------------------------------------------------


class _RecordingProcessor:
    """Captures both ``apply_chat_template`` and ``__call__``."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Any]] = []
        self.processor_calls: list[dict[str, Any]] = []

    def apply_chat_template(
        self,
        messages: list[Any],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        # Snapshot args; assert_called_once-style introspection later.
        self.chat_calls.append(messages)
        assert tokenize is False, "apply_chat_template must not pre-tokenise"
        assert add_generation_prompt is True, "apply_chat_template must prime the assistant turn"
        return "<|im_start|>user\nfake\n<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        import torch

        self.processor_calls.append(kwargs)
        return {
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
        }

    def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
        return ["النص العربي"]


class _StubModel:
    def generate(self, **kwargs: Any) -> Any:
        import torch

        class _Out:
            sequences = torch.zeros((1, 2), dtype=torch.long)
            scores: tuple[Any, ...] = ()

        return _Out()


def test_transcribe_image_calls_apply_chat_template_with_image_and_prompt() -> None:
    """Issue #26 RC: every OCR call must wrap the cropped image in
    Qwen2-VL's chat template. The user message must contain BOTH an
    image part and the OCR prompt; otherwise the model has no
    instruction to extract text and returns empty / garbage."""
    pytest.importorskip("torch")

    processor = _RecordingProcessor()
    transcriber = HFQwen2VLOCRTranscriber()
    transcriber._processor = processor
    transcriber._model = _StubModel()

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    transcriber._transcribe_image(image)

    assert len(processor.chat_calls) == 1, (
        "regression: apply_chat_template must be called exactly once "
        "per region — Qwen2-VL needs the chat-template prompt to OCR"
    )
    messages = processor.chat_calls[0]
    assert len(messages) == 1
    user_turn = messages[0]
    assert user_turn["role"] == "user"
    content = user_turn["content"]
    types_in_content = [part["type"] for part in content]
    assert (
        "image" in types_in_content
    ), "regression: the user-turn content must include an image part"
    assert (
        "text" in types_in_content
    ), "regression: the user-turn content must include the OCR prompt"
    text_part = next(p for p in content if p["type"] == "text")
    assert text_part["text"] == OCR_PROMPT


def test_processor_called_with_text_and_images_kwargs() -> None:
    """Issue #26 RC: the processor must be invoked with both
    ``text=`` and ``images=`` (Qwen2-VL's joint tokenisation API).
    Calling it with only ``images=`` (the legacy GOT-OCR-2.0 path)
    silently dropped the prompt and the model received no instruction.
    """
    pytest.importorskip("torch")

    processor = _RecordingProcessor()
    transcriber = HFQwen2VLOCRTranscriber()
    transcriber._processor = processor
    transcriber._model = _StubModel()

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    transcriber._transcribe_image(image)

    assert len(processor.processor_calls) == 1
    call = processor.processor_calls[0]
    assert "text" in call, "regression: text= prompt was not forwarded"
    assert "images" in call, "regression: images= was not forwarded"
    assert call["return_tensors"] == "pt"
    assert isinstance(call["text"], list) and len(call["text"]) == 1
    assert isinstance(call["images"], list) and len(call["images"]) == 1


def test_ocr_prompt_mentions_arabic_and_forbids_translation() -> None:
    """The default prompt must explicitly cover Arabic and forbid
    translation/commentary — otherwise Qwen2-VL is free to summarise
    or translate, defeating the OCR contract."""
    assert "Arabic" in OCR_PROMPT or "arabic" in OCR_PROMPT.lower()
    # No translation / no commentary — verbatim extraction only.
    lowered = OCR_PROMPT.lower()
    assert "translation" in lowered or "translate" in lowered or "verbatim" in lowered


# ---------------------------------------------------------------------------
# Defence-in-depth — generate kwargs no longer carry stop_strings
# ---------------------------------------------------------------------------


def test_generate_kwargs_have_no_stop_strings_for_qwen2_vl() -> None:
    """Issue #24's ``stop_strings="<|im_end|>"`` plumbing was specific
    to GOT-OCR-2.0 (which never emitted EOS). Qwen2-VL emits EOS at
    the natural end of each assistant turn, so the kwarg is obsolete
    — keeping it could silently truncate output if the chat template
    ever changes the stop token."""
    transcriber = HFQwen2VLOCRTranscriber()
    kwargs = transcriber._generate_kwargs()
    assert "stop_strings" not in kwargs
    assert "tokenizer" not in kwargs
    # Repetition controls (issue #20 RC#3) still ride along.
    assert kwargs["no_repeat_ngram_size"] == 3
    assert kwargs["repetition_penalty"] == pytest.approx(1.05)
    assert kwargs["max_new_tokens"] == 512
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True
