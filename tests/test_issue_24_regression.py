"""Regression tests for issue #24.

Pre-fix symptom: the GOT-OCR-2.0 adapter called ``model.generate``
without ``tokenizer=`` or ``stop_strings="<|im_end|>"``. The model's
chat template never emits a built-in EOS at the natural end of OCR
output, so generation ran to ``max_new_tokens`` and the trailing
tokens were sampled noise — math symbols, CJK, control bytes,
``\\n11\\n11`` line-noise. The pipeline produced a 73 KB "successful"
output that was almost entirely garbage.

Fix: ``_generate_kwargs`` now forwards ``tokenizer=processor.tokenizer``
and ``stop_strings=OCR_STOP_STRING`` so HF's generate loop terminates
at the natural end of OCR output. These tests assert the kwargs are
present (and have the expected values) on every code path that calls
``model.generate`` — the simple path and the CUDA-OOM CPU-fallback
path.
"""

from __future__ import annotations

from typing import Any

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arabic_pdf_transcribe.ocr.hf_ocr import (  # noqa: E402
    OCR_STOP_STRING,
    HFGotOCRTranscriber,
    OCRConfig,
)

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    """Stand-in for ``processor.tokenizer`` — identity is what we assert."""


class _ProcessorWithTokenizer:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()

    def __call__(self, *, images: object, return_tensors: str) -> dict[str, Any]:
        import torch

        return {
            "pixel_values": torch.zeros((1, 3, 16, 16)),
            "input_ids": torch.zeros((1, 1), dtype=torch.long),
        }

    def batch_decode(self, sequences: object, *, skip_special_tokens: bool = True) -> list[str]:
        return [""]


class _CapturingModel:
    """Records kwargs from each ``generate`` call."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> Any:
        import torch

        self.calls.append(dict(kwargs))

        class _Out:
            sequences = torch.zeros((1, 2), dtype=torch.long)
            scores: tuple[Any, ...] = ()

        return _Out()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stop_string_constant_matches_got_ocr_chat_template() -> None:
    """The chat template GOT-OCR-2.0 ships with terminates each turn
    with ``<|im_end|>``. If this constant drifts the fix breaks
    silently — generation runs to max_new_tokens again."""
    assert OCR_STOP_STRING == "<|im_end|>"


def test_generate_receives_tokenizer_and_stop_strings() -> None:
    """Issue #24 RC: both kwargs must reach ``model.generate``."""
    pytest.importorskip("torch")

    processor = _ProcessorWithTokenizer()
    model = _CapturingModel()

    transcriber = HFGotOCRTranscriber()
    transcriber._processor = processor
    transcriber._model = model

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    transcriber._transcribe_image(image)

    assert len(model.calls) == 1
    kwargs = model.calls[0]
    assert "tokenizer" in kwargs, (
        "regression: tokenizer= must be forwarded to generate so "
        "stop_strings can be matched against decoded output"
    )
    assert kwargs["tokenizer"] is processor.tokenizer
    assert kwargs.get("stop_strings") == OCR_STOP_STRING


def test_generate_kwargs_preserves_existing_repetition_controls() -> None:
    """The fix must not regress issue #20 RC#3 — repetition controls
    still need to ride along on every generate call (defence in depth
    for adversarial crops)."""
    pytest.importorskip("torch")

    processor = _ProcessorWithTokenizer()
    model = _CapturingModel()

    transcriber = HFGotOCRTranscriber(OCRConfig())
    transcriber._processor = processor
    transcriber._model = model

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    transcriber._transcribe_image(image)

    kwargs = model.calls[0]
    assert kwargs["no_repeat_ngram_size"] == 3
    assert kwargs["repetition_penalty"] == pytest.approx(1.05)
    assert kwargs["max_new_tokens"] == 512
    assert kwargs["return_dict_in_generate"] is True
    assert kwargs["output_scores"] is True


def test_oom_cpu_fallback_path_also_forwards_stop_strings() -> None:
    """The CUDA-OOM → CPU-fallback retry must also use the corrected
    kwargs; otherwise a fallback run silently degrades to garbage
    output. Walks ``_run_generate`` directly with a model that OOMs
    twice (initial + retry) to force the permanent CPU fallback."""
    pytest.importorskip("torch")
    import torch

    processor = _ProcessorWithTokenizer()
    captured: list[dict[str, Any]] = []
    oom_count = {"n": 0}

    class _OOMOnGPUModel:
        def to(self, device: str) -> _OOMOnGPUModel:
            return self

        def generate(self, **kwargs: Any) -> Any:
            captured.append(dict(kwargs))
            if oom_count["n"] < 2:
                oom_count["n"] += 1
                raise RuntimeError("CUDA out of memory")

            class _Out:
                sequences = torch.zeros((1, 2), dtype=torch.long)
                scores: tuple[Any, ...] = ()

            return _Out()

    transcriber = HFGotOCRTranscriber(OCRConfig(device="cuda"))
    transcriber._processor = processor
    transcriber._model = _OOMOnGPUModel()
    transcriber._device = "cuda"

    inputs = processor(images=Image.new("RGB", (16, 16)), return_tensors="pt")
    transcriber._run_generate(inputs, torch)

    # Three attempts: GPU, GPU retry, CPU fallback. All three must
    # carry tokenizer + stop_strings.
    assert len(captured) == 3
    for call in captured:
        assert call.get("stop_strings") == OCR_STOP_STRING
        assert call.get("tokenizer") is processor.tokenizer


def test_processor_without_tokenizer_skips_stop_strings() -> None:
    """Defensive: stub processors used in unit tests across the suite
    have no ``tokenizer`` attribute. The adapter must not raise on
    them — instead, omit ``tokenizer=``/``stop_strings=`` so existing
    tests keep working. Production HF AutoProcessor always exposes
    ``tokenizer`` so the real fix still applies end-to-end."""
    pytest.importorskip("torch")

    class _NoTokProcessor:
        def __call__(self, *, images: object, return_tensors: str) -> dict[str, Any]:
            import torch

            return {
                "pixel_values": torch.zeros((1, 3, 16, 16)),
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
            }

        def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
            return [""]

    model = _CapturingModel()
    transcriber = HFGotOCRTranscriber()
    transcriber._processor = _NoTokProcessor()
    transcriber._model = model

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    transcriber._transcribe_image(image)

    kwargs = model.calls[0]
    assert "tokenizer" not in kwargs
    assert "stop_strings" not in kwargs


def test_confidence_degrades_to_none_when_aggregation_raises() -> None:
    """Issue #24 concern #3: ``stop_strings`` interacts with score
    collection. If aggregating the per-token logits raises (e.g.
    misaligned scores tuple), the adapter must return
    ``confidence=None`` for that region rather than crashing."""
    pytest.importorskip("torch")
    import torch

    processor = _ProcessorWithTokenizer()

    class _BogusScoresModel:
        def generate(self, **kwargs: Any) -> Any:
            class _Out:
                sequences = torch.zeros((1, 3), dtype=torch.long)
                # Non-tensor scores entry → log_softmax raises RuntimeError.
                scores = (object(),)

            return _Out()

    transcriber = HFGotOCRTranscriber()
    transcriber._processor = processor
    transcriber._model = _BogusScoresModel()

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    text, confidence = transcriber._transcribe_image(image)
    assert isinstance(text, str)
    assert confidence is None
