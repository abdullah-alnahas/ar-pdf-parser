"""Regression tests for issue #22 — fp16/bf16 image inputs not cast to model dtype.

# Bug recap

v0.1.5 fixed RC#1 of issue #20 by passing ``torch_dtype=fp16`` to
``from_pretrained`` on Turing CUDA (compute capability 7.5). But the HF
image processor still returns ``pixel_values`` as **fp32** regardless of
the model's loaded dtype. The downstream forward then runs fp32 inputs
through fp16 weights:

    RuntimeError: Input type (float) and bias type (c10::Half) should be the same

The shared helper ``move_inputs_to_device`` was device-only — it never
inspected or changed dtype. The fix extends it to optionally cast every
floating-point tensor (``pixel_values`` and friends) to the model dtype
while leaving integer tensors (``input_ids``, ``attention_mask``) alone.

# What the v0.1.5 suite missed

``test_issue_20_regression.py`` mocked ``from_pretrained`` and
``generate`` with stubs that ignored tensor dtypes. ``torch_dtype=`` was
verified at the call site but no test ever ran a real fp16 weight
through a real fp32 input. These tests close that integration gap with
a real ``torch.nn.Linear(dtype=fp16)`` standing in for the conv layer
that originally raised.
"""

from __future__ import annotations

from typing import Any

import pytest

from arabic_pdf_transcribe._device import move_inputs_to_device

# All tests in this module need a real ``torch`` to exercise the
# dtype-mismatch boundary; if torch isn't installed (no [ml] extra),
# skip cleanly rather than failing.
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Sanity: the test environment can actually surface the dtype-mismatch error
# ---------------------------------------------------------------------------


def test_environment_surfaces_fp16_weight_fp32_input_mismatch() -> None:
    """Pre-condition for the rest of the file: a real fp16 ``Linear`` fed a
    fp32 tensor must raise the exact error the issue reports. Without this
    check, a passing test below could mean the bug, the test, or torch
    silently up-promoting — we want certainty it's the fix."""
    layer = torch.nn.Linear(8, 4, dtype=torch.float16)
    fp32_input = torch.zeros((1, 8), dtype=torch.float32)
    with pytest.raises(RuntimeError, match=r"(?i)input type|bias type|same"):
        layer(fp32_input)


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_move_inputs_casts_floating_point_dict_inputs_to_fp16() -> None:
    """Plain-dict path (test stubs return dicts): floating-point tensors
    cast to ``dtype``, int tensors stay int."""
    inputs = {
        "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.int64),
    }
    moved = move_inputs_to_device(inputs, "cpu", dtype=torch.float16)
    assert moved["pixel_values"].dtype is torch.float16
    assert moved["input_ids"].dtype is torch.int64
    assert moved["attention_mask"].dtype is torch.int64


def test_move_inputs_casts_bf16() -> None:
    inputs = {"pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32)}
    moved = move_inputs_to_device(inputs, "cpu", dtype=torch.bfloat16)
    assert moved["pixel_values"].dtype is torch.bfloat16


def test_move_inputs_dtype_none_keeps_fp32() -> None:
    """Backwards-compatible default: no ``dtype`` arg means no cast."""
    inputs = {"pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32)}
    moved = move_inputs_to_device(inputs, "cpu")
    assert moved["pixel_values"].dtype is torch.float32


def test_move_inputs_dtype_fp32_skips_cast_for_perf() -> None:
    """When dtype is explicitly fp32 (CPU ``auto`` resolution) we
    don't pay an unnecessary ``.to(float32)`` round-trip."""
    pv = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    inputs = {"pixel_values": pv}
    moved = move_inputs_to_device(inputs, "cpu", dtype=torch.float32)
    assert moved["pixel_values"].dtype is torch.float32


def test_move_inputs_handles_batch_encoding_like_object() -> None:
    """``BatchEncoding`` / ``BatchFeature`` have ``.to(device)`` plus
    dict-style ``__getitem__`` / ``__setitem__``. Cast must work via
    the in-place key path after ``.to`` returns the moved container."""

    class _BatchEncodingLike:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def to(self, device: str) -> _BatchEncodingLike:
            self._data = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in self._data.items()
            }
            return self

        def __getitem__(self, key: str) -> Any:
            return self._data[key]

        def __setitem__(self, key: str, value: Any) -> None:
            self._data[key] = value

        def __iter__(self):
            return iter(self._data)

    be = _BatchEncodingLike(
        {
            "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        }
    )
    moved = move_inputs_to_device(be, "cpu", dtype=torch.float16)
    assert moved["pixel_values"].dtype is torch.float16
    assert moved["input_ids"].dtype is torch.int64


# ---------------------------------------------------------------------------
# Layout adapter — real fp16 weights × fp32 processor outputs
# ---------------------------------------------------------------------------


class _Fp16LayoutModel:
    """Stand-in for ``BeitForSemanticSegmentation`` whose first layer is
    a real fp16 ``Linear``. Calling it with a fp32 ``pixel_values`` raises
    the same RuntimeError the user hit — so a passing forward proves the
    adapter cast inputs before calling us."""

    def __init__(self, id2label: dict[int, str]) -> None:
        self._layer = torch.nn.Linear(48, 2, dtype=torch.float16)
        self.dtype = torch.float16

        class _Cfg:
            def __init__(self, labels: dict[int, str]) -> None:
                self.id2label = labels

        self.config = _Cfg(id2label)

    def to(self, device: str) -> _Fp16LayoutModel:
        return self

    def __call__(self, **kwargs: Any) -> Any:
        pixel_values = kwargs["pixel_values"]
        # Flatten to a 48-feature vector then run the fp16 Linear; the
        # operation raises if pixel_values is still fp32.
        flat = pixel_values.reshape(pixel_values.shape[0], -1)
        _ = self._layer(flat)
        # Return a logits tensor that the adapter can softmax/argmax over.
        h, w = 4, 4
        logits = torch.zeros((1, 2, h, w), dtype=torch.float16)
        logits[0, 1, :, :] = 5.0

        class _Out:
            pass

        out = _Out()
        out.logits = logits  # type: ignore[attr-defined]
        return out


class _Fp32PassthroughProcessor:
    """The real DiT image processor returns fp32 ``pixel_values`` — we
    mirror that here so the adapter must cast or the fp16 model raises."""

    def __call__(self, *, images: object, return_tensors: str) -> dict[str, Any]:
        # 4x4 spatial × 3 channels = 48 features for the fp16 Linear.
        return {"pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32)}


def test_layout_adapter_casts_pixel_values_to_model_dtype() -> None:
    """Repro-equivalent: layout adapter loaded fp16, fp32 processor
    output, GPU-ish forward — must NOT raise the dtype-mismatch error."""
    from arabic_pdf_transcribe.layout.hf_detector import (
        HFDiTLayoutDetector,
        HFLayoutDetectorConfig,
    )

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig(device="cpu", dtype="float16"))
    detector._processor = _Fp32PassthroughProcessor()
    detector._model = _Fp16LayoutModel({0: "Background", 1: "Text"})
    detector._id2label = {0: "Background", 1: "Text"}
    detector._device = "cpu"
    detector._dtype = torch.float16

    from PIL import Image

    page = Image.new("RGB", (16, 16), color=(255, 255, 255))
    # Pre-fix this raised "Input type (float) and bias type (c10::Half)
    # should be the same"; post-fix it returns regions cleanly.
    regions = detector.detect(page, page_index=0)
    assert isinstance(regions, list)


# ---------------------------------------------------------------------------
# OCR adapter — same input-prep boundary, exercised through ``_transcribe_image``
# ---------------------------------------------------------------------------


def test_ocr_adapter_casts_pixel_values_to_model_dtype() -> None:
    """OCR adapter ``_transcribe_image`` is the bug-prone path; the
    fp16 ``Linear`` inside ``_StubGenerateModel`` raises the exact issue-22
    error if the adapter forgot to cast inputs to fp16.
    """
    from arabic_pdf_transcribe.ocr.hf_ocr import HFGotOCRTranscriber, OCRConfig

    class _StubProcessor:
        def __call__(self, *, images: object, return_tensors: str) -> dict[str, Any]:
            return {
                "pixel_values": torch.zeros((1, 3, 4, 4), dtype=torch.float32),
                "input_ids": torch.tensor([[101, 102]], dtype=torch.int64),
            }

        def batch_decode(self, sequences: Any, **kwargs: Any) -> list[str]:
            return [""]

    class _StubGenerateModel:
        def __init__(self) -> None:
            self._layer = torch.nn.Linear(48, 4, dtype=torch.float16)
            self.dtype = torch.float16

        def to(self, device: str) -> _StubGenerateModel:
            return self

        def generate(self, **kwargs: Any) -> Any:
            pixel_values = kwargs["pixel_values"]
            assert pixel_values.dtype is torch.float16, (
                "regression: pixel_values reached generate() as "
                f"{pixel_values.dtype}, expected float16"
            )
            # Exercise the real fp16 ``Linear`` — this is what raises
            # in the buggy code path.
            _ = self._layer(pixel_values.reshape(pixel_values.shape[0], -1))
            input_ids = kwargs.get("input_ids")
            prompt_len = int(input_ids.shape[1]) if input_ids is not None else 0

            class _Out:
                sequences: Any = None
                scores: tuple[Any, ...] = ()

            out = _Out()
            out.sequences = torch.zeros((1, prompt_len), dtype=torch.int64)
            out.scores = ()
            return out

    transcriber = HFGotOCRTranscriber(OCRConfig(device="cpu", dtype="float16"))
    transcriber._processor = _StubProcessor()
    transcriber._model = _StubGenerateModel()
    transcriber._device = "cpu"
    transcriber._dtype = torch.float16

    from PIL import Image

    crop = Image.new("RGB", (16, 16), color=(255, 255, 255))
    text, confidence = transcriber._transcribe_image(crop)
    assert isinstance(text, str)
    assert confidence is None  # no scores returned
