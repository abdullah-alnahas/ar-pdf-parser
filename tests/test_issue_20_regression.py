"""Regression tests for issue #20 — GPU OOM on 6 GB cards.

Four root causes (issue #20):

1. **RC#1** — Both layout (DiT-base) and OCR (GOT-OCR-2.0) loaded in
   fp32 by default, doubling VRAM use vs fp16/bf16. The fix adds a
   ``dtype`` knob (``auto``/``float32``/``float16``/``bfloat16``) on
   both adapter configs and forwards it as ``torch_dtype=`` to
   ``from_pretrained``. ``auto`` picks bf16 on Ampere+ CUDA, fp16 on
   older CUDA, fp32 on CPU.
2. **RC#2** — The layout model stayed resident on GPU during the OCR
   phase, eating ~330 MB + activation buffers that OCR could have
   used. The fix evicts the layout model to CPU after every
   ``detect()`` (default-on for CUDA via
   ``HFLayoutDetectorConfig.evict_after_inference``) and brings it
   back before the next page's forward.
3. **RC#3** — A single CUDA OOM in OCR permanently downgraded the
   whole run to CPU. The fix retries once on GPU after
   ``torch.cuda.empty_cache()``; only the second OOM falls back.
4. **RC#4** — ``OCRConfig.max_new_tokens`` defaulted to 1024, but
   KV cache scales linearly in seq_len; on 6 GB cards the upper bound
   inflated per-region peak VRAM unnecessarily. Lowered to 512.

Tests are deterministic: ``torch`` is stubbed where useful and
``transformers.from_pretrained`` is monkey-patched to avoid network /
disk. They do not download models or hit a GPU.
"""

from __future__ import annotations

from typing import Any

import pytest

from arabic_pdf_transcribe._device import resolve_dtype
from arabic_pdf_transcribe.cli import _maybe_build_ml_adapters, _resolve_dtype, build_parser
from arabic_pdf_transcribe.layout.hf_detector import (
    DEFAULT_DTYPE as LAYOUT_DEFAULT_DTYPE,
)
from arabic_pdf_transcribe.layout.hf_detector import (
    HFDiTLayoutDetector,
    HFLayoutDetectorConfig,
)
from arabic_pdf_transcribe.ocr.hf_ocr import (
    DEFAULT_DTYPE as OCR_DEFAULT_DTYPE,
)
from arabic_pdf_transcribe.ocr.hf_ocr import (
    DEFAULT_MAX_NEW_TOKENS,
    HFGotOCRTranscriber,
    OCRConfig,
)

# ---------------------------------------------------------------------------
# RC#4 — max_new_tokens ceiling lowered
# ---------------------------------------------------------------------------


def test_max_new_tokens_default_lowered_to_512() -> None:
    """RC#4: 1024 → 512 keeps per-region peak VRAM bounded on 6 GB cards."""
    cfg = OCRConfig()
    assert cfg.max_new_tokens == 512
    assert DEFAULT_MAX_NEW_TOKENS == 512


# ---------------------------------------------------------------------------
# RC#1 — dtype resolution
# ---------------------------------------------------------------------------


def test_dtype_default_is_auto_on_both_configs() -> None:
    """Both adapters expose a dtype knob and default to ``auto``."""
    assert OCRConfig().dtype == "auto"
    assert OCR_DEFAULT_DTYPE == "auto"
    assert HFLayoutDetectorConfig().dtype == "auto"
    assert LAYOUT_DEFAULT_DTYPE == "auto"


def test_resolve_dtype_explicit_choices_map_to_torch() -> None:
    pytest.importorskip("torch")
    import torch

    assert resolve_dtype("float32", "cpu") is torch.float32
    assert resolve_dtype("float16", "cpu") is torch.float16
    assert resolve_dtype("bfloat16", "cpu") is torch.bfloat16


def test_resolve_dtype_auto_on_cpu_picks_fp32() -> None:
    pytest.importorskip("torch")
    import torch

    assert resolve_dtype("auto", "cpu") is torch.float32


def test_resolve_dtype_auto_on_ampere_picks_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC#1 ``auto``: Ampere+ (compute capability >= 8) → bf16."""
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (8, 0))
    assert resolve_dtype("auto", "cuda") is torch.bfloat16


def test_resolve_dtype_auto_on_pre_ampere_picks_fp16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC#1 ``auto``: older CUDA (e.g. Turing GTX 1660 Ti, capability 7.5)
    → fp16. This is the exact card from the issue repro."""
    pytest.importorskip("torch")
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **kw: (7, 5))
    assert resolve_dtype("auto", "cuda") is torch.float16


def test_resolve_dtype_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="unsupported dtype"):
        resolve_dtype("int8", "cuda")


def test_ocr_adapter_passes_torch_dtype_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC#1 acceptance: ``from_pretrained(torch_dtype=fp16)`` is called
    when the user requests ``dtype="float16"``."""
    pytest.importorskip("torch")
    import torch

    captured: dict[str, object] = {}

    def _fake_processor(model: str, **kwargs: object) -> Any:
        return object()

    def _fake_model(model: str, **kwargs: object) -> Any:
        captured.update(kwargs)
        return _StubModel()

    import transformers

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(_fake_processor)
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        staticmethod(_fake_model),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    transcriber = HFGotOCRTranscriber(OCRConfig(dtype="float16"))
    transcriber._ensure_loaded()
    assert captured.get("torch_dtype") is torch.float16


def test_layout_adapter_passes_torch_dtype_to_from_pretrained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC#1 acceptance applies symmetrically to the layout detector."""
    pytest.importorskip("torch")
    import torch

    captured: dict[str, object] = {}

    def _fake_processor(model: str, **kwargs: object) -> Any:
        return object()

    def _fake_model(model: str, **kwargs: object) -> Any:
        captured.update(kwargs)
        return _StubModelWithLabels({0: "Background", 1: "Text"})

    import transformers

    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", staticmethod(_fake_processor)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSemanticSegmentation,
        "from_pretrained",
        staticmethod(_fake_model),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig(dtype="bfloat16"))
    detector._ensure_loaded()
    assert captured.get("torch_dtype") is torch.bfloat16


def test_dtype_omitted_when_unset_keeps_fp32_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dtype`` defaults to ``"auto"`` which on CPU resolves to fp32 —
    the kwarg is still forwarded explicitly so the loader picks fp32
    deterministically rather than honouring a model-card default."""
    pytest.importorskip("torch")
    import torch

    captured: dict[str, object] = {}

    def _fake_processor(model: str, **kwargs: object) -> Any:
        return object()

    def _fake_model(model: str, **kwargs: object) -> Any:
        captured.update(kwargs)
        return _StubModel()

    import transformers

    monkeypatch.setattr(
        transformers.AutoProcessor, "from_pretrained", staticmethod(_fake_processor)
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText,
        "from_pretrained",
        staticmethod(_fake_model),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    transcriber = HFGotOCRTranscriber(OCRConfig())
    transcriber._ensure_loaded()
    assert captured.get("torch_dtype") is torch.float32


# ---------------------------------------------------------------------------
# RC#2 — layout eviction between pages
# ---------------------------------------------------------------------------


def test_layout_config_evict_after_inference_default_on() -> None:
    """RC#2: eviction is on by default — the OCR adapter should never
    have to compete with a resident layout model on CUDA."""
    assert HFLayoutDetectorConfig().evict_after_inference is True


def test_layout_detector_evicts_to_cpu_after_each_detect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RC#2 acceptance: after a CUDA forward pass, the model is moved to
    CPU and ``torch.cuda.empty_cache()`` is invoked, so OCR has the full
    GPU available. The next page brings the model back to CUDA."""
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    empty_calls: list[None] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: empty_calls.append(None))

    recording_model = _RecordingForwardModel(id2label={0: "Background", 1: "Text"})

    def _fake_processor(model: str, **kwargs: object) -> Any:
        return _PassthroughProcessor()

    def _fake_model(model: str, **kwargs: object) -> Any:
        return recording_model

    import transformers

    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", staticmethod(_fake_processor)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSemanticSegmentation,
        "from_pretrained",
        staticmethod(_fake_model),
    )

    detector = HFDiTLayoutDetector()
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    detector.detect(image, page_index=0)
    detector.detect(image, page_index=1)

    # Initial place_model on CUDA, then per-page bring-back-to-CUDA +
    # evict-to-CPU pairs. After two pages we expect at least one
    # cuda → cpu eviction (and any number of no-op brings-back).
    assert "cpu" in recording_model.to_calls, recording_model.to_calls
    cpu_evictions = sum(1 for d in recording_model.to_calls if d == "cpu")
    assert cpu_evictions >= 2, recording_model.to_calls
    assert len(empty_calls) >= 2


def test_layout_detector_does_not_evict_when_evict_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction is opt-out: setting ``evict_after_inference=False``
    leaves the model on CUDA across pages (matches pre-#20 behaviour)."""
    pytest.importorskip("torch")
    import torch
    from PIL import Image

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    recording_model = _RecordingForwardModel(id2label={0: "Background", 1: "Text"})

    def _fake_processor(model: str, **kwargs: object) -> Any:
        return _PassthroughProcessor()

    def _fake_model(model: str, **kwargs: object) -> Any:
        return recording_model

    import transformers

    monkeypatch.setattr(
        transformers.AutoImageProcessor, "from_pretrained", staticmethod(_fake_processor)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSemanticSegmentation,
        "from_pretrained",
        staticmethod(_fake_model),
    )

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig(evict_after_inference=False))
    image = Image.new("RGB", (32, 32), color=(255, 255, 255))
    detector.detect(image, page_index=0)
    detector.detect(image, page_index=1)

    # With eviction off, only the initial place_model("cuda") happens —
    # no follow-up cpu/cuda ping-pong.
    assert "cpu" not in recording_model.to_calls, recording_model.to_calls


# ---------------------------------------------------------------------------
# RC#3 — OOM gets one in-place retry on GPU before CPU fallback
# ---------------------------------------------------------------------------


def test_ocr_oom_retries_once_on_gpu_then_falls_back_to_cpu() -> None:
    """RC#3: first CUDA OOM gets ``empty_cache()`` + a single retry on
    GPU; only after that retry also OOMs do we permanently fall back."""
    pytest.importorskip("torch")
    import torch

    transcriber = HFGotOCRTranscriber(OCRConfig(device="cuda"))
    transcriber._device = "cuda"
    transcriber._oom_retry_used = False

    class _OOMModel:
        def __init__(self) -> None:
            self.generate_calls = 0
            self.to_calls: list[str] = []

        def generate(self, **kwargs: object) -> Any:
            self.generate_calls += 1
            # First two calls OOM (initial + retry); third (CPU) succeeds.
            if self.generate_calls <= 2:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")

            class _Out:
                sequences = torch.zeros((1, 2), dtype=torch.long)
                scores: tuple[object, ...] = ()

            return _Out()

        def to(self, device: str) -> _OOMModel:
            self.to_calls.append(device)
            return self

    model = _OOMModel()
    transcriber._model = model
    inputs = {
        "pixel_values": torch.zeros((1, 3, 16, 16)),
        "input_ids": torch.zeros((1, 1), dtype=torch.long),
    }

    transcriber._run_generate(inputs, torch)
    assert model.generate_calls == 3  # OOM → retry-OOM → CPU success
    assert transcriber._device == "cpu"
    assert transcriber._oom_retry_used is True
    assert "cpu" in model.to_calls


def test_ocr_oom_retry_succeeds_keeps_model_on_gpu() -> None:
    """RC#3: when the retry-after-empty_cache succeeds, the model stays
    on GPU and subsequent regions also try GPU first (only the retry
    flag is now consumed, so a later OOM goes straight to CPU)."""
    pytest.importorskip("torch")
    import torch

    transcriber = HFGotOCRTranscriber(OCRConfig(device="cuda"))
    transcriber._device = "cuda"
    transcriber._oom_retry_used = False

    class _OneShotOOMModel:
        def __init__(self) -> None:
            self.generate_calls = 0
            self.to_calls: list[str] = []

        def generate(self, **kwargs: object) -> Any:
            self.generate_calls += 1
            if self.generate_calls == 1:
                raise RuntimeError("CUDA out of memory.")

            class _Out:
                sequences = torch.zeros((1, 2), dtype=torch.long)
                scores: tuple[object, ...] = ()

            return _Out()

        def to(self, device: str) -> _OneShotOOMModel:
            self.to_calls.append(device)
            return self

    model = _OneShotOOMModel()
    transcriber._model = model
    inputs = {"pixel_values": torch.zeros((1, 3, 16, 16))}

    transcriber._run_generate(inputs, torch)
    assert model.generate_calls == 2  # OOM → successful retry on GPU
    assert transcriber._device == "cuda"  # stay on GPU
    assert "cpu" not in model.to_calls
    assert transcriber._oom_retry_used is True


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def test_cli_parser_accepts_dtype_flag() -> None:
    parser = build_parser()
    for choice in ("auto", "float32", "float16", "bfloat16"):
        ns = parser.parse_args(["input.pdf", "--dtype", choice])
        assert ns.dtype == choice


def test_cli_parser_rejects_unknown_dtype() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["input.pdf", "--dtype", "int8"])


def test_cli_resolve_dtype_precedence() -> None:
    """CLI --dtype > [runtime].dtype > "auto"."""
    assert _resolve_dtype("float16", {"runtime": {"dtype": "float32"}}) == ("float16", True)
    assert _resolve_dtype(None, {"runtime": {"dtype": "bfloat16"}}) == ("bfloat16", False)
    assert _resolve_dtype(None, {}) == ("auto", False)
    assert _resolve_dtype(None, {"runtime": {"dtype": ""}}) == ("auto", False)


def test_runtime_dtype_propagates_to_adapter_configs() -> None:
    layout, ocr = _maybe_build_ml_adapters({}, dtype="float16")
    if layout is None or ocr is None:
        pytest.skip("[ml] extra unavailable")
    assert layout.config.dtype == "float16"  # type: ignore[attr-defined]
    assert ocr.config.dtype == "float16"  # type: ignore[attr-defined]


def test_per_section_dtype_overrides_runtime_dtype() -> None:
    """Per-section ``[layout].dtype`` is more specific than
    ``[runtime].dtype`` — wins when force_dtype=False."""
    doc: dict[str, object] = {"layout": {"dtype": "bfloat16"}, "ocr": {}}
    layout, ocr = _maybe_build_ml_adapters(doc, dtype="float16", force_dtype=False)
    if layout is None or ocr is None:
        pytest.skip("[ml] extra unavailable")
    assert layout.config.dtype == "bfloat16"  # type: ignore[attr-defined]
    assert ocr.config.dtype == "float16"  # type: ignore[attr-defined]


def test_cli_dtype_flag_overrides_per_section_dtype() -> None:
    """``--dtype float16`` MUST force every adapter, even when a
    per-section dtype is set — same escape hatch as ``--device``."""
    doc: dict[str, object] = {
        "layout": {"dtype": "float32"},
        "ocr": {"dtype": "float32"},
    }
    layout, ocr = _maybe_build_ml_adapters(doc, dtype="float16", force_dtype=True)
    if layout is None or ocr is None:
        pytest.skip("[ml] extra unavailable")
    assert layout.config.dtype == "float16"  # type: ignore[attr-defined]
    assert ocr.config.dtype == "float16"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubModelConfig:
    def __init__(self, id2label: dict[int, str] | None = None) -> None:
        self.id2label = id2label or {}


class _StubModel:
    def __init__(self) -> None:
        self.config = _StubModelConfig()

    def to(self, device: str) -> _StubModel:
        return self


class _StubModelWithLabels(_StubModel):
    def __init__(self, id2label: dict[int, str]) -> None:
        super().__init__()
        self.config = _StubModelConfig(id2label)


class _PassthroughProcessor:
    def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
        import torch

        # 4x4 grid of class 1 — yields a single non-Background region.
        h, w = 4, 4
        return {"pixel_values": torch.zeros((1, 3, h, w))}


class _RecordingForwardModel:
    """Layout-style model: callable + records ``.to`` calls.

    Returns logits whose argmax is class 1 everywhere — a single
    contiguous ``Text`` region — so ``detect()`` produces output but
    we can also inspect the to/empty_cache pattern.
    """

    def __init__(self, id2label: dict[int, str]) -> None:
        self.config = _StubModelConfig(id2label)
        self.to_calls: list[str] = []
        self._num_classes = max(id2label) + 1 if id2label else 2

    def to(self, device: str) -> _RecordingForwardModel:
        self.to_calls.append(device)
        return self

    def __call__(self, **kwargs: object) -> Any:
        import torch

        h, w = 4, 4
        logits = torch.zeros((1, self._num_classes, h, w))
        # Class 1 wins everywhere → one Text region.
        logits[0, 1, :, :] = 5.0

        class _Out:
            pass

        out = _Out()
        out.logits = logits  # type: ignore[attr-defined]
        return out
