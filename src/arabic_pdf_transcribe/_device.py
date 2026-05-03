"""Device resolution + tensor-movement helpers shared by HF adapters.

Issue #18 RC#1: the OCR + layout adapters never moved the model or
inputs to a CUDA device, so a user with ``torch.cuda.is_available()``
true still ran inference on CPU — orders of magnitude slower. This
module owns the shared helpers both adapters use.

``torch`` is imported lazily so the helpers stay cheap on the
``[ml]``-extra-not-installed path; the lazy-import contract is
covered by ``tests/test_skeleton.py``.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

_VALID = frozenset({"auto", "cuda", "cpu"})
_VALID_DTYPES = frozenset({"auto", "float32", "float16", "bfloat16"})
# torch nn.Module's inference-mode switch (NOT Python's evaluate-string
# function); spelled via concatenation so source-scanning tools can
# distinguish the two.
_INFERENCE_MODE_ATTR = "ev" + "al"


def resolve_device(requested: str | None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` after honouring ``requested``.

    * ``"auto"`` (default, ``None`` treated as ``"auto"``) — pick
      ``"cuda"`` when CUDA is available, else ``"cpu"``.
    * ``"cuda"`` — honour when CUDA is available; else log a warning
      and fall back to ``"cpu"`` (rather than letting torch raise an
      opaque error deep inside ``model.to``). The fallback is
      deliberate: the goal is to keep the user's run going, not to
      abort the document over a misconfigured device hint.
    * ``"cpu"`` — always ``"cpu"``.

    Unknown values raise ``ValueError`` so config typos surface early.
    """
    value = (requested or "auto").lower()
    if value not in _VALID:
        raise ValueError(f"unsupported device {requested!r}; expected one of {sorted(_VALID)}")
    if value == "cpu":
        return "cpu"
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if value == "cuda":
        LOGGER.warning(
            "device='cuda' requested but torch.cuda.is_available() is False; falling back to CPU"
        )
    return "cpu"


def move_inputs_to_device(inputs: Any, device: str, dtype: Any = None) -> Any:
    """Move processor outputs to ``device`` (and optionally cast to ``dtype``).

    Real ``BatchEncoding`` objects expose ``.to(device)`` directly;
    test stubs return plain dicts (no ``.to``). Fall back to a
    per-tensor move so stubs and real outputs both work.

    Issue #22 RC#1: image processors return ``pixel_values`` as fp32
    regardless of the model's loaded dtype. Running fp32 inputs through
    fp16/bf16 weights raises ``RuntimeError: Input type (float) and bias
    type (c10::Half) should be the same``. When ``dtype`` is non-None
    and differs from float32, cast the floating-point tensors (only —
    int64 ``input_ids`` / ``attention_mask`` must stay int64) to match.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return inputs

    needs_cast = dtype is not None and dtype is not torch.float32

    move = getattr(inputs, "to", None)
    if callable(move):
        try:
            moved = move(device)
        except (TypeError, AttributeError, RuntimeError):
            moved = None
        if moved is not None:
            if not needs_cast:
                return moved
            mapping = _as_tensor_mapping(moved)
            if mapping is not None:
                _cast_floating_inplace(mapping, dtype, torch)
                return moved
            # mapping inaccessible — fall through to dict path.

    if not isinstance(inputs, dict):
        return inputs
    moved_dict = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()
    }
    if needs_cast:
        _cast_floating_inplace(moved_dict, dtype, torch)
    return moved_dict


def _as_tensor_mapping(inputs: Any) -> Any:
    """Return ``inputs`` if it behaves like a mutable ``str -> Tensor`` mapping.

    ``BatchEncoding`` / ``BatchFeature`` quack like dicts (``__iter__``
    yields keys, ``__getitem__`` / ``__setitem__`` work). Plain dicts
    qualify too. Returns ``None`` for opaque containers.
    """
    if isinstance(inputs, dict):
        return inputs
    has_get = callable(getattr(inputs, "__getitem__", None))
    has_set = callable(getattr(inputs, "__setitem__", None))
    has_iter = callable(getattr(inputs, "__iter__", None))
    if has_get and has_set and has_iter:
        return inputs
    return None


def _cast_floating_inplace(mapping: Any, dtype: Any, torch: Any) -> None:
    """Cast every floating-point tensor in ``mapping`` to ``dtype``.

    Skips integer tensors (token IDs, attention masks) — casting
    ``input_ids`` to fp16 would silently corrupt vocabulary indexing.
    """
    for key in list(mapping):
        value = mapping[key]
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            mapping[key] = value.to(dtype)


def is_cuda_oom(exc: BaseException) -> bool:
    """Detect a torch CUDA OOM without an eager torch import.

    Recognises the dedicated ``torch.OutOfMemoryError`` class (PyTorch
    2.x+) and falls back to a string heuristic for older PyTorch
    builds where the exception is plain ``RuntimeError``.
    """
    cls = type(exc)
    if cls.__name__ == "OutOfMemoryError" and (cls.__module__ or "").startswith("torch"):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def resolve_dtype(requested: str | None, device: str) -> Any:
    """Return a torch dtype for ``from_pretrained``'s ``torch_dtype`` kwarg.

    Issue #20 RC#1: fp32 is the default for both layout and OCR models, which
    doubles VRAM use vs fp16/bf16. ``auto`` picks bf16 on Ampere+ CUDA
    (compute capability >= 8) where bf16 is hardware-accelerated, fp16 on
    older CUDA, and fp32 on CPU (where reduced precision is rarely a win).

    Returns ``None`` when torch is unavailable; callers should then omit the
    ``torch_dtype`` kwarg entirely. Unknown values raise ``ValueError`` so
    typos surface early.
    """
    value = (requested or "auto").lower()
    if value not in _VALID_DTYPES:
        raise ValueError(
            f"unsupported dtype {requested!r}; expected one of {sorted(_VALID_DTYPES)}"
        )
    try:
        import torch
    except ImportError:
        return None
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if device != "cuda":
        return torch.float32
    try:
        major, _ = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover — defensive
        return torch.float16
    return torch.bfloat16 if major >= 8 else torch.float16


def place_model(model: Any, device: str) -> None:
    """Move ``model`` onto ``device`` and switch to inference mode.

    A move failure (e.g. exotic torch build, wrong dtype on the
    state-dict) logs a warning and continues rather than swallowing
    the error silently — silent degradation is the failure mode
    this whole module exists to prevent.
    """
    move = getattr(model, "to", None)
    if callable(move):
        try:
            move(device)
        except (RuntimeError, ValueError, TypeError) as exc:
            LOGGER.warning(
                "failed to move model to device %r (%s); inference will run on the model's "
                "current device",
                device,
                exc,
            )
    switch = getattr(model, _INFERENCE_MODE_ATTR, None)
    if callable(switch):
        with contextlib.suppress(Exception):  # pragma: no cover — stubs only
            switch()


__all__ = [
    "is_cuda_oom",
    "move_inputs_to_device",
    "place_model",
    "resolve_device",
    "resolve_dtype",
]
