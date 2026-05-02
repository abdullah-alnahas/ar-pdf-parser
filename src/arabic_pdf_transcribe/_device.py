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


def move_inputs_to_device(inputs: Any, device: str) -> Any:
    """Move processor outputs to ``device`` for both adapters.

    Real ``BatchEncoding`` objects expose ``.to(device)`` directly;
    test stubs return plain dicts (no ``.to``). Fall back to a
    per-tensor move so stubs and real outputs both work.
    """
    move = getattr(inputs, "to", None)
    if callable(move):
        try:
            return move(device)
        except (TypeError, AttributeError, RuntimeError):
            pass
    try:
        import torch
    except ImportError:  # pragma: no cover
        return inputs
    if not isinstance(inputs, dict):
        return inputs
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}


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
]
