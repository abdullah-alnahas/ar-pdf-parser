"""Device resolution shared by HF adapters.

Issue #18 RC#1: the OCR + layout adapters never moved the model or
inputs to a CUDA device, so a user with ``torch.cuda.is_available()``
true still ran inference on CPU — orders of magnitude slower. This
helper resolves a user-supplied device hint (``"auto"`` / ``"cuda"``
/ ``"cpu"``) against ``torch.cuda.is_available()`` and returns the
concrete device string the adapters pass to ``model.to(...)``.

``torch`` is imported lazily so the helper stays cheap on the
``[ml]``-extra-not-installed path; the lazy-import contract is
covered by ``tests/test_skeleton.py``.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)

_VALID = frozenset({"auto", "cuda", "cpu"})


def resolve_device(requested: str | None) -> str:
    """Return ``"cuda"`` or ``"cpu"`` after honouring ``requested``.

    * ``"auto"`` (default, ``None`` treated as ``"auto"``) — pick
      ``"cuda"`` when CUDA is available, else ``"cpu"``.
    * ``"cuda"`` — honour when CUDA is available; else log a warning
      and fall back to ``"cpu"`` (rather than letting torch raise an
      opaque error deep inside ``model.to``).
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


__all__ = ["resolve_device"]
