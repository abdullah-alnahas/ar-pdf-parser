"""Regression test for issue #12 — torchvision / torch ABI pin.

The ``[ml]`` extra used to pin ``torch==2.5.1`` but leave ``torchvision``
floating, so pip resolved a torchvision wheel built against a different
torch ABI. Loading the layout model's image processor then crashed with
``operator torchvision::nms does not exist`` (or, depending on the
import order, ``partially initialized module 'torchvision' has no
attribute 'extension'``).

This test asserts:

1. ``import torch`` followed by ``import torchvision`` succeeds in a
   fresh interpreter (the canonical surface of the ABI mismatch).
2. The installed pair matches the project's pinned versions
   (``torch==2.5.1`` + ``torchvision==0.20.1``).
3. Constructing the layout adapter (without warming up real weights)
   does not raise — the constructor reads ``transformers`` lazily, so
   the failure surface here is dependency import, not network /
   model-load.

The test is skipped when the ``[ml]`` extra is not installed (the same
gate ``test_layout_hf_detector.py`` uses).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


PINNED_TORCH = "2.5.1"
PINNED_TORCHVISION = "0.20.1"


def _strip_local(version: str) -> str:
    """Drop a PEP 440 local-version segment (``+cpu``, ``+cu121``, ...)."""
    return version.split("+", 1)[0]


def test_torch_torchvision_imports_in_fresh_interpreter() -> None:
    """``import torch; import torchvision`` must not raise.

    Runs in a subprocess so a previously-cached good import in the
    parent test process cannot mask a broken install. This is the
    canonical reproduction surface for issue #12.
    """
    code = textwrap.dedent(
        """
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        # Touch the op that was missing on the bad pin to force the
        # native registration to evaluate.
        assert hasattr(torchvision.ops, "nms")
        print("ok")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"torch+torchvision import failed (issue #12 ABI mismatch?):\n"
        f"--- stdout ---\n{proc.stdout}\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert proc.stdout.strip() == "ok"


def test_pinned_versions_match_project_extra() -> None:
    """Installed torch / torchvision match the pins in the ``[ml]`` extra."""
    assert _strip_local(torch.__version__) == PINNED_TORCH, (
        f"torch=={torch.__version__} does not match project pin {PINNED_TORCH}; "
        f"issue #12 mismatch surface."
    )
    assert _strip_local(torchvision.__version__) == PINNED_TORCHVISION, (
        f"torchvision=={torchvision.__version__} does not match project pin "
        f"{PINNED_TORCHVISION}; issue #12 ABI mismatch surface."
    )


def test_layout_adapter_constructs_without_raising() -> None:
    """The layout adapter constructor must not pull torchvision into a
    broken state.

    We do not call ``warm_up`` (that loads real weights from the HF
    cache and is covered by the slow nightly job). The constructor
    importing ``transformers`` indirectly imports torchvision via the
    image processor registry on some installs, so this is enough to
    surface the ABI mismatch.
    """
    pytest.importorskip("transformers")
    from arabic_pdf_transcribe.layout.hf_detector import HFDiTLayoutDetector

    detector = HFDiTLayoutDetector()
    assert detector is not None
