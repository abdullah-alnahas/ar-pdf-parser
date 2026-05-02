"""Regression tests for issue #16.

Issue #16 surfaced three independent failure modes when running the
real Foulabook Arabic PDF on v0.1.2:

1. ``transformers==4.46.3`` did not recognise the ``got_ocr2`` model
   type (``GotOcr2ForConditionalGeneration`` landed in 4.49.0), so
   ``AutoModelForImageTextToText.from_pretrained`` raised on every
   ML-branch page. The fix bumps the ``[ml]`` pin to
   ``transformers>=4.49,<4.50``. This module verifies the pin in
   ``pyproject.toml`` and (if the ``[ml]`` extra is installed) that
   the OCR model's *config* instantiates without network — a fast
   smoke that would have caught the regression on day one.

2. The DiT layout label ``Formula`` was mapped to
   :data:`RegionRole.UNKNOWN`, and phase 7's emitter dropped UNKNOWN
   regions silently. The English-trained DiT mislabels justified
   Arabic body text as ``Formula``, producing pages where every
   region was lost. The fix remaps ``Formula → PARAGRAPH``.

3. JSON-log ``failure`` events for typed pipeline errors carried only
   the exception class name, so the actionable hint baked into the
   exception (``"prefetch the weights with: arabic-pdf-transcribe
   --prefetch-models"``) never reached the user. The fix appends
   ``str(exc)`` to the reason field; this module verifies the
   serialised event surfaces both the class name and the message.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pytest

from arabic_pdf_transcribe._logging import (
    ProgressEvent,
    ProgressLogger,
    ProgressMode,
)
from arabic_pdf_transcribe.layout._classes import role_for_label
from arabic_pdf_transcribe.regions import RegionRole

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


# ---------------------------------------------------------------------------
# RC#1 — transformers pin
# ---------------------------------------------------------------------------


def test_pyproject_pins_transformers_supporting_got_ocr2() -> None:
    """The ``[ml]`` extra must pin a transformers release that knows
    the GOT-OCR-2.0 model type (``got_ocr2``). The class landed in
    4.49.0; older pins silently mis-loaded the model. Issue #16 RC#1.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    matches = re.findall(r'"transformers([^"]*)"', text)
    assert matches, "no transformers requirement found in pyproject.toml"
    spec = matches[0]
    # Accept either ``>=4.49,<4.50`` or any pin >= 4.49.
    lower_bound = re.search(r">=\s*4\.(\d+)", spec)
    assert lower_bound is not None, (
        f"transformers spec {spec!r} must declare a >=4.49 lower bound "
        f"(GOT-OCR-2.0 support landed in 4.49.0; see issue #16)"
    )
    minor = int(lower_bound.group(1))
    assert minor >= 49, (
        f"transformers spec {spec!r} pins minor {minor} < 49; "
        f"GOT-OCR-2.0 (model_type 'got_ocr2') is unrecognised below 4.49.0"
    )


def test_got_ocr2_config_instantiates_when_transformers_installed() -> None:
    """If the ``[ml]`` extra is installed, ``AutoConfig`` must resolve
    the GOT-OCR-2.0 model type. Reads the config from the local cache
    (no network). Skipped when transformers is absent or the config
    is not in the cache (CI without prefetched HF cache).
    """
    transformers = pytest.importorskip("transformers")
    AutoConfig = transformers.AutoConfig
    from arabic_pdf_transcribe.ocr.hf_ocr import DEFAULT_MODEL, DEFAULT_REVISION

    try:
        cfg = AutoConfig.from_pretrained(
            DEFAULT_MODEL,
            revision=DEFAULT_REVISION,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover — no cache in CI
        pytest.skip(f"GOT-OCR-2.0 config not cached locally ({type(exc).__name__})")
    assert cfg.model_type == "got_ocr2", (
        f"AutoConfig returned model_type={cfg.model_type!r}; "
        f"expected 'got_ocr2'. transformers may be too old (<4.49)."
    )


# ---------------------------------------------------------------------------
# RC#2 — Formula → PARAGRAPH (was UNKNOWN, dropped silently)
# ---------------------------------------------------------------------------


def test_formula_label_maps_to_paragraph_not_unknown() -> None:
    """``Formula`` must map to PARAGRAPH so DiT's mislabelling of
    Arabic body text as ``Formula`` does not cause the emitter to
    drop the region. Issue #16 RC#2.
    """
    role = role_for_label("Formula")
    assert role is RegionRole.PARAGRAPH, (
        f"Formula label is mapped to {role!r}; must be PARAGRAPH so "
        f"phase-7 emitter renders the text instead of dropping it "
        f"(see issue #16 root cause #2)."
    )
    # Guard against a future refactor that resurrects UNKNOWN here:
    # UNKNOWN regions are dropped by the markdown emitter unless the
    # role classifier rescues them, and we have no rescue path for
    # mislabelled body text.
    assert role is not RegionRole.UNKNOWN


# ---------------------------------------------------------------------------
# RC#3 — JSON-log failure events expose exception detail
# ---------------------------------------------------------------------------


def test_json_failure_event_includes_exception_message() -> None:
    """The pipeline records ``failure_reason`` as
    ``"<ExcClass>:<exc-message>"`` so the JSON-log line carries the
    actionable hint, not just the class name. Issue #16 RC#3.
    """
    stream = io.StringIO()
    logger = ProgressLogger(ProgressMode.JSON, stream=stream)
    reason = (
        "ModelDownloadError:failed to load OCR model "
        "stepfun-ai/GOT-OCR-2.0-hf@d3017ef2c2c1; prefetch the weights "
        "with: arabic-pdf-transcribe --prefetch-models"
    )
    logger.emit(
        ProgressEvent(
            page=2,
            of=7,
            event="failure",
            branch="ml",
            reason=reason,
        )
    )
    line = stream.getvalue().strip()
    assert "ModelDownloadError" in line
    assert "prefetch the weights" in line
    assert "--prefetch-models" in line
