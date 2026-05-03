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
   :data:`RegionRole.UNKNOWN`. The text was not dropped (the
   markdown emitter renders both UNKNOWN and PARAGRAPH via the same
   paragraph renderer) but the hf_detector's "label X mapped to
   UNKNOWN" warning fired on every Arabic body-text region the
   English-trained DiT mislabelled as Formula — 39 such warnings on
   a single page in the bug report. The fix remaps
   ``Formula → PARAGRAPH``: semantically more accurate (it is body
   text, not unknown content) and silences the spurious noise.

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


def test_pyproject_pins_transformers_supporting_default_ocr_model() -> None:
    """The ``[ml]`` extra must pin a transformers release that knows
    the default OCR model's type. Issue #26 swapped the default to
    ``Qwen/Qwen2-VL-2B-Instruct`` (model_type ``qwen2_vl``); Qwen2-VL
    support landed in transformers 4.45.0. The original #16 RC pinned
    ``>=4.49`` for ``got_ocr2``; that pin still satisfies qwen2_vl.

    Accepts either an exact pin (``==4.49.0``, the project convention)
    or a range with a ``>=4.45`` lower bound.
    """
    text = PYPROJECT.read_text(encoding="utf-8")
    matches = re.findall(r'"transformers([^"]*)"', text)
    assert matches, "no transformers requirement found in pyproject.toml"
    spec = matches[0]
    bound = re.search(r"(?:>=|==)\s*4\.(\d+)", spec)
    assert bound is not None, (
        f"transformers spec {spec!r} must declare a >=4.45 or ==4.45+ pin "
        f"(Qwen2-VL support landed in 4.45.0; see issue #26)"
    )
    minor = int(bound.group(1))
    assert minor >= 45, (
        f"transformers spec {spec!r} pins minor {minor} < 45; "
        f"Qwen2-VL (model_type 'qwen2_vl') is unrecognised below 4.45.0"
    )


def test_default_ocr_model_type_is_registered_in_auto_image_text_to_text() -> None:
    """``AutoModelForImageTextToText`` must know about ``qwen2_vl``.

    Issue #26: the default OCR model is now Qwen2-VL-2B-Instruct.
    The auto-class mapping is populated at import time from a
    name-only registry (``MODEL_FOR_*_MAPPING_NAMES``), so this check
    exercises the exact failing API path with zero network or weight
    I/O. Skipped when the ``[ml]`` extra is absent.
    """
    pytest.importorskip("transformers")
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    mapping: dict[str, str] = {
        str(k): str(v) for k, v in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.items()
    }
    assert "qwen2_vl" in mapping, (
        "AutoModelForImageTextToText has no entry for model_type "
        "'qwen2_vl'; transformers is too old (<4.45). This is the "
        "exact API path that exploded on every ML-branch page in #16."
    )
    assert mapping["qwen2_vl"] == "Qwen2VLForConditionalGeneration", (
        f"unexpected qwen2_vl mapping target {mapping['qwen2_vl']!r}; "
        f"upstream may have renamed the class — check OCR adapter."
    )


def test_default_ocr_model_config_instantiates_when_transformers_installed() -> None:
    """If the default OCR model's config is in the local HF cache,
    ``AutoConfig`` must instantiate it. Network-free. Skipped when
    transformers is absent or the config is not cached (CI without
    prefetched cache).
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
        pytest.skip(f"OCR model config not cached locally ({type(exc).__name__})")
    assert cfg.model_type == "qwen2_vl", (
        f"AutoConfig returned model_type={cfg.model_type!r}; "
        f"expected 'qwen2_vl'. transformers may be too old (<4.45)."
    )


# ---------------------------------------------------------------------------
# RC#2 — Formula → PARAGRAPH (was UNKNOWN, dropped silently)
# ---------------------------------------------------------------------------


def test_formula_label_maps_to_paragraph_not_unknown() -> None:
    """``Formula`` must map to PARAGRAPH.

    The DiT layout model frequently mislabels Arabic body text as
    ``Formula``. The previous mapping to UNKNOWN did not drop the
    text — both UNKNOWN and PARAGRAPH render through the markdown
    emitter's paragraph path — but it triggered the hf_detector's
    "label X mapped to UNKNOWN" warning on every mislabelled region,
    flooding the log with spurious noise (39 warnings on a single
    page in #16). The remap silences this and is semantically more
    accurate. See issue #16 root cause #2.
    """
    role = role_for_label("Formula")
    assert role is RegionRole.PARAGRAPH, (
        f"Formula label is mapped to {role!r}; must be PARAGRAPH so "
        f"the hf_detector does not log a spurious UNKNOWN warning on "
        f"every mislabelled Arabic body-text region (see issue #16)."
    )
    assert role is not RegionRole.UNKNOWN


def test_formula_label_does_not_log_unknown_warning_on_detection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end check: a stub layout model that emits a ``Formula``
    label must not produce the hf_detector's "mapped to UNKNOWN"
    warning, and the resulting Region must render as paragraph text
    in the markdown emitter. This is the user-visible artefact of
    issue #16's root cause #2.
    """
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    import logging

    # Inline minimal stubs (mirrors the patterns in
    # ``tests/test_layout_hf_detector.py``).
    import torch
    from PIL import Image

    from arabic_pdf_transcribe.emit.markdown import emit_markdown
    from arabic_pdf_transcribe.layout.hf_detector import (
        HFDiTLayoutDetector,
        HFLayoutDetectorConfig,
    )

    class _Outputs:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class _Config:
        def __init__(self, id2label: dict[int, str]) -> None:
            self.id2label = {str(k): v for k, v in id2label.items()}

    class _Model:
        def __init__(self, logits: torch.Tensor, id2label: dict[int, str]) -> None:
            self._logits = logits
            self.config = _Config(id2label)

        def __call__(self, **_: object) -> _Outputs:
            return _Outputs(self._logits)

    class _Processor:
        def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
            return {"pixel_values": object()}

    id2label = {0: "Background", 1: "Formula"}
    # 4×4 grid: full coverage by class 1 ("Formula").
    logits = torch.zeros((1, 2, 4, 4))
    logits[0, 1, :, :] = 5.0

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig())
    detector._processor = _Processor()  # type: ignore[attr-defined]
    detector._model = _Model(logits, id2label)  # type: ignore[attr-defined]
    detector._id2label = id2label  # type: ignore[attr-defined]

    page_image = Image.new("RGB", (256, 256), color="white")
    fake_text = "نص عربي تجريبي"  # arbitrary non-empty Arabic text

    with caplog.at_level(logging.WARNING, logger="arabic_pdf_transcribe.layout.hf_detector"):
        regions = list(detector.detect(page_image, page_index=0))
    unknown_warnings = [r for r in caplog.records if "mapped to UNKNOWN" in r.getMessage()]
    assert not unknown_warnings, (
        f"Formula label must not log 'mapped to UNKNOWN' warnings; "
        f"got {len(unknown_warnings)} on a single page (issue #16)."
    )
    assert regions, "expected at least one Region from the stub Formula segmentation"
    assert all(r.role is RegionRole.PARAGRAPH for r in regions), (
        f"expected all regions to carry RegionRole.PARAGRAPH; " f"got {[r.role for r in regions]}"
    )

    # Render-side: a Formula-labelled region with body text MUST appear
    # in the markdown output. Synthesise a region that mimics a fully
    # transcribed Formula→PARAGRAPH region (the real OCR step happens
    # in a separate adapter and is unit-tested elsewhere).
    transcribed = regions[0].with_text(fake_text)
    markdown = emit_markdown([transcribed])
    assert fake_text in markdown, (
        "transcribed text from a Formula-labelled region must reach "
        "the markdown output (issue #16 contract)."
    )


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
