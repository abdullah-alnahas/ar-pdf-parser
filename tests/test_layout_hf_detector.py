"""Phase-4 HF DiT layout-detector adapter tests.

The adapter wraps a Hugging Face semantic-segmentation model. We do
not actually load the real weights in CI: the tests stub
``transformers.AutoImageProcessor`` and
``transformers.AutoModelForSemanticSegmentation`` so the adapter
exercises its end-to-end flow (segmentation map → connected
components → bbox → ``Region``) against a deterministic fake output.

A separate ``@pytest.mark.slow`` test loads the real model and runs
it on the phase-4 image-scan fixture; it is skipped by default.
"""

from __future__ import annotations

import sys
import types

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arabic_pdf_transcribe.errors import ModelDownloadError  # noqa: E402
from arabic_pdf_transcribe.layout import LayoutDetector  # noqa: E402
from arabic_pdf_transcribe.layout.hf_detector import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    HFDiTLayoutDetector,
    HFLayoutDetectorConfig,
)
from arabic_pdf_transcribe.regions import Region, RegionRole, RegionSource  # noqa: E402

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeProcessor:
    def __call__(self, *, images: object, return_tensors: str) -> dict[str, object]:
        return {"pixel_values": object()}


class _FakeOutputs:
    def __init__(self, logits: object) -> None:
        self.logits = logits


class _FakeConfig:
    def __init__(self, id2label: dict[int, str]) -> None:
        self.id2label = {str(k): v for k, v in id2label.items()}


class _FakeModel:
    def __init__(self, logits: object, id2label: dict[int, str]) -> None:
        self._logits = logits
        self.config = _FakeConfig(id2label)

    def __call__(self, **kwargs: object) -> _FakeOutputs:
        return _FakeOutputs(self._logits)


def _build_logits(class_grid: list[list[int]], num_classes: int) -> object:
    """Return a tensor of shape ``(1, num_classes, H, W)`` whose argmax
    along ``dim=1`` matches ``class_grid`` and whose softmax probs
    favour the chosen class."""
    import torch

    h = len(class_grid)
    w = len(class_grid[0])
    logits = torch.zeros((1, num_classes, h, w))
    for y in range(h):
        for x in range(w):
            logits[0, class_grid[y][x], y, x] = 5.0
    return logits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_config_pins_apache_licensed_dit() -> None:
    """The defaults must match the audited models.toml entry."""
    cfg = HFLayoutDetectorConfig()
    assert cfg.model == DEFAULT_MODEL
    assert cfg.revision == DEFAULT_REVISION
    assert DEFAULT_MODEL == "cmarkea/dit-base-layout-detection"
    # Revision is a 40-char SHA.
    assert len(DEFAULT_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in DEFAULT_REVISION)


def test_detector_satisfies_layout_detector_protocol() -> None:
    detector = HFDiTLayoutDetector()
    assert isinstance(detector, LayoutDetector)


def test_detect_with_stub_model_returns_expected_regions() -> None:
    """Mocked detector: a 6x6 class-id grid with two regions (Title,
    Text) becomes two Regions of the right role."""
    pytest.importorskip("torch")

    id2label = {
        0: "Background",
        1: "Title",
        2: "Text",
    }
    # Top-left 2x4 → Title; bottom-right 2x4 → Text.
    class_grid = [
        [1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 2, 2, 2, 2],
        [0, 0, 2, 2, 2, 2],
    ]
    logits = _build_logits(class_grid, num_classes=3)

    detector = HFDiTLayoutDetector(
        HFLayoutDetectorConfig(min_region_area_px=1, detect_table_cells=False)
    )
    detector._processor = _FakeProcessor()
    detector._model = _FakeModel(logits, id2label)
    detector._id2label = id2label

    image = Image.new("RGB", (60, 60), color=(255, 255, 255))
    regions = list(detector.detect(image, page_index=7))

    assert len(regions) == 2
    roles = {r.role for r in regions}
    assert roles == {RegionRole.HEADING, RegionRole.PARAGRAPH}
    title_region = next(r for r in regions if r.role is RegionRole.HEADING)
    assert title_region.heading_level == 1
    assert title_region.source is RegionSource.OCR
    assert title_region.text == ""
    assert title_region.page_index == 7
    assert title_region.confidence is not None
    assert title_region.confidence > 0.5


def test_detect_with_stub_model_skips_background_pixels() -> None:
    """All-Background grid yields zero regions."""
    pytest.importorskip("torch")
    id2label = {0: "Background"}
    class_grid = [[0] * 4 for _ in range(4)]
    logits = _build_logits(class_grid, num_classes=1)

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig(min_region_area_px=1))
    detector._processor = _FakeProcessor()
    detector._model = _FakeModel(logits, id2label)
    detector._id2label = id2label

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    regions = list(detector.detect(image, page_index=0))
    assert regions == []


def test_detect_filters_subthreshold_area() -> None:
    """A region smaller than ``min_region_area_px`` (in input-image
    pixels) is dropped."""
    pytest.importorskip("torch")
    id2label = {0: "Background", 1: "Text"}
    # One single-pixel Text. After scale-up to a 100x100 input image
    # this is ~ (100/4)^2 = 625 px area. Set the threshold above that
    # and the region drops.
    class_grid = [
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    logits = _build_logits(class_grid, num_classes=2)

    detector = HFDiTLayoutDetector(HFLayoutDetectorConfig(min_region_area_px=10000))
    detector._processor = _FakeProcessor()
    detector._model = _FakeModel(logits, id2label)
    detector._id2label = id2label

    image = Image.new("RGB", (100, 100), color=(255, 255, 255))
    regions = list(detector.detect(image, page_index=0))
    assert regions == []


def test_detect_table_region_always_has_populated_table_grid() -> None:
    """Plan contract: every TABLE Region must carry a populated
    ``table_grid``. When ruled-line morphology cannot recover a grid
    (blank crop, no ruled lines), the adapter falls back to a
    one-cell-per-row coalescing — never ``None``.
    """
    pytest.importorskip("torch")
    id2label = {0: "Background", 1: "Table"}
    # 4x4 Table block. The blank-page crop has no ruled lines so
    # detect_table_cells returns None — the adapter must fill the
    # fallback grid covering the whole table bbox.
    class_grid = [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    logits = _build_logits(class_grid, num_classes=2)

    detector = HFDiTLayoutDetector(
        HFLayoutDetectorConfig(min_region_area_px=1, detect_table_cells=True)
    )
    detector._processor = _FakeProcessor()
    detector._model = _FakeModel(logits, id2label)
    detector._id2label = id2label

    image = Image.new("RGB", (60, 60), color=(255, 255, 255))
    regions = list(detector.detect(image, page_index=0))
    assert len(regions) == 1
    table_region = regions[0]
    assert table_region.role is RegionRole.TABLE
    assert table_region.table_grid is not None
    # Fallback grid: one row, one cell, bbox matches the table region.
    assert table_region.table_grid.n_rows == 1
    assert table_region.table_grid.n_cols == 1
    only_cell = table_region.table_grid.rows[0][0]
    assert only_cell.text == ""
    assert only_cell.bbox == table_region.bbox


def test_detect_table_region_disabled_cell_detection_still_has_fallback_grid() -> None:
    """Even when ``detect_table_cells=False``, the TABLE Region must
    carry the one-cell-per-row fallback grid — the contract is the
    Region's, not the morphology helper's.
    """
    pytest.importorskip("torch")
    id2label = {0: "Background", 1: "Table"}
    class_grid = [
        [0, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
    ]
    logits = _build_logits(class_grid, num_classes=2)

    detector = HFDiTLayoutDetector(
        HFLayoutDetectorConfig(min_region_area_px=1, detect_table_cells=False)
    )
    detector._processor = _FakeProcessor()
    detector._model = _FakeModel(logits, id2label)
    detector._id2label = id2label

    image = Image.new("RGB", (40, 40), color=(255, 255, 255))
    regions = list(detector.detect(image, page_index=0))
    assert len(regions) == 1
    assert regions[0].role is RegionRole.TABLE
    assert regions[0].table_grid is not None
    assert regions[0].table_grid.n_rows == 1
    assert regions[0].table_grid.n_cols == 1


def test_ensure_loaded_raises_model_download_error_on_transformers_missing() -> None:
    """Simulate ``transformers`` not being installed.

    We patch ``sys.modules`` so the local import inside
    ``_ensure_loaded`` raises ImportError; the adapter must convert
    that into :class:`ModelDownloadError` (CLI exit 5).
    """
    detector = HFDiTLayoutDetector()
    sentinel = "transformers"
    saved = sys.modules.pop(sentinel, None)
    # Pre-install a sentinel that raises ImportError on access by
    # nested attribute (the from-import).
    fake_module = types.ModuleType("transformers")
    sys.modules[sentinel] = fake_module
    try:
        with pytest.raises(ModelDownloadError):
            detector._ensure_loaded()
    finally:
        if saved is not None:
            sys.modules[sentinel] = saved
        else:
            sys.modules.pop(sentinel, None)


@pytest.mark.slow
def test_real_model_loads_and_detects_on_image_scan_fixture() -> None:
    """Optional real-model run; off by default in CI.

    Run locally via ``pytest -m slow``. PR author attaches the output
    to the PR description.
    """
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    from pathlib import Path

    fixture = (
        Path(__file__).resolve().parent / "fixtures" / "pdfs" / "image-scan" / "scan-ar-1col.pdf"
    )
    if not fixture.exists():
        pytest.skip("phase-4 image-scan fixture missing; run tools/generate_fixtures.py")

    import pypdfium2 as pdfium

    from arabic_pdf_transcribe.layout._rasterise import rasterise_page

    pdf = pdfium.PdfDocument(str(fixture))
    page = pdf[0]
    image = rasterise_page(page, dpi=150)

    detector = HFDiTLayoutDetector()
    regions = list(detector.detect(image, page_index=0))

    assert all(isinstance(r, Region) for r in regions)
    assert all(r.source is RegionSource.OCR for r in regions)
    assert all(r.text == "" for r in regions)
