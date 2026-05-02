"""Phase-5 region-cropping tests."""

from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from arabic_pdf_transcribe.ocr._crop import DEFAULT_PADDING_PX, crop_region  # noqa: E402
from arabic_pdf_transcribe.regions import BBox  # noqa: E402


def test_crop_returns_rgb_with_padding() -> None:
    img = Image.new("RGB", (200, 200), color=(128, 128, 128))
    bbox = BBox(50.0, 60.0, 150.0, 160.0)
    crop = crop_region(img, bbox, padding_px=4)
    assert crop.mode == "RGB"
    # 100x100 bbox + 4px padding on every side = 108x108.
    assert crop.size == (108, 108)


def test_crop_clips_to_page_bounds() -> None:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    bbox = BBox(-50.0, -50.0, 150.0, 150.0)
    crop = crop_region(img, bbox, padding_px=4)
    # Bbox clipped to (0, 0, 100, 100).
    assert crop.size == (100, 100)


def test_crop_rejects_negative_padding() -> None:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    with pytest.raises(ValueError, match="padding_px must be non-negative"):
        crop_region(img, BBox(0.0, 0.0, 50.0, 50.0), padding_px=-1)


def test_crop_rejects_degenerate_bbox() -> None:
    img = Image.new("RGB", (100, 100), color=(255, 255, 255))
    # Bbox entirely outside image; clipping yields zero-size crop.
    with pytest.raises(ValueError, match="degenerate crop"):
        crop_region(img, BBox(200.0, 200.0, 300.0, 300.0), padding_px=0)


def test_default_padding() -> None:
    assert DEFAULT_PADDING_PX == 4


def test_crop_converts_non_rgb_input() -> None:
    img = Image.new("L", (100, 100), color=128)
    bbox = BBox(10.0, 10.0, 50.0, 50.0)
    crop = crop_region(img, bbox, padding_px=0)
    assert crop.mode == "RGB"
