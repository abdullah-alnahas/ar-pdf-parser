"""Phase-4 page rasterisation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL")
import pypdfium2 as pdfium  # noqa: E402
from PIL.Image import Image as PILImage  # noqa: E402

from arabic_pdf_transcribe.layout._rasterise import (  # noqa: E402
    DEFAULT_DPI,
    rasterise_page,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pdfs" / "image-scan" / "scan-ar-1col.pdf"


def _open_first_page() -> pdfium.PdfPage:
    pdf = pdfium.PdfDocument(str(FIXTURE))
    return pdf[0]


def test_rasterise_returns_pil_rgb_image() -> None:
    page = _open_first_page()
    image = rasterise_page(page, dpi=72)
    assert isinstance(image, PILImage)
    assert image.mode == "RGB"
    # Letter at 72 DPI: 612 x 792.
    assert image.size == (612, 792)


def test_rasterise_scales_with_dpi() -> None:
    page = _open_first_page()
    low = rasterise_page(page, dpi=72)
    high = rasterise_page(page, dpi=144)
    assert high.size[0] >= 2 * low.size[0] - 2
    assert high.size[1] >= 2 * low.size[1] - 2


def test_rasterise_rejects_non_positive_dpi() -> None:
    page = _open_first_page()
    with pytest.raises(ValueError, match="dpi must be positive"):
        rasterise_page(page, dpi=0)
    with pytest.raises(ValueError, match="dpi must be positive"):
        rasterise_page(page, dpi=-100)


def test_default_dpi_matches_spec_target() -> None:
    """The phase-4 plan picks 200 DPI as the default; the spec quotes
    the same target for the ML branch's 8 GB peak-RSS budget."""
    assert DEFAULT_DPI == 200
