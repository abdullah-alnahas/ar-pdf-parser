"""Crop a Region's bbox out of a page image with configurable padding.

The OCR adapter expects an RGB PIL image of the region in its
expected orientation (no rotation in v1; phase 9 may add deskew).
This module's only job is to slice the bbox out of the rasterised
page with a small padding so glyphs at the bbox edge are not
clipped.

Default padding is 4 px — a balance between picking up the trailing
edge of glyphs (descenders, diacritics) and not pulling in adjacent
regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from arabic_pdf_transcribe.regions import BBox

DEFAULT_PADDING_PX = 4


def crop_region(
    page_image: PILImage,
    bbox: BBox,
    *,
    padding_px: int = DEFAULT_PADDING_PX,
) -> PILImage:
    """Return the bbox crop of ``page_image`` with ``padding_px`` margin.

    The bbox is in pixel coordinates (top-left origin). Coordinates
    that fall outside the page image are clipped to the page bounds.
    The crop is converted to RGB (the OCR adapter's expected input).
    """
    if padding_px < 0:
        raise ValueError(f"padding_px must be non-negative, got {padding_px}")
    x0 = max(0, int(bbox.x0) - padding_px)
    y0 = max(0, int(bbox.y0) - padding_px)
    x1 = min(page_image.width, int(bbox.x1) + padding_px)
    y1 = min(page_image.height, int(bbox.y1) + padding_px)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox {bbox!r} produces a degenerate crop after clipping to page bounds")
    crop = page_image.crop((x0, y0, x1, y1))
    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    return crop
