"""Phase-6 row-banding + within-band ordering tests."""

from __future__ import annotations

from arabic_pdf_transcribe.order._rows import (
    ROW_BAND_TOLERANCE_FACTOR,
    group_into_row_bands,
    order_within_band,
)
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


def _region(x0: float, y0: float, x1: float, y1: float, *, text: str = "x") -> Region:
    return Region(
        page_index=0,
        bbox=BBox(x0, y0, x1, y1),
        text=text,
        role=RegionRole.PARAGRAPH,
        source=RegionSource.NATIVE,
    )


def test_empty_returns_empty() -> None:
    assert group_into_row_bands([]) == []


def test_single_row_band_groups_aligned_regions() -> None:
    regions = [
        _region(0, 100, 50, 130),
        _region(60, 100, 110, 130),
        _region(120, 100, 170, 130),
    ]
    bands = group_into_row_bands(regions)
    assert len(bands) == 1
    assert len(bands[0]) == 3


def test_distinct_row_bands_separated_by_height() -> None:
    regions = [
        _region(0, 100, 50, 130),
        _region(0, 200, 50, 230),
        _region(0, 300, 50, 330),
    ]
    bands = group_into_row_bands(regions)
    assert len(bands) == 3


def test_within_tolerance_y_diffs_stay_together() -> None:
    """Regions with sub-pixel Y differences should bin into one band."""
    regions = [
        _region(0, 100, 50, 130),
        _region(60, 102, 110, 132),  # 2px lower
        _region(120, 99, 170, 129),  # 1px higher
    ]
    bands = group_into_row_bands(regions)
    # All three regions span the same band.
    assert len(bands) == 1


def test_order_within_band_rtl_sorts_rightmost_first() -> None:
    band = [
        _region(50, 100, 100, 130, text="L"),
        _region(200, 100, 250, 130, text="R"),
        _region(125, 100, 175, 130, text="C"),
    ]
    ordered = order_within_band(band, rtl=True)
    assert [r.text for r in ordered] == ["R", "C", "L"]


def test_order_within_band_ltr_sorts_leftmost_first() -> None:
    band = [
        _region(200, 100, 250, 130, text="R"),
        _region(50, 100, 100, 130, text="L"),
        _region(125, 100, 175, 130, text="C"),
    ]
    ordered = order_within_band(band, rtl=False)
    assert [r.text for r in ordered] == ["L", "C", "R"]


def test_tolerance_constant_matches_spec() -> None:
    assert ROW_BAND_TOLERANCE_FACTOR == 0.5
