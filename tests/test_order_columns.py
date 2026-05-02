"""Phase-6 column-detection tests.

Tests the X-projection histogram column detector on synthetic
region lists for 1, 2, and 3-column layouts.
"""

from __future__ import annotations

from arabic_pdf_transcribe.order._columns import (
    BIN_COUNT,
    MAX_COLUMNS,
    MIN_REGIONS_FOR_COLUMNS,
    Column,
    assign_to_columns,
    detect_columns,
)
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


def _region(x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(
        page_index=0,
        bbox=BBox(x0, y0, x1, y1),
        text="x",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.NATIVE,
    )


def test_no_regions_returns_single_full_page_column() -> None:
    cols = detect_columns([], page_width=600.0)
    assert len(cols) == 1
    assert cols[0].x0 == 0.0
    assert cols[0].x1 == 600.0


def test_few_regions_falls_back_to_single_column() -> None:
    """Below ``MIN_REGIONS_FOR_COLUMNS`` we always emit one column."""
    regions = [_region(0, 0, 200, 50)]
    cols = detect_columns(regions, page_width=600.0)
    assert len(cols) == 1


def test_zero_page_width_returns_safe_fallback() -> None:
    cols = detect_columns([_region(0, 0, 100, 50)], page_width=0.0)
    assert len(cols) == 1


def test_two_column_layout_detected() -> None:
    """Regions packed into two horizontal halves split into two columns."""
    regions: list[Region] = []
    # Left column: x in 50..250
    for y in range(0, 500, 50):
        regions.append(_region(50, y, 250, y + 30))
    # Right column: x in 350..550
    for y in range(0, 500, 50):
        regions.append(_region(350, y, 550, y + 30))
    cols = detect_columns(regions, page_width=600.0)
    assert len(cols) == 2
    # Left column should be left of right column.
    assert cols[0].x_centre < cols[1].x_centre


def test_three_column_layout_detected() -> None:
    regions: list[Region] = []
    for x_centre in (100, 300, 500):
        for y in range(0, 500, 50):
            regions.append(_region(x_centre - 50, y, x_centre + 50, y + 30))
    cols = detect_columns(regions, page_width=600.0)
    assert len(cols) == 3
    assert cols[0].x_centre < cols[1].x_centre < cols[2].x_centre


def test_more_than_three_columns_collapses_to_three() -> None:
    """Spec rule: cap at 3 columns."""
    regions: list[Region] = []
    for x_centre in (60, 180, 300, 420, 540):
        for y in range(0, 200, 50):
            regions.append(_region(x_centre - 20, y, x_centre + 20, y + 30))
    cols = detect_columns(regions, page_width=600.0)
    assert len(cols) == MAX_COLUMNS


def test_short_gutters_below_min_fraction_collapse() -> None:
    """Two clusters whose centroids land in adjacent bins (gap < threshold) fuse.

    The detector measures gaps in the histogram of bbox X-centres,
    not raw edge-to-edge distance. We construct a layout where the
    centres lie close enough that the inter-bin gap is below the
    short-gutter threshold (``MIN_GUTTER_FRACTION``) so the two
    runs merge into a single column.
    """
    # Page width 600 → bin width 12 → threshold = ~2 bins (24px).
    # Centres at 150 and 165 lie in adjacent bins (12 and 13). No
    # zero-bin between them; even if there were, 1 < 2 → merge.
    regions: list[Region] = []
    for y in range(0, 500, 50):
        regions.append(_region(125, y, 175, y + 30))  # centre 150
        regions.append(_region(140, y, 190, y + 30))  # centre 165
    cols = detect_columns(regions, page_width=600.0)
    assert len(cols) == 1


def test_column_contains_centre() -> None:
    col = Column(0.0, 100.0)
    assert col.contains_centre(_region(20, 0, 60, 10))
    assert not col.contains_centre(_region(150, 0, 200, 10))


def test_assign_to_columns_buckets_by_centre() -> None:
    cols = [Column(0.0, 300.0), Column(300.0, 600.0)]
    regions = [
        _region(50, 0, 100, 10),
        _region(400, 0, 500, 10),
        _region(20, 50, 60, 60),
    ]
    buckets = assign_to_columns(regions, cols)
    assert len(buckets[0]) == 2
    assert len(buckets[1]) == 1


def test_min_regions_constant() -> None:
    """Sanity: tuning constants match the documented values."""
    assert MIN_REGIONS_FOR_COLUMNS == 4
    assert MAX_COLUMNS == 3
    assert BIN_COUNT == 50
