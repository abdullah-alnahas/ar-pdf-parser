"""Row-band assignment within a column.

Within a single column, a "row band" is a horizontal strip whose
height is at most ``ROW_BAND_TOLERANCE_FACTOR * median_region_height``.
Two regions in the same band are read as a single line, ordered
right-to-left for RTL or left-to-right for LTR.

Implementation walks regions in input (stream) order and bins each
into the most recent band whose top-edge differs from the new
region's top-edge by at most the band tolerance. New regions whose
top-edge falls outside the existing band's tolerance start a fresh
band.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import median

from arabic_pdf_transcribe.regions import Region

ROW_BAND_TOLERANCE_FACTOR = 0.5


def group_into_row_bands(regions: Sequence[Region]) -> list[list[Region]]:
    """Return regions grouped into top-down row bands.

    Each band is a list of regions whose Y-extents overlap within the
    tolerance. Bands are returned top-to-bottom (smallest y0 first
    when the bbox is in top-left-origin coordinates).
    """
    if not regions:
        return []
    heights = [r.bbox.height for r in regions if r.bbox.height > 0]
    median_height = median(heights) if heights else 1.0
    tolerance = max(1.0, ROW_BAND_TOLERANCE_FACTOR * median_height)

    sorted_by_y = sorted(regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
    bands: list[list[Region]] = []
    current_band_y0: float | None = None
    for region in sorted_by_y:
        if current_band_y0 is None or region.bbox.y0 - current_band_y0 > tolerance:
            bands.append([region])
            current_band_y0 = region.bbox.y0
        else:
            bands[-1].append(region)
    return bands


def order_within_band(band: list[Region], *, rtl: bool) -> list[Region]:
    """Order a single row band's regions.

    For RTL pages, sort by X-right (largest first) so the rightmost
    region comes first. For LTR pages, sort by X-left (smallest
    first). Tie-break by Y-top so multiple regions at the same X
    stay in stream order.
    """
    if rtl:
        return sorted(band, key=lambda r: (-r.bbox.x1, r.bbox.y0))
    return sorted(band, key=lambda r: (r.bbox.x0, r.bbox.y0))
