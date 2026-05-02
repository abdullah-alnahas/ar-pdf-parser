"""Column detection via X-projection histogram on region bboxes.

A page can be 1-, 2-, or 3-column. We bucket bbox X-centres into a
fixed-width histogram and look for separated peaks: the gaps between
peaks are the inter-column gutters. The detector is conservative —
when it cannot tell, it falls back to a single column (the safe
no-op, downstream behaves as if there were no columns at all).

The rule:

* Single column when fewer than ``MIN_REGIONS_FOR_COLUMNS`` regions.
* Otherwise, build a histogram over ``BIN_COUNT`` bins of the page's
  X axis and find connected non-zero runs separated by zero-runs of
  width >= ``MIN_GUTTER_FRACTION * page_width``.
* Cap at 3 columns (deeper splits collapse to 3 — matches spec).

Returns the ordered list of column X-ranges left to right; the
caller decides whether to read them right-to-left for RTL.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arabic_pdf_transcribe.regions import Region

BIN_COUNT = 50
MIN_REGIONS_FOR_COLUMNS = 4
MIN_GUTTER_FRACTION = 0.04
MAX_COLUMNS = 3


@dataclass(frozen=True, slots=True)
class Column:
    """X-range covering a single detected column."""

    x0: float
    x1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def x_centre(self) -> float:
        return 0.5 * (self.x0 + self.x1)

    def contains_centre(self, region: Region) -> bool:
        cx = 0.5 * (region.bbox.x0 + region.bbox.x1)
        return self.x0 <= cx <= self.x1


def detect_columns(regions: Sequence[Region], page_width: float) -> list[Column]:
    """Return columns covering the page in left-to-right order.

    Always returns at least one column spanning the full page width
    (the safe single-column fallback).
    """
    if page_width <= 0:
        return [Column(0.0, max(1.0, page_width))]
    full_page = [Column(0.0, page_width)]
    if len(regions) < MIN_REGIONS_FOR_COLUMNS:
        return full_page

    histogram = [0] * BIN_COUNT
    bin_width = page_width / BIN_COUNT
    for region in regions:
        cx = 0.5 * (region.bbox.x0 + region.bbox.x1)
        if cx < 0 or cx > page_width:
            continue
        idx = min(BIN_COUNT - 1, int(cx / bin_width))
        histogram[idx] += 1

    runs = _connected_runs(histogram)
    if not runs:
        return full_page
    runs = _filter_short_gutters(runs, page_width=page_width, bin_width=bin_width)
    if len(runs) <= 1:
        return full_page
    if len(runs) > MAX_COLUMNS:
        # Collapse extra columns into the rightmost one — the spec's
        # cap-at-3 rule. The first MAX_COLUMNS-1 runs stay; everything
        # else fuses with the last one.
        head = runs[: MAX_COLUMNS - 1]
        tail_lo = min(run[0] for run in runs[MAX_COLUMNS - 1 :])
        tail_hi = max(run[1] for run in runs[MAX_COLUMNS - 1 :])
        runs = [*head, (tail_lo, tail_hi)]

    columns: list[Column] = []
    for lo_idx, hi_idx in runs:
        x0 = lo_idx * bin_width
        x1 = (hi_idx + 1) * bin_width
        columns.append(Column(x0=x0, x1=min(page_width, x1)))
    # Expand each column to fill its half of any inter-column gutter so
    # regions whose X-centre lands in the gutter still get assigned a
    # column. We share the gutter equally between neighbours.
    return _expand_to_cover_gutters(columns, page_width=page_width)


def _connected_runs(histogram: list[int]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, count in enumerate(histogram):
        if count > 0 and not in_run:
            start = i
            in_run = True
        elif count == 0 and in_run:
            runs.append((start, i - 1))
            in_run = False
    if in_run:
        runs.append((start, len(histogram) - 1))
    return runs


def _filter_short_gutters(
    runs: list[tuple[int, int]], *, page_width: float, bin_width: float
) -> list[tuple[int, int]]:
    """Merge runs separated by gutters narrower than the threshold.

    Two adjacent runs whose inter-run gap is below
    ``MIN_GUTTER_FRACTION * page_width`` count as a single column —
    they are most likely the same column with a short horizontal gap
    inside it.
    """
    if not runs:
        return runs
    threshold_bins = max(1, int(MIN_GUTTER_FRACTION * page_width / bin_width))
    merged: list[tuple[int, int]] = [runs[0]]
    for lo, hi in runs[1:]:
        prev_lo, prev_hi = merged[-1]
        gap = lo - prev_hi - 1
        if gap < threshold_bins:
            merged[-1] = (prev_lo, hi)
        else:
            merged.append((lo, hi))
    return merged


def _expand_to_cover_gutters(columns: list[Column], *, page_width: float) -> list[Column]:
    """Extend column ranges so every X coordinate is inside some column."""
    if not columns:
        return columns
    result: list[Column] = []
    for i, col in enumerate(columns):
        x0 = 0.0 if i == 0 else 0.5 * (columns[i - 1].x1 + col.x0)
        x1 = page_width if i == len(columns) - 1 else 0.5 * (col.x1 + columns[i + 1].x0)
        result.append(Column(x0=x0, x1=x1))
    return result


def assign_to_columns(regions: Sequence[Region], columns: Sequence[Column]) -> list[list[Region]]:
    """Bucket each region into the column whose centre it lands in."""
    if not columns:
        return [list(regions)]
    buckets: list[list[Region]] = [[] for _ in columns]
    for region in regions:
        cx = 0.5 * (region.bbox.x0 + region.bbox.x1)
        chosen = 0
        for idx, col in enumerate(columns):
            if col.x0 <= cx <= col.x1:
                chosen = idx
                break
            if cx > col.x1:
                chosen = idx
        buckets[chosen].append(region)
    return buckets
