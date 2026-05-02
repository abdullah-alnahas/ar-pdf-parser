"""Top-level reading-order reconstruction.

Composes :mod:`._columns` (column detection) and :mod:`._rows`
(row banding + within-band ordering) into a single deterministic
pipeline:

1. Detect columns.
2. Bucket regions per column.
3. Within each column, group into row bands (top-down).
4. Within each band, order right-to-left when ``rtl`` else
   left-to-right.
5. Concatenate columns: right-to-left for RTL pages, left-to-right
   otherwise.

Pure: same input → same output. The function does not mutate input
regions; it returns a new list referencing the same Region instances
in their new order.
"""

from __future__ import annotations

from collections.abc import Sequence

from arabic_pdf_transcribe.order._columns import assign_to_columns, detect_columns
from arabic_pdf_transcribe.order._rows import group_into_row_bands, order_within_band
from arabic_pdf_transcribe.regions import Region


def reorder(
    regions: Sequence[Region],
    page_width: float,
    page_height: float,
    *,
    rtl: bool = True,
) -> list[Region]:
    """Reorder a flat list of regions into reading order.

    ``page_width`` / ``page_height`` are used by column detection and
    by the header/footer prune in the role classifier (a separate
    module). The reorderer itself does not need ``page_height`` but
    it is part of the public signature for symmetry with future
    extensions (page-shape-aware reordering).
    """
    if not regions:
        return []
    _ = page_height  # reserved for future use; documented in module docstring
    columns = detect_columns(regions, page_width=page_width)
    column_buckets = assign_to_columns(regions, columns)
    column_iter = reversed(column_buckets) if rtl else iter(column_buckets)
    ordered: list[Region] = []
    for bucket in column_iter:
        bands = group_into_row_bands(bucket)
        for band in bands:
            ordered.extend(order_within_band(band, rtl=rtl))
    return ordered
