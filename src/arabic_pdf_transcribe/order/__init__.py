"""Reading-order reconstruction package.

Phase 6 takes the flat :class:`Region` list produced by phase 2 (native)
or phases 4-5 (ML) and reorders it into logical reading order:

1. Detect columns (1, 2, or 3) via an X-projection histogram on bbox
   centres + edges.
2. Within each column, group regions into row bands (Y-overlap with
   tolerance ``0.5 * median_region_height``).
3. Within each band, sort right-to-left when ``rtl=True`` (the default
   for Arabic), left-to-right otherwise.
4. Concatenate columns from right to left for RTL pages, left to right
   for LTR.

The whole pipeline is pure: same input → same output, no global state,
no I/O. Unit-test friendly.
"""

from __future__ import annotations

from arabic_pdf_transcribe.order.reorder import reorder

__all__ = ["reorder"]
