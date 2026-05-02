"""Pillow-only structural cell detection for table regions.

The layout detector emits a ``Table`` bbox; this module turns that bbox
into a :class:`TableGrid` by finding ruled-line intersections inside the
crop. Real-world tables use either ruled lines or whitespace separation
— phase 4 covers the ruled-line case, the most common in Arabic
typesetting (newspapers, government forms, Quranic study editions). The
whitespace-only case returns ``None`` and the caller falls back to
one-cell-per-row coalescing because phase 6's role classifier handles
row-level prose just fine.

OpenCV is not a runtime dependency. The morphological operations
(erode, projection) are implemented directly over Pillow's pixel
buffers — no ``cv2`` import. Pillow is gated behind the ``[ml]`` extra,
which is the correct stratum for this code: it only runs after the
layout detector has flagged a region as a table, which only happens on
the ML branch.

The algorithm in plain language:

1. Crop the table bbox out of the page image, convert to grayscale.
2. Threshold to a binary "ink / background" image.
3. Erode along the row axis with a long horizontal window → keeps
   horizontal ruled lines, drops glyphs.
4. Erode along the column axis with a long vertical window → keeps
   vertical ruled lines.
5. Sum each filtered image along the orthogonal axis to find row /
   column projections. Local maxima above a confidence threshold are
   the ruled-line positions.
6. Pair adjacent positions to get cell rectangles. The grid is rejected
   (and the caller falls back to one-cell-per-row) if fewer than 2
   row-positions OR 2 col-positions are found.

Cell text is left empty: phase 5 OCRs each cell.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from arabic_pdf_transcribe.regions import BBox, TableCell, TableGrid

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Tuning: the projection threshold ratio is the fraction of the
# orthogonal-axis pixels that must be "ink" for a row to count as a
# ruled line. 0.30 picks up real ruled lines while ignoring word
# strokes that happen to align horizontally.
PROJECTION_RATIO = 0.30
# Kernel length is expressed as a fraction of the cropped table dim;
# 0.30 keeps horizontal lines >= 30% of the table width / vertical
# lines >= 30% of the table height.
KERNEL_FRACTION = 0.30
# Binary threshold on the grayscale image (0..255). Pages rasterised at
# 200 DPI tend to have ink near 0 and paper near 255; 200 is a safe
# generic threshold.
BINARY_THRESHOLD = 200
# Minimum ruled-line spacing (pixels) — below this we collapse adjacent
# detections to a single line. Prevents anti-aliasing from doubling
# every line. 3 px at 200 DPI is roughly 0.4 pt.
MIN_LINE_SPACING_PX = 3


def detect_table_cells(
    page_image: PILImage,
    table_bbox: BBox,
) -> TableGrid | None:
    """Return a :class:`TableGrid` covering ``table_bbox`` on ``page_image``.

    Returns ``None`` when no usable grid is found (caller falls back).
    The bbox is in pixel units relative to ``page_image``'s top-left
    origin.
    """
    crop_box = (
        max(0, int(table_bbox.x0)),
        max(0, int(table_bbox.y0)),
        min(page_image.width, int(table_bbox.x1)),
        min(page_image.height, int(table_bbox.y1)),
    )
    if crop_box[2] - crop_box[0] <= 1 or crop_box[3] - crop_box[1] <= 1:
        return None
    crop = page_image.crop(crop_box).convert("L")
    width, height = crop.size
    pixels = list(crop.getdata())  # type: ignore[arg-type]
    binary = _binarise(pixels)  # ink=1, paper=0
    h_kernel = max(3, _odd(int(width * KERNEL_FRACTION)))
    v_kernel = max(3, _odd(int(height * KERNEL_FRACTION)))
    horiz_lines = _erode_horizontal(binary, width, height, kernel=h_kernel)
    vert_lines = _erode_vertical(binary, width, height, kernel=v_kernel)
    row_positions = _row_projection_peaks(horiz_lines, width, height)
    col_positions = _col_projection_peaks(vert_lines, width, height)
    if len(row_positions) < 2 or len(col_positions) < 2:
        return None
    cells = _build_cells(
        row_positions=row_positions,
        col_positions=col_positions,
        offset_x=crop_box[0],
        offset_y=crop_box[1],
    )
    return TableGrid(rows=cells)


def _odd(n: int) -> int:
    """Round n up to the next odd >= 3."""
    return n if n % 2 == 1 else n + 1


def _binarise(pixels: list[int]) -> list[int]:
    """Threshold a flat L-mode pixel buffer to ink (1) / paper (0)."""
    return [1 if v <= BINARY_THRESHOLD else 0 for v in pixels]


def _erode_horizontal(binary: list[int], width: int, height: int, *, kernel: int) -> list[int]:
    """Min-filter along rows with an odd 1-D kernel (radius = kernel // 2).

    A pixel survives only when every neighbour within ``kernel`` along
    the row is also ink. That removes glyph strokes (short horizontal
    runs) while keeping ruled lines (long horizontal runs).
    """
    radius = kernel // 2
    out = [0] * (width * height)
    for y in range(height):
        row_start = y * width
        row = binary[row_start : row_start + width]
        for x in range(width):
            lo = x - radius if x - radius >= 0 else 0
            hi = x + radius + 1 if x + radius + 1 <= width else width
            window = row[lo:hi]
            out[row_start + x] = 1 if all(window) else 0
    return out


def _erode_vertical(binary: list[int], width: int, height: int, *, kernel: int) -> list[int]:
    """Min-filter along columns. Mirror of :func:`_erode_horizontal`."""
    radius = kernel // 2
    out = [0] * (width * height)
    for x in range(width):
        col = [binary[y * width + x] for y in range(height)]
        for y in range(height):
            lo = y - radius if y - radius >= 0 else 0
            hi = y + radius + 1 if y + radius + 1 <= height else height
            window = col[lo:hi]
            out[y * width + x] = 1 if all(window) else 0
    return out


def _row_projection_peaks(filtered: list[int], width: int, height: int) -> list[int]:
    """Return the row indices whose ink projection exceeds the threshold."""
    threshold = max(1, int(width * PROJECTION_RATIO))
    sums = [sum(filtered[y * width : (y + 1) * width]) for y in range(height)]
    return _collapse_runs([i for i, s in enumerate(sums) if s >= threshold])


def _col_projection_peaks(filtered: list[int], width: int, height: int) -> list[int]:
    """Return the column indices whose ink projection exceeds the threshold."""
    threshold = max(1, int(height * PROJECTION_RATIO))
    sums = [sum(filtered[y * width + x] for y in range(height)) for x in range(width)]
    return _collapse_runs([i for i, s in enumerate(sums) if s >= threshold])


def _collapse_runs(positions: list[int]) -> list[int]:
    """Collapse adjacent positions within ``MIN_LINE_SPACING_PX`` to their median.

    Anti-aliasing makes a single ruled line show up as 2-3 adjacent
    high-projection rows; this folds them back into one position.
    """
    if not positions:
        return []
    collapsed: list[int] = []
    run: list[int] = [positions[0]]
    for idx in positions[1:]:
        if idx - run[-1] <= MIN_LINE_SPACING_PX:
            run.append(idx)
        else:
            collapsed.append(run[len(run) // 2])
            run = [idx]
    collapsed.append(run[len(run) // 2])
    return collapsed


def _build_cells(
    *,
    row_positions: list[int],
    col_positions: list[int],
    offset_x: int,
    offset_y: int,
) -> tuple[tuple[TableCell, ...], ...]:
    """Pair adjacent row/col positions into cells with empty text."""
    rows: list[tuple[TableCell, ...]] = []
    for r in range(len(row_positions) - 1):
        y0, y1 = row_positions[r], row_positions[r + 1]
        row_cells: list[TableCell] = []
        for c in range(len(col_positions) - 1):
            x0, x1 = col_positions[c], col_positions[c + 1]
            cell = TableCell(
                text="",
                confidence=None,
                bbox=BBox(
                    x0=float(offset_x + x0),
                    y0=float(offset_y + y0),
                    x1=float(offset_x + x1),
                    y1=float(offset_y + y1),
                ),
            )
            row_cells.append(cell)
        rows.append(tuple(row_cells))
    return tuple(rows)
