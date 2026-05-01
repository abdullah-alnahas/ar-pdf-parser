"""Basic native-table detection.

Phase 2 only handles the easy case: a page whose native text lines form a
regular grid — every line splits cleanly into the same number of cells when
broken on large horizontal gaps. When the heuristic does not find a clean
grid we return ``None`` and the page falls through to the heuristic / ML
table path that phase 4 / phase 6 deliver.

The detector is deliberately conservative — false positives (prose lines
mistaken for table rows) are far more painful downstream than false
negatives. Acceptance criteria (phase 2):

- a synthetic 3 by 3 grid is detected as a table,
- prose paragraphs are NOT detected as a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from arabic_pdf_transcribe.regions import BBox, TableCell, TableGrid

# Minimum number of consecutive same-shape rows required before we accept a
# region as a table. Two would let any paragraph with line-internal tabs be
# interpreted as a 2-row table; three is the smallest count that's
# meaningful as a "grid".
MIN_ROWS = 3
MIN_COLS = 2


@dataclass(frozen=True, slots=True)
class _Word:
    """Internal: a single word with its bbox."""

    bbox: BBox
    text: str


@dataclass(frozen=True, slots=True)
class _Line:
    """Internal: words ordered left-to-right plus the line's outer bbox."""

    bbox: BBox
    words: tuple[_Word, ...]


def detect_table(lines: Sequence[_Line]) -> tuple[TableGrid, list[int]] | None:
    """Find the largest contiguous run of lines that forms a regular grid.

    Returns ``(grid, indices)`` where ``indices`` lists which input lines
    were consumed by the table; or ``None`` if no qualifying run is found.

    A line splits into "cells" on horizontal gaps that are at least as wide
    as the previous word's bounding box. The longest contiguous run where
    every line yields the same cell count >= :data:`MIN_COLS` and the run
    length is >= :data:`MIN_ROWS` becomes the table. A final column-X
    uniformity check rejects prose that happens to share line lengths.
    """
    if len(lines) < MIN_ROWS:
        return None

    cells_per_line: list[tuple[TableCell, ...]] = [_split_line_into_cells(line) for line in lines]

    best_run: tuple[int, int] | None = None  # (start, length)
    expected: int | None = None
    run_start = 0
    for index, cells in enumerate(cells_per_line):
        n = len(cells)
        if n >= MIN_COLS and (expected is None or expected == n):
            if expected is None:
                expected = n
                run_start = index
        else:
            run_len = index - run_start
            if (
                expected is not None
                and run_len >= MIN_ROWS
                and (best_run is None or run_len > best_run[1])
            ):
                best_run = (run_start, run_len)
            expected = None
            if n >= MIN_COLS:
                expected = n
                run_start = index
    if expected is not None:
        run_len = len(cells_per_line) - run_start
        if run_len >= MIN_ROWS and (best_run is None or run_len > best_run[1]):
            best_run = (run_start, run_len)

    if best_run is None:
        return None

    start, length = best_run
    rows = tuple(cells_per_line[start : start + length])
    if not _columns_aligned(rows):
        return None
    return TableGrid(rows=rows), list(range(start, start + length))


def _split_line_into_cells(line: _Line) -> tuple[TableCell, ...]:
    """Group adjacent words into cells, splitting on large horizontal gaps.

    Threshold rule: a gap between two successive words (by absolute
    horizontal distance — RTL stream order has the next word to the *left*
    of the previous one) counts as a cell boundary if it is at least as
    wide as the previous word's bounding box. Word-internal whitespace is
    much narrower than a word's width; column whitespace in a typical
    table is much wider. The rule needs no page-wide tuning because it is
    self-relative per gap.
    """
    if not line.words:
        return ()
    cells: list[TableCell] = []
    current: list[_Word] = [line.words[0]]
    for word in line.words[1:]:
        prev = current[-1]
        # Absolute distance between the bbox edges that face each other.
        if word.bbox.x0 >= prev.bbox.x1:
            gap = word.bbox.x0 - prev.bbox.x1
        elif prev.bbox.x0 >= word.bbox.x1:
            gap = prev.bbox.x0 - word.bbox.x1
        else:
            gap = 0.0
        threshold = max(prev.bbox.width, 1.0)
        if gap >= threshold:
            cells.append(_cell_from_words(current))
            current = [word]
        else:
            current.append(word)
    if current:
        cells.append(_cell_from_words(current))
    return tuple(cells)


def _cell_from_words(words: list[_Word]) -> TableCell:
    bbox = BBox(
        x0=min(w.bbox.x0 for w in words),
        y0=min(w.bbox.y0 for w in words),
        x1=max(w.bbox.x1 for w in words),
        y1=max(w.bbox.y1 for w in words),
    )
    text = " ".join(w.text for w in words).strip()
    return TableCell(text=text, confidence=None, bbox=bbox)


def _columns_aligned(rows: Sequence[Sequence[TableCell]]) -> bool:
    if not rows:
        return False
    n_cols = len(rows[0])
    for col_index in range(n_cols):
        centres = [(r[col_index].bbox.x0 + r[col_index].bbox.x1) / 2 for r in rows]
        widths = [r[col_index].bbox.width for r in rows]
        median_width = _median(widths) or 1.0
        spread = max(centres) - min(centres)
        if spread > median_width * 0.5:
            return False
    return True


def _median(values: Sequence[float]) -> float:
    """Sequence median; ``0.0`` for empty input."""
    seq = sorted(values)
    if not seq:
        return 0.0
    mid = len(seq) // 2
    if len(seq) % 2:
        return float(seq[mid])
    return (seq[mid - 1] + seq[mid]) / 2.0


__all__ = ["_Line", "_Word", "detect_table"]
