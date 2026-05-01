"""Native PDF text-extraction adapter (text-layer path).

Walks each page's text objects via ``pypdfium2``, groups characters into
words, words into lines, and lines into paragraphs by geometric proximity.
Each paragraph becomes a :class:`Region` with ``role=UNKNOWN`` and
``source=NATIVE``; role classification (heading vs paragraph vs list-item)
is phase 6's job.

The extractor also captures a per-page font-size histogram so phase 6 can
infer heading levels, and surfaces ``has_text_layer=False`` when the page
has no text objects so the validator (phase 3) can short-circuit to the ML
branch.

Coordinate system: PDF native is bottom-left, y increasing upward. The
extractor flips to top-left at the boundary so every downstream consumer
sees one convention.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from arabic_pdf_transcribe.extract import _native_tables
from arabic_pdf_transcribe.pdf._pypdfium2_loader import open_pdf
from arabic_pdf_transcribe.regions import (
    BBox,
    Region,
    RegionRole,
    RegionSource,
)

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass(frozen=True, slots=True)
class NativePage:
    """Output of :func:`extract_native` for a single page.

    ``regions`` carries every extracted paragraph (and any detected native
    table) in *extraction order* — reading-order reconstruction is phase 6.
    """

    page_index: int
    page_width: float
    page_height: float
    regions: list[Region] = field(default_factory=list)
    font_size_hist: Counter[float] = field(default_factory=Counter)
    has_text_layer: bool = True


# ---- Public API ---------------------------------------------------------


def extract_native(pdf_path: Path) -> Iterator[NativePage]:
    """Yield one :class:`NativePage` per page in the input PDF.

    Iterator-shaped so a 500-page document can be processed without
    holding every region in memory at once.
    """
    with open_pdf(pdf_path) as document:
        for page_index in range(len(document)):
            page = document[page_index]
            try:
                yield _extract_page(page, page_index)
            finally:
                page.close()


# ---- Per-page extraction ------------------------------------------------


@runtime_checkable
class _PdfTextPage(Protocol):  # pragma: no cover — protocol declaration
    def count_chars(self) -> int: ...
    def get_charbox(self, index: int, loose: bool = False) -> tuple[float, float, float, float]: ...
    def get_text_range(self, index: int, count: int) -> str: ...
    def close(self) -> None: ...


def _extract_page(page: object, page_index: int) -> NativePage:
    """Extract regions + font histogram from a single PDFium page."""
    width = float(page.get_width())  # type: ignore[attr-defined]
    height = float(page.get_height())  # type: ignore[attr-defined]
    text_page = page.get_textpage()  # type: ignore[attr-defined]

    try:
        char_count = text_page.count_chars()
        if char_count == 0:
            return NativePage(
                page_index=page_index,
                page_width=width,
                page_height=height,
                regions=[],
                font_size_hist=Counter(),
                has_text_layer=False,
            )

        chars = list(_iter_chars(text_page, char_count, height))
        font_size_hist: Counter[float] = Counter(round(c.font_size, 1) for c in chars)
        words = _group_words(chars)
        lines = _group_lines(words)

        # Native-table detection: find the largest contiguous run of lines
        # whose cell count is uniform; that run becomes a TABLE region.
        # Lines outside the run fall through to paragraph grouping.
        table_lines = _to_table_lines(lines)
        detection = _native_tables.detect_table(table_lines)
        regions: list[Region] = []
        consumed_lines: set[int] = set()
        if detection is not None:
            table_grid, consumed_indices = detection
            consumed_lines = set(consumed_indices)
            outer_bboxes = [lines[i].bbox for i in consumed_indices]
            regions.append(
                Region(
                    page_index=page_index,
                    bbox=_outer_bbox(outer_bboxes),
                    text="",
                    role=RegionRole.TABLE,
                    source=RegionSource.NATIVE,
                    table_grid=table_grid,
                )
            )

        remaining_lines = [line for idx, line in enumerate(lines) if idx not in consumed_lines]
        paragraphs = _group_paragraphs(remaining_lines)
        for para in paragraphs:
            regions.append(
                Region(
                    page_index=page_index,
                    bbox=para.bbox,
                    text=para.text,
                    role=RegionRole.UNKNOWN,
                    source=RegionSource.NATIVE,
                )
            )

        return NativePage(
            page_index=page_index,
            page_width=width,
            page_height=height,
            regions=regions,
            font_size_hist=font_size_hist,
            has_text_layer=True,
        )
    finally:
        text_page.close()


# ---- Geometry helpers ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Char:
    """One glyph from the text layer, in top-left coordinates."""

    char: str
    bbox: BBox
    font_size: float


@dataclass(frozen=True, slots=True)
class _Word:
    text: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    bbox: BBox
    words: tuple[_Word, ...] = ()


@dataclass(frozen=True, slots=True)
class _Paragraph:
    text: str
    bbox: BBox


def _iter_chars(text_page: _PdfTextPage, count: int, page_height: float) -> Iterator[_Char]:
    for index in range(count):
        try:
            x0, y0, x1, y1 = text_page.get_charbox(index, loose=False)
        except Exception:  # pragma: no cover — malformed glyph
            continue
        glyph = text_page.get_text_range(index, 1)
        # ``pypdfium2`` does not expose font size directly per character;
        # the glyph bbox height is a robust proxy (PDF cap-height plus
        # descender depending on the glyph). Phase 6's heading-level
        # inference cares only about *relative* size, so this is enough.
        font_size = float(y1 - y0)
        # Convert PDF (bottom-left) → top-left.
        top = page_height - y1
        bottom = page_height - y0
        yield _Char(
            char=glyph,
            bbox=BBox(x0=float(x0), y0=top, x1=float(x1), y1=bottom),
            font_size=font_size,
        )


def _group_words(chars: list[_Char]) -> list[_Word]:
    """Group characters into words by whitespace + adjacency."""
    if not chars:
        return []
    words: list[_Word] = []
    current_chars: list[_Char] = []
    for ch in chars:
        if ch.char.isspace():
            if current_chars:
                words.append(_word_from_chars(current_chars))
                current_chars = []
            continue
        current_chars.append(ch)
    if current_chars:
        words.append(_word_from_chars(current_chars))
    return words


def _word_from_chars(chars: list[_Char]) -> _Word:
    text = "".join(c.char for c in chars)
    bbox = _outer_bbox([c.bbox for c in chars])
    return _Word(text=text, bbox=bbox)


def _group_lines(words: list[_Word]) -> list[_Line]:
    """Group words into lines, preserving PDF-stream order.

    The grouping does **not** sort words by Y or X. Instead it walks the
    input in stream order and bins each word into the most recent line
    band whose Y is within ``median_height * 0.5``; if no band matches a
    new line is started. Stream-order preservation matters for Arabic:
    PDF spec puts text in logical (RTL = right-most-first) order in the
    content stream and uses positioning for visual layout. Sorting by X
    within a band would reverse logical order on RTL lines and
    permanently corrupt the text — phase 6 only reorders *regions*, not
    words inside a region's text. Sorting by Y is also unsafe because
    glyph bboxes within the same visual line typically differ by sub-
    pixel amounts (descenders, accents), so a strict Y-sort can flip
    word order *within* a single line.
    """
    if not words:
        return []
    median_height = _median([w.bbox.height for w in words]) or 1.0
    band = median_height * 0.5

    line_groups: list[list[_Word]] = []
    line_y: list[float] = []
    for word in words:
        placed = False
        for index, ref_y in enumerate(line_y):
            if abs(word.bbox.y0 - ref_y) <= band:
                line_groups[index].append(word)
                placed = True
                break
        if not placed:
            line_groups.append([word])
            line_y.append(word.bbox.y0)

    return [_line_from_words(group) for group in line_groups]


def _line_from_words(words: list[_Word]) -> _Line:
    # Words inside the line are kept in PDF-stream order — see ``_group_lines``
    # for why. The table detector uses the line's words tuple to split on
    # large horizontal gaps; cell-internal order is therefore stream order
    # too, which is the right thing for both LTR and RTL content.
    text = " ".join(w.text for w in words)
    bbox = _outer_bbox([w.bbox for w in words])
    return _Line(text=text, bbox=bbox, words=tuple(words))


def _to_table_lines(lines: list[_Line]) -> list[_native_tables._Line]:
    """Convert internal extraction lines to the table-detector's line type."""
    out: list[_native_tables._Line] = []
    for line in lines:
        out.append(
            _native_tables._Line(
                bbox=line.bbox,
                words=tuple(_native_tables._Word(bbox=w.bbox, text=w.text) for w in line.words),
            )
        )
    return out


def _group_paragraphs(lines: list[_Line]) -> list[_Paragraph]:
    """Group lines into paragraphs by vertical proximity."""
    if not lines:
        return []
    sorted_lines = sorted(lines, key=lambda line: line.bbox.y0)
    median_height = _median([line.bbox.height for line in sorted_lines]) or 1.0
    paragraph_gap = median_height * 1.5

    paragraphs: list[_Paragraph] = []
    current: list[_Line] = []
    last_bottom: float | None = None
    for line in sorted_lines:
        if last_bottom is None or (line.bbox.y0 - last_bottom) <= paragraph_gap:
            current.append(line)
        else:
            paragraphs.append(_paragraph_from_lines(current))
            current = [line]
        last_bottom = line.bbox.y1
    if current:
        paragraphs.append(_paragraph_from_lines(current))
    return paragraphs


def _paragraph_from_lines(lines: list[_Line]) -> _Paragraph:
    text = "\n".join(line.text for line in lines)
    bbox = _outer_bbox([line.bbox for line in lines])
    return _Paragraph(text=text, bbox=bbox)


def _outer_bbox(bboxes: list[BBox]) -> BBox:
    if not bboxes:
        return BBox(0.0, 0.0, 0.0, 0.0)
    return BBox(
        x0=min(b.x0 for b in bboxes),
        y0=min(b.y0 for b in bboxes),
        x1=max(b.x1 for b in bboxes),
        y1=max(b.y1 for b in bboxes),
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[mid])
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


__all__ = ["NativePage", "extract_native"]
