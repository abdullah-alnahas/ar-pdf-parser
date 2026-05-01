"""Phase 2: native PDF extraction tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from arabic_pdf_transcribe.errors import (
    CorruptedPDFError,
    EncryptedPDFError,
)
from arabic_pdf_transcribe.extract import extract_native
from arabic_pdf_transcribe.extract._native_tables import (
    MIN_COLS,
    MIN_ROWS,
    _Line,
    _Word,
    detect_table,
)
from arabic_pdf_transcribe.regions import (
    BBox,
    RegionRole,
    RegionSource,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DIGITAL_CLEAN = REPO_ROOT / "tests" / "fixtures" / "pdfs" / "digital-clean"


# ---- Real-fixture extraction ---------------------------------------------


def test_extract_native_2col_yields_two_pages() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    assert len(pages) == 2
    for page in pages:
        assert page.has_text_layer
        assert page.regions
        assert page.page_width > 0
        assert page.page_height > 0


def test_extract_native_2col_preserves_text() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    full_text = " ".join(r.text for page in pages for r in page.regions)
    assert "lorem" in full_text
    assert "alpha" in full_text
    assert "Page 1" in full_text
    assert "Page 2" in full_text


def test_extract_native_2col_regions_use_top_left_origin() -> None:
    """Regions should have ``y0 < y1`` (top-left origin), not flipped."""
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    for page in pages:
        for region in page.regions:
            assert region.bbox.y0 < region.bbox.y1
            assert region.bbox.x0 < region.bbox.x1
            # Bbox sits inside the page.
            assert region.bbox.x0 >= 0
            assert region.bbox.x1 <= page.page_width + 1.0
            assert region.bbox.y0 >= 0
            assert region.bbox.y1 <= page.page_height + 1.0


def test_extract_native_2col_carries_native_source_and_unknown_role() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    for page in pages:
        for region in page.regions:
            assert region.source is RegionSource.NATIVE
            # Phase 2 emits role=UNKNOWN; phase 6 will refine to heading /
            # paragraph / list_item.
            assert region.role in {RegionRole.UNKNOWN, RegionRole.TABLE}


def test_extract_native_2col_font_size_histogram_is_populated() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    for page in pages:
        assert sum(page.font_size_hist.values()) > 0


# ---- Real-Arabic fixture --------------------------------------------------


def test_extract_native_real_arabic_codepoints_round_trip() -> None:
    """Real Arabic codepoints in the text layer must reach the regions intact.

    The fixture is *not* bidi-shaped — we deliberately avoid pulling
    ``arabic_reshaper`` / ``python-bidi`` because their licenses do not
    match the project allow-list. As a result the reportlab-rendered
    fixture carries the codepoints in PDF stream order (which, for this
    fixture, happens to be visual order). What this test asserts is that
    every Arabic codepoint we wrote round-trips through extraction — phase
    9 will exercise correct logical ordering on a bidi-shaped corpus.
    """
    fixture = DIGITAL_CLEAN / "lorem-ar-real.pdf"
    if not fixture.exists():
        pytest.skip("real-arabic fixture only generated when an Arabic-capable TTF is installed")
    pages = list(extract_native(fixture))
    assert len(pages) == 1
    text = " ".join(r.text for r in pages[0].regions)
    # Source contained: "كتاب قلم ورق" + "مدرسة"
    expected_codepoints = set("كتاب قلم ورقمدرسة".replace(" ", ""))
    extracted_codepoints = set(text.replace(" ", ""))
    missing = expected_codepoints - extracted_codepoints
    assert not missing, f"Arabic codepoints missing from extraction: {missing}"


def test_line_grouping_preserves_pdf_stream_order() -> None:
    """Lines must keep their words in PDF-stream order, not re-sorted by X.

    Sorting by X would corrupt RTL text — phase 6 only reorders regions,
    not words inside a region's text. Verified directly via the helper
    instead of via a PDF fixture so the assertion is independent of
    reportlab's bidi handling.
    """
    from arabic_pdf_transcribe.extract.native import _group_lines, _Word
    from arabic_pdf_transcribe.regions import BBox

    # Three Arabic words, stream order = high-x first (proper RTL stream).
    words = [
        _Word(text="كتاب", bbox=BBox(200, 100, 250, 115)),
        _Word(text="قلم", bbox=BBox(150, 100, 190, 115)),
        _Word(text="ورق", bbox=BBox(100, 100, 140, 115)),
    ]
    lines = _group_lines(words)
    assert len(lines) == 1
    assert lines[0].text == "كتاب قلم ورق"


# ---- Mixed Arabic/English fixture ----------------------------------------


def test_extract_native_mixed_yields_one_page() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-en-mixed.pdf"))
    assert len(pages) == 1
    assert pages[0].has_text_layer
    text = " ".join(r.text for r in pages[0].regions)
    assert "alpha" in text
    assert "lorem" in text


# ---- Table fixture --------------------------------------------------------


def test_extract_native_table_fixture_detects_table() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-table.pdf"))
    assert len(pages) == 1
    table_regions = [r for r in pages[0].regions if r.role is RegionRole.TABLE]
    assert len(table_regions) == 1
    grid = table_regions[0].table_grid
    assert grid is not None
    assert grid.n_rows == 3
    assert grid.n_cols == 3
    # Cell text round-trips.
    expected = [
        ["r1c1", "r1c2", "r1c3"],
        ["r2c1", "r2c2", "r2c3"],
        ["r3c1", "r3c2", "r3c3"],
    ]
    actual = [[c.text for c in row] for row in grid.rows]
    assert actual == expected


def test_extract_native_table_fixture_preserves_title_outside_grid() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-table.pdf"))
    text_regions = [r for r in pages[0].regions if r.role is RegionRole.UNKNOWN]
    titles = [r.text for r in text_regions]
    assert any("Table fixture" in t for t in titles)


def test_extract_native_2col_does_not_emit_table() -> None:
    pages = list(extract_native(DIGITAL_CLEAN / "lorem-ar-2col.pdf"))
    for page in pages:
        assert all(r.role is not RegionRole.TABLE for r in page.regions)


# ---- detect_table direct unit tests --------------------------------------


def _word(x0: float, x1: float, y0: float, text: str) -> _Word:
    return _Word(text=text, bbox=BBox(x0, y0, x1, y0 + 10))


def _line_with_words(words: list[_Word]) -> _Line:
    if not words:
        return _Line(bbox=BBox(0, 0, 0, 0), words=())
    return _Line(
        bbox=BBox(
            x0=min(w.bbox.x0 for w in words),
            y0=min(w.bbox.y0 for w in words),
            x1=max(w.bbox.x1 for w in words),
            y1=max(w.bbox.y1 for w in words),
        ),
        words=tuple(words),
    )


def test_detect_table_short_input_returns_none() -> None:
    """Fewer than MIN_ROWS lines must never produce a table."""
    lines = [_line_with_words([_word(0, 10, y, "x")]) for y in range(MIN_ROWS - 1)]
    assert detect_table(lines) is None


def test_detect_table_prose_returns_none() -> None:
    """Prose lines (one wide block of words) must not be a table."""
    lines: list[_Line] = []
    for y in range(5):
        words = [_word(x, x + 20, y * 20, f"word{i}") for i, x in enumerate([0, 25, 50, 75])]
        lines.append(_line_with_words(words))
    assert detect_table(lines) is None


def test_detect_table_finds_grid_among_noise() -> None:
    """A 3-row clean grid in the middle of noisy lines is detected."""
    noise = [_line_with_words([_word(0, 50, y, "title")]) for y in range(2)]
    rows = []
    for y in range(2, 2 + MIN_ROWS):
        # Cell width = 20, gap = 80 (> word width)
        cells = [
            _word(0, 20, y * 20, f"r{y}c1"),
            _word(100, 120, y * 20, f"r{y}c2"),
            _word(200, 220, y * 20, f"r{y}c3"),
        ]
        rows.append(_line_with_words(cells))
    detection = detect_table(noise + rows)
    assert detection is not None
    grid, indices = detection
    assert grid.n_rows == MIN_ROWS
    assert grid.n_cols == 3
    assert indices == list(range(2, 2 + MIN_ROWS))


def test_detect_table_min_cols_enforced() -> None:
    """A grid with only one column per row is not a table."""
    rows: list[_Line] = []
    for y in range(MIN_ROWS):
        cells = [_word(0, 20, y * 20, "single")]
        rows.append(_line_with_words(cells))
    assert detect_table(rows) is None
    assert MIN_COLS >= 2


# ---- Error-translation boundary -------------------------------------------


def test_extract_native_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.pdf"
    with pytest.raises(FileNotFoundError):
        list(extract_native(missing))


def test_extract_native_corrupted_file(tmp_path: Path) -> None:
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not a pdf at all")
    with pytest.raises(CorruptedPDFError):
        list(extract_native(bad))


def test_encrypted_pdf_error_class_exists() -> None:
    """The :class:`EncryptedPDFError` must exist for phase-8 wiring."""
    # Generating an actually-encrypted fixture in-process needs a TTF/AES
    # path that pulls heavy deps. The existence check guards the typed
    # exception against accidental removal until phase 8 wires the CLI
    # exit code.
    assert EncryptedPDFError.__name__ == "EncryptedPDFError"
    assert issubclass(EncryptedPDFError, Exception)
