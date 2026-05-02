"""Phase-7 Markdown emitter tests."""

from __future__ import annotations

from arabic_pdf_transcribe.emit import emit_markdown
from arabic_pdf_transcribe.regions import (
    BBox,
    ListMarker,
    Region,
    RegionRole,
    RegionSource,
    TableCell,
    TableGrid,
)


def _region(
    *,
    role: RegionRole = RegionRole.PARAGRAPH,
    text: str = "",
    page_index: int = 0,
    heading_level: int | None = None,
    list_marker: ListMarker | None = None,
    table_grid: TableGrid | None = None,
    group_id: str | None = None,
    failure_reason: str | None = None,
    bbox: BBox = BBox(0.0, 0.0, 100.0, 30.0),
    meta: dict[str, object] | None = None,
) -> Region:
    region = Region(
        page_index=page_index,
        bbox=bbox,
        text=text,
        role=role,
        source=RegionSource.NATIVE,
        heading_level=heading_level,
        list_marker=list_marker,
        table_grid=table_grid,
        group_id=group_id,
        failure_reason=failure_reason,
    )
    if meta:
        region = region.with_meta(**meta)
    return region


# ---------------------------------------------------------------------------
# Empty + role mappings
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_string() -> None:
    assert emit_markdown([]) == ""


def test_paragraph_emits_plain_text() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="hello world")
    assert emit_markdown([r]) == "hello world\n"


def test_heading_emits_hash_prefix() -> None:
    r = _region(role=RegionRole.HEADING, text="Title", heading_level=1)
    assert emit_markdown([r]) == "# Title\n"


def test_heading_default_level_is_h2_when_missing() -> None:
    r = _region(role=RegionRole.HEADING, text="Title", heading_level=None)
    assert emit_markdown([r]) == "## Title\n"


def test_heading_level_capped_at_h6() -> None:
    r = _region(role=RegionRole.HEADING, text="X", heading_level=99)
    assert emit_markdown([r]) == "###### X\n"


def test_heading_level_floored_at_h1() -> None:
    r = _region(role=RegionRole.HEADING, text="X", heading_level=0)
    assert emit_markdown([r]) == "# X\n"


def test_caption_ungrouped_emits_italic_paragraph() -> None:
    r = _region(role=RegionRole.CAPTION, text="A caption")
    assert emit_markdown([r]) == "*A caption*\n"


def test_header_footer_is_suppressed() -> None:
    r = _region(role=RegionRole.HEADER_FOOTER, text="page 1")
    assert emit_markdown([r]) == ""


def test_failure_placeholder_emits_html_comment() -> None:
    r = _region(
        role=RegionRole.FAILURE_PLACEHOLDER,
        page_index=4,
        failure_reason="ocr_failed",
    )
    out = emit_markdown([r])
    assert "<!--" in out
    assert "page 5" in out  # 1-based
    assert "ocr_failed" in out


def test_unknown_role_emits_as_paragraph() -> None:
    r = _region(role=RegionRole.UNKNOWN, text="orphan")
    assert emit_markdown([r]) == "orphan\n"


# ---------------------------------------------------------------------------
# List items
# ---------------------------------------------------------------------------


def test_bullet_list_item() -> None:
    r = _region(
        role=RegionRole.LIST_ITEM,
        text="- first",
        list_marker=ListMarker(kind="bullet", raw_marker="-"),
    )
    assert emit_markdown([r]) == "- first\n"


def test_ordered_list_item_uses_ordinal() -> None:
    r = _region(
        role=RegionRole.LIST_ITEM,
        text="3) third",
        list_marker=ListMarker(kind="ordered", ordinal=3, raw_marker="3)"),
    )
    assert emit_markdown([r]) == "3. third\n"


def test_consecutive_list_items_share_block() -> None:
    items = [
        _region(
            role=RegionRole.LIST_ITEM,
            text="- a",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- b",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- c",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
    ]
    out = emit_markdown(items)
    assert out == "- a\n- b\n- c\n"


def test_list_then_paragraph_breaks_block() -> None:
    regions = [
        _region(
            role=RegionRole.LIST_ITEM,
            text="- a",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(role=RegionRole.PARAGRAPH, text="prose"),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- b",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
    ]
    out = emit_markdown(regions)
    assert out == "- a\n\nprose\n\n- b\n"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def _grid(rows: list[list[str]]) -> TableGrid:
    return TableGrid(
        rows=tuple(
            tuple(TableCell(text=cell, confidence=None, bbox=BBox(0, 0, 10, 10)) for cell in row)
            for row in rows
        )
    )


def test_table_emits_pipe_table() -> None:
    grid = _grid([["h1", "h2"], ["a", "b"]])
    r = _region(role=RegionRole.TABLE, table_grid=grid)
    out = emit_markdown([r])
    assert "| h1 | h2 |" in out
    assert "| --- | --- |" in out
    assert "| a | b |" in out


def test_table_pipes_in_cells_escaped() -> None:
    grid = _grid([["h1"], ["x | y"]])
    r = _region(role=RegionRole.TABLE, table_grid=grid)
    out = emit_markdown([r])
    assert r"x \| y" in out


def test_table_v1_simplification_emits_comment() -> None:
    grid = _grid([["a", "b"]])
    r = _region(
        role=RegionRole.TABLE,
        table_grid=grid,
        meta={"v1_table_simplification": True},
    )
    out = emit_markdown([r])
    assert "<!-- v1: merged cells flattened -->" in out


def test_table_short_row_padded_to_n_cols() -> None:
    grid = _grid([["a", "b", "c"], ["x"]])
    r = _region(role=RegionRole.TABLE, table_grid=grid)
    out = emit_markdown([r])
    assert "| x |  |  |" in out


def test_table_empty_grid_emits_nothing() -> None:
    r = _region(role=RegionRole.TABLE, table_grid=TableGrid(rows=()))
    out = emit_markdown([r])
    assert out == ""


# ---------------------------------------------------------------------------
# Figures + captions
# ---------------------------------------------------------------------------


def test_figure_without_group_uses_default_alt() -> None:
    r = _region(role=RegionRole.FIGURE, page_index=2)
    out = emit_markdown([r])
    assert out == "![figure on page 3](#)\n"


def test_grouped_figure_caption_pair_uses_caption_as_alt() -> None:
    fig = _region(role=RegionRole.FIGURE, page_index=0, group_id="g1")
    cap = _region(role=RegionRole.CAPTION, text="A horse", group_id="g1")
    out = emit_markdown([fig, cap])
    assert "![A horse](#)" in out
    # Caption suppressed (no italic line).
    assert "*A horse*" not in out


def test_orphan_grouped_caption_still_renders() -> None:
    """Regression: caption with group_id but no matching figure must
    still emit as a caption — otherwise text is lost."""
    cap = _region(role=RegionRole.CAPTION, text="orphan", group_id="solo")
    out = emit_markdown([cap])
    assert "*orphan*" in out


def test_caption_brackets_in_alt_text_escaped() -> None:
    fig = _region(role=RegionRole.FIGURE, group_id="g1")
    cap = _region(role=RegionRole.CAPTION, text="[bracketed]", group_id="g1")
    out = emit_markdown([fig, cap])
    assert r"\[bracketed\]" in out


# ---------------------------------------------------------------------------
# Adversarial text
# ---------------------------------------------------------------------------


def test_paragraph_starting_with_hash_is_escaped() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="# not a heading")
    out = emit_markdown([r])
    assert out == "\\# not a heading\n"


def test_paragraph_starting_with_dash_is_escaped() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="- not a bullet")
    out = emit_markdown([r])
    assert out == "\\- not a bullet\n"


def test_paragraph_with_pipes_is_safe_outside_table() -> None:
    """Pipes outside tables don't need escaping in standard Markdown,
    but they should at least not crash the emitter."""
    r = _region(role=RegionRole.PARAGRAPH, text="a | b | c")
    out = emit_markdown([r])
    assert out == "a | b | c\n"


def test_paragraph_with_inline_emphasis_chars_escaped() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="**bold** _italic_")
    out = emit_markdown([r])
    assert out == r"\*\*bold\*\* \_italic\_" + "\n"


# ---------------------------------------------------------------------------
# Bidi + normalisation
# ---------------------------------------------------------------------------


def test_arabic_with_latin_gets_rlm_prefix() -> None:
    r = _region(
        role=RegionRole.PARAGRAPH,
        text="هذه كلمة Arabic في اللغة",
    )
    out = emit_markdown([r])
    assert out.startswith("‏")


def test_pure_arabic_no_rlm() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="السلام عليكم")
    out = emit_markdown([r])
    assert "‏" not in out


def test_apply_bidi_false_skips_rlm() -> None:
    r = _region(role=RegionRole.PARAGRAPH, text="هذه Arabic")
    out = emit_markdown([r], apply_bidi=False)
    assert "‏" not in out


def test_nfkc_collapses_allah_ligature() -> None:
    """Verifies opt-in NFKC routes through the emitter."""
    r = _region(role=RegionRole.PARAGRAPH, text="ﷲ")
    nfc_out = emit_markdown([r], normalisation="NFC", apply_bidi=False)
    nfkc_out = emit_markdown([r], normalisation="NFKC", apply_bidi=False)
    assert nfc_out != nfkc_out
    # NFKC expansion is longer.
    assert len(nfkc_out) > len(nfc_out)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_emit_markdown_byte_identical_across_runs() -> None:
    """Acceptance criterion: snapshot stability."""
    regions = [
        _region(role=RegionRole.HEADING, text="Chapter 1", heading_level=1),
        _region(role=RegionRole.PARAGRAPH, text="Some prose."),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- a",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- b",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
    ]
    a = emit_markdown(regions)
    b = emit_markdown(regions)
    assert a == b


def test_full_document_snapshot() -> None:
    """End-to-end multi-role rendering snapshot."""
    grid = _grid([["A", "B"], ["1", "2"]])
    regions = [
        _region(role=RegionRole.HEADING, text="Title", heading_level=1),
        _region(role=RegionRole.PARAGRAPH, text="intro paragraph"),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- one",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(
            role=RegionRole.LIST_ITEM,
            text="- two",
            list_marker=ListMarker(kind="bullet", raw_marker="-"),
        ),
        _region(role=RegionRole.HEADING, text="Section", heading_level=2),
        _region(role=RegionRole.TABLE, table_grid=grid),
    ]
    expected = (
        "# Title\n\n"
        "intro paragraph\n\n"
        "- one\n- two\n\n"
        "## Section\n\n"
        "| A | B |\n| --- | --- |\n| 1 | 2 |\n"
    )
    assert emit_markdown(regions) == expected
