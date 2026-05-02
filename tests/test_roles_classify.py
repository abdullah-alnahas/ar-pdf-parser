"""Phase-6 role-classifier tests."""

from __future__ import annotations

from arabic_pdf_transcribe.regions import (
    BBox,
    Region,
    RegionRole,
    RegionSource,
)
from arabic_pdf_transcribe.roles import classify_page
from arabic_pdf_transcribe.roles.classify import ClassifyConfig


def _region(
    *,
    role: RegionRole = RegionRole.PARAGRAPH,
    x0: float = 0.0,
    y0: float = 0.0,
    x1: float = 100.0,
    y1: float = 30.0,
    text: str = "",
    page_index: int = 0,
    meta: dict[str, object] | None = None,
    source: RegionSource = RegionSource.NATIVE,
) -> Region:
    region = Region(
        page_index=page_index,
        bbox=BBox(x0, y0, x1, y1),
        text=text,
        role=role,
        source=source,
    )
    if meta:
        region = region.with_meta(**meta)
    return region


# ---------------------------------------------------------------------------
# Heading-level inference
# ---------------------------------------------------------------------------


def test_no_size_signal_at_all_emits_h2() -> None:
    """Spec rule: when no size signal exists, every heading is H2."""
    regions = [
        _region(role=RegionRole.HEADING, x0=0, y0=0, x1=0, y1=0),  # zero-height
        _region(role=RegionRole.HEADING, x0=0, y0=100, x1=0, y1=100),
    ]
    out = classify_page(regions, page_width=600.0, page_height=800.0)
    for r in out:
        assert r.role is RegionRole.HEADING
        assert r.heading_level == 2


def test_heading_levels_from_font_size_meta() -> None:
    """Three headings with distinct meta font sizes bin into H1/H2/H3."""
    regions = [
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=0,
            x1=100,
            y1=20,
            text="big",
            meta={"font_size": 24.0},
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=100,
            x1=100,
            y1=120,
            text="medium",
            meta={"font_size": 16.0},
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=200,
            x1=100,
            y1=220,
            text="small",
            meta={"font_size": 12.0},
        ),
    ]
    out = classify_page(regions, page_width=600.0, page_height=800.0)
    levels = {r.text: r.heading_level for r in out}
    assert levels["big"] == 1
    assert levels["medium"] == 2
    assert levels["small"] == 3


def test_heading_levels_from_region_height_on_ml_path() -> None:
    """ML / OCR pages have no font-size meta; bin by bbox height."""
    regions = [
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=0,
            x1=100,
            y1=60,
            text="big",
            source=RegionSource.OCR,
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=100,
            x1=100,
            y1=140,
            text="medium",
            source=RegionSource.OCR,
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=200,
            x1=100,
            y1=220,
            text="small",
            source=RegionSource.OCR,
        ),
    ]
    out = classify_page(regions, page_width=600.0, page_height=800.0)
    levels = {r.text: r.heading_level for r in out}
    assert levels["big"] == 1
    assert levels["medium"] == 2
    assert levels["small"] == 3


def test_native_headings_without_font_size_meta_default_to_h2() -> None:
    """Spec contract: native-path headings without font_size meta
    fall back to H2 even when bbox heights vary. Height is NOT a
    substitute on the native path — the spec is explicit."""
    regions = [
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=0,
            x1=100,
            y1=60,
            text="big-bbox",
            source=RegionSource.NATIVE,
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=100,
            x1=100,
            y1=140,
            text="medium-bbox",
            source=RegionSource.NATIVE,
        ),
        _region(
            role=RegionRole.HEADING,
            x0=0,
            y0=200,
            x1=100,
            y1=220,
            text="small-bbox",
            source=RegionSource.NATIVE,
        ),
    ]
    out = classify_page(regions, page_width=600.0, page_height=800.0)
    for r in out:
        assert (
            r.heading_level == 2
        ), f"native heading {r.text!r} without font_size meta must fall to H2"


def test_two_headings_only_falls_back_to_h2() -> None:
    """Quantile binning needs >= 3 samples; below that, default to H2."""
    regions = [
        _region(role=RegionRole.HEADING, x0=0, y0=0, x1=100, y1=40, text="a"),
        _region(role=RegionRole.HEADING, x0=0, y0=100, x1=100, y1=120, text="b"),
    ]
    out = classify_page(regions, page_width=600.0, page_height=800.0)
    for r in out:
        assert r.heading_level == 2


# ---------------------------------------------------------------------------
# List-item detection
# ---------------------------------------------------------------------------


def test_bullet_dash_promotes_to_list_item() -> None:
    region = _region(role=RegionRole.PARAGRAPH, text="- first bullet")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.LIST_ITEM
    assert out.list_marker is not None
    assert out.list_marker.kind == "bullet"
    assert out.list_marker.raw_marker == "-"


def test_bullet_arabic_glyph_promotes_to_list_item() -> None:
    region = _region(role=RegionRole.PARAGRAPH, text="• item")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.LIST_ITEM
    assert out.list_marker is not None
    assert out.list_marker.raw_marker == "•"


def test_ordered_western_digits() -> None:
    region = _region(role=RegionRole.PARAGRAPH, text="3) third item")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.LIST_ITEM
    assert out.list_marker is not None
    assert out.list_marker.kind == "ordered"
    assert out.list_marker.ordinal == 3


def test_ordered_arabic_indic_digits() -> None:
    """Arabic-Indic '٣)' must parse as ordinal 3."""
    region = _region(role=RegionRole.PARAGRAPH, text="٣) ثالثاً")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.LIST_ITEM
    assert out.list_marker is not None
    assert out.list_marker.ordinal == 3


def test_non_list_text_stays_paragraph() -> None:
    # Place at y=400 so the header/footer prune does not catch it.
    region = _region(
        role=RegionRole.PARAGRAPH,
        x0=0,
        y0=400,
        x1=100,
        y1=430,
        text="ordinary prose",
    )
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.PARAGRAPH
    assert out.list_marker is None


# ---------------------------------------------------------------------------
# Header / footer prune
# ---------------------------------------------------------------------------


def test_top_band_paragraph_gets_pruned() -> None:
    region = _region(role=RegionRole.PARAGRAPH, x0=0, y0=10, x1=100, y1=30, text="header")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.HEADER_FOOTER


def test_bottom_band_paragraph_gets_pruned() -> None:
    region = _region(role=RegionRole.PARAGRAPH, x0=0, y0=775, x1=100, y1=795, text="footer")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.HEADER_FOOTER


def test_middle_band_paragraph_stays() -> None:
    region = _region(role=RegionRole.PARAGRAPH, x0=0, y0=400, x1=100, y1=430, text="body")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.PARAGRAPH


def test_heading_in_top_band_is_not_demoted() -> None:
    """Conservative: HEADING role stays even if positioned at the page top."""
    region = _region(role=RegionRole.HEADING, x0=0, y0=10, x1=100, y1=30, text="title")
    [out] = classify_page([region], page_width=600.0, page_height=800.0)
    assert out.role is RegionRole.HEADING


def test_prune_disabled_via_config() -> None:
    region = _region(role=RegionRole.PARAGRAPH, x0=0, y0=10, x1=100, y1=30)
    [out] = classify_page(
        [region],
        page_width=600.0,
        page_height=800.0,
        config=ClassifyConfig(prune_header_footer=False),
    )
    assert out.role is RegionRole.PARAGRAPH


# ---------------------------------------------------------------------------
# Caption-figure linkage
# ---------------------------------------------------------------------------


def test_caption_close_below_figure_groups_them() -> None:
    figure = _region(role=RegionRole.FIGURE, x0=100, y0=200, x1=400, y1=400, text="fig")
    caption = _region(
        role=RegionRole.CAPTION,
        x0=100,
        y0=410,
        x1=400,
        y1=430,
        text="A picture",
    )
    out = classify_page([figure, caption], page_width=600.0, page_height=800.0)
    assert out[0].group_id is not None
    assert out[1].group_id is not None
    assert out[0].group_id == out[1].group_id


def test_caption_far_below_figure_stays_ungrouped() -> None:
    figure = _region(role=RegionRole.FIGURE, x0=100, y0=200, x1=400, y1=400, text="fig")
    # 500px below figure on an 800px-tall page → > 5% threshold.
    caption = _region(role=RegionRole.CAPTION, x0=100, y0=700, x1=400, y1=730)
    out = classify_page([figure, caption], page_width=600.0, page_height=800.0)
    assert out[0].group_id is None
    assert out[1].group_id is None


def test_caption_above_figure_stays_ungrouped() -> None:
    """Caption ABOVE figure does not link — only below qualifies."""
    figure = _region(role=RegionRole.FIGURE, x0=100, y0=400, x1=400, y1=600, text="fig")
    caption = _region(role=RegionRole.CAPTION, x0=100, y0=200, x1=400, y1=230)
    out = classify_page([figure, caption], page_width=600.0, page_height=800.0)
    assert out[0].group_id is None
    assert out[1].group_id is None


def test_caption_no_horizontal_overlap_stays_ungrouped() -> None:
    figure = _region(role=RegionRole.FIGURE, x0=0, y0=200, x1=200, y1=400, text="fig")
    caption = _region(role=RegionRole.CAPTION, x0=400, y0=410, x1=600, y1=430)
    out = classify_page([figure, caption], page_width=600.0, page_height=800.0)
    assert out[0].group_id is None
    assert out[1].group_id is None


def test_two_figures_pair_with_distinct_captions() -> None:
    fig_a = _region(role=RegionRole.FIGURE, x0=0, y0=200, x1=200, y1=300, text="A")
    cap_a = _region(role=RegionRole.CAPTION, x0=0, y0=310, x1=200, y1=330, text="capA")
    fig_b = _region(role=RegionRole.FIGURE, x0=300, y0=500, x1=500, y1=600, text="B")
    cap_b = _region(role=RegionRole.CAPTION, x0=300, y0=610, x1=500, y1=630, text="capB")
    out = classify_page([fig_a, cap_a, fig_b, cap_b], page_width=600.0, page_height=800.0)
    assert out[0].group_id is not None
    assert out[0].group_id == out[1].group_id
    assert out[2].group_id is not None
    assert out[2].group_id == out[3].group_id
    assert out[0].group_id != out[2].group_id


# ---------------------------------------------------------------------------
# Determinism + purity
# ---------------------------------------------------------------------------


def test_classify_is_pure() -> None:
    region = _region(role=RegionRole.PARAGRAPH, text="* bullet")
    [out1] = classify_page([region], page_width=600.0, page_height=800.0)
    [out2] = classify_page([region], page_width=600.0, page_height=800.0)
    # Pure: same input → same output (group_id uses uuid5 for
    # determinism, list-marker is field-equality identical).
    assert out1.role is out2.role
    assert out1.list_marker == out2.list_marker
    assert region.role is RegionRole.PARAGRAPH  # input not mutated
