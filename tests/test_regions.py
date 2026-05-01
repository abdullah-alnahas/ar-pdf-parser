"""Phase 2: Region schema tests."""

from __future__ import annotations

import pytest

from arabic_pdf_transcribe.regions import (
    REGION_SCHEMA_VERSION,
    BBox,
    ListMarker,
    Region,
    RegionRole,
    RegionSource,
    TableCell,
    TableGrid,
    iter_pages,
)


def _bbox(x0: float = 0, y0: float = 0, x1: float = 10, y1: float = 10) -> BBox:
    return BBox(x0, y0, x1, y1)


def _region(
    *,
    page_index: int = 0,
    role: RegionRole = RegionRole.PARAGRAPH,
    text: str = "hello",
    **kwargs: object,
) -> Region:
    return Region(
        page_index=page_index,
        bbox=_bbox(),
        text=text,
        role=role,
        source=RegionSource.NATIVE,
        **kwargs,  # type: ignore[arg-type]
    )


# ---- Schema version -------------------------------------------------------


def test_schema_version_is_one() -> None:
    assert REGION_SCHEMA_VERSION == "1"


# ---- BBox -----------------------------------------------------------------


def test_bbox_geometry() -> None:
    b = BBox(1, 2, 5, 6)
    assert b.width == 4
    assert b.height == 4
    assert b.area == 16


# ---- Region immutability + helpers ---------------------------------------


def test_region_is_frozen() -> None:
    r = _region()
    with pytest.raises((AttributeError, Exception)):
        r.text = "mutated"  # type: ignore[misc]


def test_region_with_text_returns_copy() -> None:
    r = _region(text="orig")
    other = r.with_text("changed")
    assert r.text == "orig"
    assert other.text == "changed"


def test_region_with_role_and_heading_level() -> None:
    r = _region(role=RegionRole.UNKNOWN)
    promoted = r.with_role(RegionRole.HEADING).with_heading_level(2)
    assert promoted.role is RegionRole.HEADING
    assert promoted.heading_level == 2


def test_region_with_meta_merges_keys() -> None:
    r = _region().with_meta(foo="bar")
    r2 = r.with_meta(baz="qux")
    assert dict(r.meta) == {"foo": "bar"}
    assert dict(r2.meta) == {"foo": "bar", "baz": "qux"}


def test_region_meta_is_read_only() -> None:
    r = _region().with_meta(foo="bar")
    with pytest.raises(TypeError):
        r.meta["new"] = "x"  # type: ignore[index]


def test_region_is_hashable() -> None:
    r = _region()
    assert isinstance(hash(r), int)
    assert hash(r) == hash(_region())


def test_two_equal_regions_collapse_in_set() -> None:
    s = {_region(), _region()}
    assert len(s) == 1


def test_failure_placeholder_factory() -> None:
    bbox = _bbox(0, 0, 100, 50)
    placeholder = Region.as_failure_placeholder(
        "ocr_failed: out-of-memory on page 7",
        page_index=7,
        bbox=bbox,
    )
    assert placeholder.role is RegionRole.FAILURE_PLACEHOLDER
    assert placeholder.failure_reason == "ocr_failed: out-of-memory on page 7"
    assert placeholder.text == ""


# ---- Serialisation --------------------------------------------------------


def test_region_json_round_trip_minimal() -> None:
    r = _region()
    blob = r.to_json()
    restored = Region.from_json(blob)
    assert restored == r


def test_region_json_round_trip_with_list_marker_and_meta() -> None:
    r = _region(
        role=RegionRole.LIST_ITEM,
        list_marker=ListMarker(kind="ordered", ordinal=3, raw_marker="3."),
    ).with_meta(font_size=12.5, source_class="text")
    blob = r.to_json()
    restored = Region.from_json(blob)
    assert restored == r
    assert restored.list_marker is not None
    assert restored.list_marker.ordinal == 3


def test_region_json_round_trip_with_table_grid() -> None:
    cell = TableCell(text="hello", confidence=0.9, bbox=_bbox(0, 0, 50, 20))
    grid = TableGrid(rows=((cell, cell), (cell, cell)))
    r = _region(role=RegionRole.TABLE, text="").with_table_grid(grid)
    restored = Region.from_json(r.to_json())
    assert restored == r
    assert restored.table_grid is not None
    assert restored.table_grid.n_rows == 2
    assert restored.table_grid.n_cols == 2


def test_region_from_json_rejects_wrong_schema_version() -> None:
    blob = '{"schema_version": "999", "region": {}}'
    with pytest.raises(ValueError, match="schema version"):
        Region.from_json(blob)


def test_region_to_json_is_stable_across_runs() -> None:
    r = _region().with_meta(z="last", a="first")
    assert r.to_json() == r.to_json()


def test_region_to_json_preserves_unicode() -> None:
    r = _region(text="مرحبا")
    blob = r.to_json()
    assert "مرحبا" in blob
    assert Region.from_json(blob).text == "مرحبا"


# ---- iter_pages -----------------------------------------------------------


def test_iter_pages_groups_by_page() -> None:
    a = _region(page_index=0, text="page0a")
    b = _region(page_index=1, text="page1a")
    c = _region(page_index=0, text="page0b")
    pages = list(iter_pages([a, b, c]))
    assert [page for page, _ in pages] == [0, 1]
    assert [r.text for r in pages[0][1]] == ["page0a", "page0b"]
    assert [r.text for r in pages[1][1]] == ["page1a"]


def test_iter_pages_empty() -> None:
    assert list(iter_pages([])) == []
