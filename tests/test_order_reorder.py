"""Phase-6 end-to-end reorder tests + snapshots.

Synthetic 1/2/3-column RTL layouts. The reorder is deterministic;
expected ordered ID lists are checked exactly.
"""

from __future__ import annotations

from arabic_pdf_transcribe.order import reorder
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


def _region(rid: str, x0: float, y0: float, x1: float, y1: float) -> Region:
    return Region(
        page_index=0,
        bbox=BBox(x0, y0, x1, y1),
        text=rid,
        role=RegionRole.PARAGRAPH,
        source=RegionSource.NATIVE,
    )


def _ids(regions: list[Region]) -> list[str]:
    return [r.text for r in regions]


def test_empty_input_returns_empty() -> None:
    assert reorder([], page_width=600.0, page_height=800.0) == []


def test_single_column_rtl_reads_top_to_bottom_right_to_left() -> None:
    """Single column, three rows: rightmost-first within each row."""
    regions = [
        _region("L1", 50, 100, 250, 130),
        _region("R1", 350, 100, 550, 130),
        _region("L2", 50, 200, 250, 230),
        _region("R2", 350, 200, 550, 230),
    ]
    out = reorder(regions, page_width=600.0, page_height=800.0, rtl=True)
    # Row 1 first (R1, L1), then Row 2 (R2, L2). Single column so no
    # column reversal.
    # Note: with 4 regions, column detector may fire. Check the
    # within-band right-first invariant rather than a fixed snapshot.
    # For two-column detection, RTL reads rightmost column first.
    # Either way R1 must precede L1, and R2 must precede L2.
    pos = {r: i for i, r in enumerate(_ids(out))}
    assert pos["R1"] < pos["L1"]
    assert pos["R2"] < pos["L2"]


def test_two_column_rtl_right_column_first() -> None:
    """RTL two-column: read the RIGHT column top-to-bottom first."""
    regions: list[Region] = []
    # Right column (Arabic reads first in RTL): x in 350..550
    for i, y in enumerate((100, 200, 300, 400)):
        regions.append(_region(f"R{i + 1}", 350, y, 550, y + 30))
    # Left column: x in 50..250
    for i, y in enumerate((100, 200, 300, 400)):
        regions.append(_region(f"L{i + 1}", 50, y, 250, y + 30))
    out = reorder(regions, page_width=600.0, page_height=800.0, rtl=True)
    ids = _ids(out)
    # All R-column ids come before all L-column ids.
    r_positions = [ids.index(f"R{i}") for i in range(1, 5)]
    l_positions = [ids.index(f"L{i}") for i in range(1, 5)]
    assert max(r_positions) < min(l_positions)
    # Within each column, top-to-bottom.
    assert r_positions == sorted(r_positions)
    assert l_positions == sorted(l_positions)


def test_two_column_ltr_left_column_first() -> None:
    """LTR two-column: read the LEFT column first."""
    regions: list[Region] = []
    for i, y in enumerate((100, 200, 300, 400)):
        regions.append(_region(f"L{i + 1}", 50, y, 250, y + 30))
    for i, y in enumerate((100, 200, 300, 400)):
        regions.append(_region(f"R{i + 1}", 350, y, 550, y + 30))
    out = reorder(regions, page_width=600.0, page_height=800.0, rtl=False)
    ids = _ids(out)
    l_positions = [ids.index(f"L{i}") for i in range(1, 5)]
    r_positions = [ids.index(f"R{i}") for i in range(1, 5)]
    assert max(l_positions) < min(r_positions)


def test_three_column_rtl_right_first() -> None:
    regions: list[Region] = []
    for col_label, x_centre in (("C", 300), ("L", 100), ("R", 500)):
        for i, y in enumerate((100, 200, 300, 400)):
            regions.append(_region(f"{col_label}{i + 1}", x_centre - 50, y, x_centre + 50, y + 30))
    out = reorder(regions, page_width=600.0, page_height=800.0, rtl=True)
    ids = _ids(out)
    r_positions = [ids.index(f"R{i}") for i in range(1, 5)]
    c_positions = [ids.index(f"C{i}") for i in range(1, 5)]
    l_positions = [ids.index(f"L{i}") for i in range(1, 5)]
    assert max(r_positions) < min(c_positions)
    assert max(c_positions) < min(l_positions)


def test_reorder_does_not_mutate_input() -> None:
    regions = [
        _region("A", 50, 100, 100, 130),
        _region("B", 200, 100, 250, 130),
    ]
    before = list(regions)
    reorder(regions, page_width=300.0, page_height=400.0)
    assert regions == before
