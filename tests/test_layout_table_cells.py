"""Phase-4 Pillow-only table-cell morphology tests.

The morphology operates on Pillow images. Pillow is part of the [ml]
extra; the tests skip cleanly when Pillow is not installed (CI without
[ml] still passes).
"""

from __future__ import annotations

import pytest

from arabic_pdf_transcribe.regions import BBox, TableGrid

PIL = pytest.importorskip("PIL")
from PIL import Image, ImageDraw  # noqa: E402

from arabic_pdf_transcribe.layout._table_cells import detect_table_cells  # noqa: E402


def _draw_grid(width: int, height: int, *, rows: int, cols: int) -> Image.Image:
    """Draw a black-on-white ruled grid with ``rows`` x ``cols`` cells.

    Line positions are placed inside the image (not on the boundary)
    so PIL's coordinate semantics don't clip the outer edges. Phase-9
    will replace synthetic fixtures with real scanned tables.
    """
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 10
    inner_w = width - 2 * margin
    inner_h = height - 2 * margin
    cell_w = inner_w // cols
    cell_h = inner_h // rows
    for r in range(rows + 1):
        y = margin + r * cell_h
        draw.line([(margin, y), (margin + inner_w, y)], fill=(0, 0, 0), width=2)
    for c in range(cols + 1):
        x = margin + c * cell_w
        draw.line([(x, margin), (x, margin + inner_h)], fill=(0, 0, 0), width=2)
    return img


def test_detect_table_cells_finds_3x3_grid() -> None:
    img = _draw_grid(300, 300, rows=3, cols=3)
    grid = detect_table_cells(img, BBox(0.0, 0.0, 300.0, 300.0))
    assert isinstance(grid, TableGrid)
    assert grid.n_rows == 3
    assert grid.n_cols == 3


def test_detect_table_cells_returns_none_for_empty_crop() -> None:
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    grid = detect_table_cells(img, BBox(0.0, 0.0, 300.0, 300.0))
    assert grid is None


def test_detect_table_cells_returns_none_for_zero_size_bbox() -> None:
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    grid = detect_table_cells(img, BBox(0.0, 0.0, 0.0, 0.0))
    assert grid is None


def test_detect_table_cells_handles_crop_outside_image_gracefully() -> None:
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    # Bbox extends past image bounds; crop is clipped to image size.
    grid = detect_table_cells(img, BBox(-10.0, -10.0, 1000.0, 1000.0))
    assert grid is None  # blank page → no grid found


def test_detect_table_cells_2x2_grid_cell_bboxes_inside_table_bbox() -> None:
    img = _draw_grid(200, 200, rows=2, cols=2)
    grid = detect_table_cells(img, BBox(0.0, 0.0, 200.0, 200.0))
    assert grid is not None
    for row in grid.rows:
        for cell in row:
            assert 0.0 <= cell.bbox.x0 <= 200.0
            assert 0.0 <= cell.bbox.y0 <= 200.0
            assert 0.0 <= cell.bbox.x1 <= 200.0
            assert 0.0 <= cell.bbox.y1 <= 200.0
            assert cell.text == ""
            assert cell.confidence is None


def test_detect_table_cells_offset_bbox_translates_cells() -> None:
    """A grid drawn at (0,0) but addressed via an offset bbox returns
    cell bboxes in the page coordinate system."""
    img = Image.new("RGB", (400, 400), color=(255, 255, 255))
    grid_img = _draw_grid(200, 200, rows=2, cols=2)
    img.paste(grid_img, (100, 100))
    grid = detect_table_cells(img, BBox(100.0, 100.0, 300.0, 300.0))
    assert grid is not None
    flat_bboxes = [cell.bbox for row in grid.rows for cell in row]
    # Every cell should sit between x=100 and x=300 in page coordinates.
    for bbox in flat_bboxes:
        assert 100.0 <= bbox.x0 <= 300.0
        assert 100.0 <= bbox.y0 <= 300.0
