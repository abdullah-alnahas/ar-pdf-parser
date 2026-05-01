"""Unified document model.

Both branches of the pipeline (native text extraction in phase 2 and the ML
branch in phases 4-5) converge on this representation. Downstream stages
(reorder + role classification in phase 6, emitters in phase 7) consume
``Region`` exclusively, so the schema is intentionally rich enough to carry
heading levels, list ordinals, table grids, figure-caption linkage, and
failure placeholders without losing information at the boundary.

The schema is versioned via :data:`REGION_SCHEMA_VERSION`; the version is
included in :meth:`Region.to_json`'s envelope so a future on-disk reader can
distinguish schema versions and migrate forward.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

REGION_SCHEMA_VERSION = "1"


class RegionRole(Enum):
    """Role of a region within the document.

    Order matches the spec's "Semantic output contract" listing in the
    Resolved Decisions section.
    """

    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER_FOOTER = "header_footer"
    FAILURE_PLACEHOLDER = "failure_placeholder"
    UNKNOWN = "unknown"


class RegionSource(Enum):
    """Which branch of the pipeline produced this region."""

    NATIVE = "native"
    OCR = "ocr"


ListKind = Literal["bullet", "ordered"]


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box, top-left origin, PDF user units.

    The pipeline normalises the coordinate system at the PDF-extraction
    boundary so every downstream consumer sees the same convention. PDF
    native is bottom-left; the native extractor in phase 2 converts.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class ListMarker:
    """Bullet or ordered-list marker captured from a list-item region.

    ``kind`` distinguishes bullets from ordered items; ``ordinal`` is the
    integer for ordered items (``None`` for bullets); ``raw_marker`` keeps
    the original glyph(s) so the emitter can preserve the document's style
    (Arabic-Indic numerals, Latin numerals, Arabic dash, etc.).
    """

    kind: ListKind
    ordinal: int | None = None
    raw_marker: str | None = None


@dataclass(frozen=True, slots=True)
class TableCell:
    text: str
    confidence: float | None
    bbox: BBox


@dataclass(frozen=True, slots=True)
class TableGrid:
    """Row-major grid of cells.

    v1 supports basic grids only; merged / spanned cells flow through with
    ``meta["v1_table_simplification"] = True`` on the parent region (phase 4
    flattens row/colspan > 1 to per-row text per the plan).
    """

    rows: tuple[tuple[TableCell, ...], ...]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)


# Internal representation of ``meta``: a tuple of (key, value) pairs so the
# whole ``Region`` is hashable. Public access goes through ``Region.meta``.
_MetaItems = tuple[tuple[str, Any], ...]


def _freeze_meta(meta: Mapping[str, Any] | None) -> _MetaItems:
    if not meta:
        return ()
    return tuple(sorted(meta.items()))


@dataclass(frozen=True, slots=True)
class Region:
    """A single rectangular region on a single page.

    The fields are flat on purpose: any consumer can introspect a region in
    one read, no follow-up lookups required. Optional fields are ``None``
    unless the role calls for them.
    """

    page_index: int
    bbox: BBox
    text: str
    role: RegionRole
    source: RegionSource
    confidence: float | None = None
    # ``heading_level`` is ``1``/``2``/``3`` when ``role == HEADING`` (capped
    # at H3 per the spec); ``None`` otherwise.
    heading_level: int | None = None
    list_marker: ListMarker | None = None
    table_grid: TableGrid | None = None
    # ``group_id`` ties a figure region to its caption region (and vice
    # versa); ``None`` for ungrouped regions.
    group_id: str | None = None
    failure_reason: str | None = None
    # ``meta`` stored as an immutable, sorted tuple; the public ``meta``
    # property exposes a read-only mapping.
    _meta_items: _MetaItems = ()

    # ---- Helpers --------------------------------------------------------

    @property
    def meta(self) -> Mapping[str, Any]:
        """Return ``meta`` as a read-only mapping."""
        return MappingProxyType(dict(self._meta_items))

    def with_text(self, text: str) -> Region:
        return replace(self, text=text)

    def with_role(self, role: RegionRole) -> Region:
        return replace(self, role=role)

    def with_heading_level(self, level: int | None) -> Region:
        return replace(self, heading_level=level)

    def with_list_marker(self, marker: ListMarker | None) -> Region:
        return replace(self, list_marker=marker)

    def with_table_grid(self, grid: TableGrid | None) -> Region:
        return replace(self, table_grid=grid)

    def with_group_id(self, group_id: str | None) -> Region:
        return replace(self, group_id=group_id)

    def with_meta(self, **kwargs: Any) -> Region:
        merged = dict(self._meta_items)
        merged.update(kwargs)
        return replace(self, _meta_items=_freeze_meta(merged))

    @classmethod
    def as_failure_placeholder(
        cls,
        reason: str,
        *,
        page_index: int,
        bbox: BBox,
        source: RegionSource = RegionSource.NATIVE,
    ) -> Region:
        return cls(
            page_index=page_index,
            bbox=bbox,
            text="",
            role=RegionRole.FAILURE_PLACEHOLDER,
            source=source,
            failure_reason=reason,
        )

    # ---- Serialisation --------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_index": self.page_index,
            "bbox": [self.bbox.x0, self.bbox.y0, self.bbox.x1, self.bbox.y1],
            "text": self.text,
            "role": self.role.value,
            "source": self.source.value,
            "confidence": self.confidence,
            "heading_level": self.heading_level,
            "list_marker": (
                None
                if self.list_marker is None
                else {
                    "kind": self.list_marker.kind,
                    "ordinal": self.list_marker.ordinal,
                    "raw_marker": self.list_marker.raw_marker,
                }
            ),
            "table_grid": (
                None
                if self.table_grid is None
                else {
                    "rows": [
                        [
                            {
                                "text": cell.text,
                                "confidence": cell.confidence,
                                "bbox": [
                                    cell.bbox.x0,
                                    cell.bbox.y0,
                                    cell.bbox.x1,
                                    cell.bbox.y1,
                                ],
                            }
                            for cell in row
                        ]
                        for row in self.table_grid.rows
                    ]
                }
            ),
            "group_id": self.group_id,
            "failure_reason": self.failure_reason,
            "meta": dict(self._meta_items),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Region:
        bbox = BBox(*data["bbox"])
        list_marker_raw = data.get("list_marker")
        list_marker = (
            None
            if list_marker_raw is None
            else ListMarker(
                kind=list_marker_raw["kind"],
                ordinal=list_marker_raw.get("ordinal"),
                raw_marker=list_marker_raw.get("raw_marker"),
            )
        )
        table_grid_raw = data.get("table_grid")
        if table_grid_raw is None:
            table_grid: TableGrid | None = None
        else:
            rows: list[tuple[TableCell, ...]] = []
            for row in table_grid_raw["rows"]:
                cells = tuple(
                    TableCell(
                        text=c["text"],
                        confidence=c.get("confidence"),
                        bbox=BBox(*c["bbox"]),
                    )
                    for c in row
                )
                rows.append(cells)
            table_grid = TableGrid(rows=tuple(rows))
        return cls(
            page_index=data["page_index"],
            bbox=bbox,
            text=data["text"],
            role=RegionRole(data["role"]),
            source=RegionSource(data["source"]),
            confidence=data.get("confidence"),
            heading_level=data.get("heading_level"),
            list_marker=list_marker,
            table_grid=table_grid,
            group_id=data.get("group_id"),
            failure_reason=data.get("failure_reason"),
            _meta_items=_freeze_meta(data.get("meta") or {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            {"schema_version": REGION_SCHEMA_VERSION, "region": self.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, blob: str) -> Region:
        envelope = json.loads(blob)
        version = envelope.get("schema_version")
        if version != REGION_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported Region schema version {version!r}; "
                f"this build supports {REGION_SCHEMA_VERSION!r}"
            )
        return cls.from_dict(envelope["region"])


def iter_pages(regions: list[Region]) -> Iterator[tuple[int, list[Region]]]:
    """Yield ``(page_index, regions_on_page)`` tuples in page order.

    Within each page the input order is preserved (reading-order
    reconstruction is phase 6, not phase 2).
    """
    if not regions:
        return
    by_page: dict[int, list[Region]] = {}
    for region in regions:
        by_page.setdefault(region.page_index, []).append(region)
    for page_index in sorted(by_page):
        yield page_index, by_page[page_index]


__all__ = [
    "REGION_SCHEMA_VERSION",
    "BBox",
    "ListKind",
    "ListMarker",
    "Region",
    "RegionRole",
    "RegionSource",
    "TableCell",
    "TableGrid",
    "iter_pages",
]
