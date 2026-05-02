"""Phase-4 layout-detection Protocol + class-mapping tests.

These tests do not require ``transformers`` / ``torch`` / ``Pillow``.
They cover:

* :class:`LayoutDetector` Protocol conformance via ``isinstance`` (the
  Protocol is ``runtime_checkable``).
* The class-label → :class:`RegionRole` mapping in
  :mod:`arabic_pdf_transcribe.layout._classes`.

The HF-backed adapter and image-pipeline tests live in their own
files so this one stays cheap.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from arabic_pdf_transcribe.layout import LayoutDetector
from arabic_pdf_transcribe.layout._classes import (
    DIT_LAYOUT_LABELS,
    DROPPED_LABELS,
    LABEL_HEADING_LEVEL,
    LABEL_TO_ROLE,
    heading_level_for_label,
    role_for_label,
)
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


class _StubDetector:
    """Minimal Protocol-conforming detector used in shape tests."""

    def detect(self, page_image: object, page_index: int) -> Sequence[Region]:
        return ()


class _NotADetector:
    """Has the wrong method shape on purpose."""

    def detect_other(self) -> None:
        pass


def test_layout_detector_protocol_is_runtime_checkable() -> None:
    assert isinstance(_StubDetector(), LayoutDetector)


def test_layout_detector_protocol_rejects_wrong_shape() -> None:
    assert not isinstance(_NotADetector(), LayoutDetector)


# ---------------------------------------------------------------------------
# class-label mapping
# ---------------------------------------------------------------------------


def test_dit_label_set_matches_documented_classes() -> None:
    """The 12 documented DiT labels include all of them."""
    assert "Background" in DIT_LAYOUT_LABELS
    expected = {
        "Background",
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    }
    assert set(DIT_LAYOUT_LABELS) == expected


def test_background_is_dropped_not_mapped() -> None:
    assert "Background" in DROPPED_LABELS
    assert "Background" not in LABEL_TO_ROLE


@pytest.mark.parametrize(
    ("label", "role"),
    [
        ("Caption", RegionRole.CAPTION),
        ("Footnote", RegionRole.HEADER_FOOTER),
        # Issue #16: ``Formula`` remapped from UNKNOWN to PARAGRAPH —
        # English-trained DiT mislabels Arabic body text as Formula,
        # and the previous mapping silently dropped that body text.
        ("Formula", RegionRole.PARAGRAPH),
        ("List-item", RegionRole.LIST_ITEM),
        ("Page-footer", RegionRole.HEADER_FOOTER),
        ("Page-header", RegionRole.HEADER_FOOTER),
        ("Picture", RegionRole.FIGURE),
        ("Section-header", RegionRole.HEADING),
        ("Table", RegionRole.TABLE),
        ("Text", RegionRole.PARAGRAPH),
        ("Title", RegionRole.HEADING),
    ],
)
def test_label_to_role_mapping(label: str, role: RegionRole) -> None:
    assert role_for_label(label) is role


def test_unknown_label_falls_through_to_unknown() -> None:
    assert role_for_label("not-a-real-class") is RegionRole.UNKNOWN


def test_heading_levels() -> None:
    assert heading_level_for_label("Title") == 1
    assert heading_level_for_label("Section-header") == 2
    assert heading_level_for_label("Text") is None
    assert LABEL_HEADING_LEVEL == {"Title": 1, "Section-header": 2}


def test_every_mapped_role_is_a_known_region_role() -> None:
    """Defends against typos like ``RegionRole.HEADIN`` slipping through."""
    for role in LABEL_TO_ROLE.values():
        assert role in RegionRole


# ---------------------------------------------------------------------------
# Region shape produced by stubs (compile-time check that the Protocol's
# return type is sane).
# ---------------------------------------------------------------------------


def test_stub_produces_region_compatible_with_downstream() -> None:
    region = Region(
        page_index=0,
        bbox=BBox(0.0, 0.0, 100.0, 50.0),
        text="",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.OCR,
    )
    assert region.text == ""
    assert region.source is RegionSource.OCR
