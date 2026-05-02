"""Phase-5 OCR Protocol conformance tests.

These tests do not require ``transformers`` / ``torch`` / ``Pillow``.
The Protocol is ``runtime_checkable`` so we exercise ``isinstance``
against minimal stubs.
"""

from __future__ import annotations

from arabic_pdf_transcribe.ocr import OCRTranscriber
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


class _StubTranscriber:
    def transcribe(self, region: Region, page_image: object) -> Region:
        return region


class _NotATranscriber:
    def something_else(self) -> None:
        pass


def test_ocr_transcriber_protocol_is_runtime_checkable() -> None:
    assert isinstance(_StubTranscriber(), OCRTranscriber)


def test_ocr_transcriber_protocol_rejects_wrong_shape() -> None:
    assert not isinstance(_NotATranscriber(), OCRTranscriber)


def test_region_carries_confidence_helper() -> None:
    """Phase 5 added :meth:`Region.with_confidence` for adapters that
    fill the field after construction. Verify the helper works as a
    pure functional update (frozen Region returns a new instance)."""
    region = Region(
        page_index=0,
        bbox=BBox(0.0, 0.0, 100.0, 50.0),
        text="",
        role=RegionRole.PARAGRAPH,
        source=RegionSource.OCR,
    )
    assert region.confidence is None
    enriched = region.with_confidence(0.87)
    assert enriched.confidence == 0.87
    # Original frozen instance is not mutated.
    assert region.confidence is None
    # Other fields are preserved.
    assert enriched.text == region.text
    assert enriched.role is region.role
    # None round-trips.
    assert enriched.with_confidence(None).confidence is None
