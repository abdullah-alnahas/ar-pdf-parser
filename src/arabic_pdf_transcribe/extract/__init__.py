"""Native PDF extraction (text-layer path).

Phase 2 lands :func:`arabic_pdf_transcribe.extract.native.extract_native`
plus a basic native-table detector. The ML branch (layout detection +
per-region OCR) lives in ``arabic_pdf_transcribe.layout`` and
``arabic_pdf_transcribe.ocr`` and is wired up in phases 4 and 5.
"""

from arabic_pdf_transcribe.extract.native import NativePage, extract_native

__all__ = ["NativePage", "extract_native"]
