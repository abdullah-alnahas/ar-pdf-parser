"""OCR / VLM transcription adapter package.

The pipeline's ML branch runs in two stages: (1) layout detection on a
rasterised page (phase 4 — :mod:`arabic_pdf_transcribe.layout`),
(2) per-region OCR (this package, phase 5).

The :class:`OCRTranscriber` Protocol is the integration boundary. The
orchestrator (phase 8) depends only on the Protocol; concrete adapters
(currently :mod:`arabic_pdf_transcribe.ocr.hf_ocr`) implement it. This
keeps ``transformers`` and ``torch`` out of the import graph until the
ML branch actually runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from arabic_pdf_transcribe.regions import Region


@runtime_checkable
class OCRTranscriber(Protocol):
    """Fill in the ``text`` of a single :class:`Region`.

    Contract:

    * For ``role == FIGURE`` regions the adapter must NOT call the OCR
      model — it returns the input unchanged (figures stay
      ``text=""``).
    * For ``role == CAPTION``, ``PARAGRAPH``, ``HEADING``, ``LIST_ITEM``
      and ``HEADER_FOOTER`` regions, the adapter crops the bbox out of
      ``page_image`` and returns a new :class:`Region` with ``text``
      filled and ``confidence`` set (``None`` when the model does not
      expose a confidence score).
    * For ``role == TABLE`` regions, the adapter walks
      ``region.table_grid.rows[i].cells[j]``, OCRs each cell's bbox,
      and returns a new :class:`Region` whose ``table_grid`` is fully
      populated (each :class:`TableCell` carries ``text`` and
      ``confidence``). The outer Region's ``text`` stays empty;
      table content lives in cells.

    Implementations must raise
    :class:`arabic_pdf_transcribe.errors.OCRTranscriptionError` on
    decoder failure or model error. Phase 8's orchestrator maps the
    exception to a per-page failure or, with ``--strict``, to a
    pipeline abort.

    Implementations must raise
    :class:`arabic_pdf_transcribe.errors.ModelDownloadError` (mappable
    to CLI exit code 5) when the model is missing from the local
    cache and download is not possible (offline mode, no network).
    """

    def transcribe(self, region: Region, page_image: PILImage) -> Region:
        """Return ``region`` with text / confidence filled."""
        ...


__all__ = ["OCRTranscriber"]
