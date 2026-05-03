"""Layout-detection adapter package.

The pipeline's ML branch runs in two stages: (1) layout detection on a
rasterised page (this package, phase 4), (2) per-region OCR (phase 5).

The :class:`LayoutDetector` Protocol is the integration boundary. The
orchestrator depends only on the Protocol; concrete adapters
(:mod:`arabic_pdf_transcribe.layout.full_page` and
:mod:`arabic_pdf_transcribe.layout.doclayout_yolo`) implement it. The
adapter is selected at the CLI by ``--layout`` and instantiated lazily
so the heavy optional deps load only when the ML branch runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from arabic_pdf_transcribe.regions import Region


@runtime_checkable
class LayoutDetector(Protocol):
    """Detect document-layout regions on a single page image.

    Implementations return a sequence of :class:`Region` instances whose
    ``bbox`` is in the same coordinate system as ``page_image`` (top-left
    origin, pixel units), ``text`` is the empty string (text is filled by
    phase 5's OCR step), ``source`` is :data:`RegionSource.OCR`, and
    ``role`` is the project ``RegionRole`` mapped from the model's class
    label. ``confidence`` is the detector's per-region score where the
    model exposes one, ``None`` otherwise.

    For ``role == TABLE`` regions the adapter is responsible for filling
    ``table_grid`` with a :class:`TableGrid` whose ``TableCell.text`` are
    empty strings; phase 5 fills cell text per cell.

    Implementations must raise
    :class:`arabic_pdf_transcribe.errors.ModelDownloadError` (mappable to
    CLI exit code 5) when the model is missing from the local cache and
    download is not possible (offline mode, no network, etc.).
    """

    def detect(self, page_image: PILImage, page_index: int) -> Sequence[Region]:
        """Return regions detected on ``page_image``."""
        ...


__all__ = ["LayoutDetector"]
