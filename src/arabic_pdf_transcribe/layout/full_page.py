"""Full-page pseudo-layout detector.

Returns one PARAGRAPH region covering the entire rasterised page. Use this
when the OCR backend does its own internal layout (e.g. surya). Selected
in the model survey at notebooks/00_model_survey.ipynb as the default
because adding a separate layout pass before surya did not improve
accuracy on the test corpus.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


class FullPageLayoutDetector:
    """Single-region "layout": the whole page is one PARAGRAPH."""

    def detect(self, page_image: PILImage, page_index: int) -> Sequence[Region]:
        w, h = page_image.size
        return [
            Region(
                page_index=page_index,
                bbox=BBox(x0=0.0, y0=0.0, x1=float(w), y1=float(h)),
                text="",
                role=RegionRole.PARAGRAPH,
                source=RegionSource.OCR,
            )
        ]


__all__ = ["FullPageLayoutDetector"]
