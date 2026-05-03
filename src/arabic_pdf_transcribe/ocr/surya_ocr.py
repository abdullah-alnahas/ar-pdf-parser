"""Surya OCR transcriber.

Surya (datalab-to/surya) is a multilingual line-level OCR model that
emerged as the body-text winner in the model survey at
notebooks/00_model_survey.ipynb. It is the default OCR backend.

Surya owns its own line detection internally, so this adapter crops the
region from the page, runs surya, and joins the recognised lines with
newlines. ``role == FIGURE`` is excluded by the pipeline before we are
called, so we do not special-case it here.
"""

from __future__ import annotations

from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, TableCell, TableGrid


def _crop(image: PILImage, bbox: BBox) -> PILImage:
    return image.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)))


class SuryaOCRTranscriber:
    """OCR transcriber backed by surya.

    Loads the surya predictors lazily on the first call. The same
    predictors are reused across regions and pages.
    """

    def __init__(self) -> None:
        self._rec: object | None = None
        self._det: object | None = None

    def _load(self) -> tuple[object, object]:
        if self._rec is not None and self._det is not None:
            return self._rec, self._det
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise ModelDownloadError(
                "surya backend requires the 'surya-ocr' package; install with: "
                "pip install 'surya-ocr'"
            ) from exc
        try:
            foundation = FoundationPredictor()
            self._rec = RecognitionPredictor(foundation)
            self._det = DetectionPredictor()
        except Exception as exc:
            raise ModelDownloadError(
                f"failed to initialise surya predictors: {exc}"
            ) from exc
        return self._rec, self._det

    def _ocr_image(self, image: PILImage) -> str:
        rec, det = self._load()
        try:
            results = rec([image], det_predictor=det)  # type: ignore[operator]
        except Exception as exc:
            raise OCRTranscriptionError(f"surya failed: {exc}") from exc
        if not results:
            return ""
        return "\n".join(line.text for line in results[0].text_lines)

    def transcribe(self, region: Region, page_image: PILImage) -> Region:
        if region.role is RegionRole.FIGURE:
            return region
        if region.role is RegionRole.TABLE and region.table_grid is not None:
            new_rows: list[tuple[TableCell, ...]] = []
            for row in region.table_grid.rows:
                new_cells = []
                for cell in row:
                    text = self._ocr_image(_crop(page_image, cell.bbox))
                    new_cells.append(
                        TableCell(text=text, confidence=None, bbox=cell.bbox)
                    )
                new_rows.append(tuple(new_cells))
            return region.with_table_grid(TableGrid(rows=tuple(new_rows)))
        text = self._ocr_image(_crop(page_image, region.bbox))
        return region.with_text(text)


__all__ = ["SuryaOCRTranscriber"]
