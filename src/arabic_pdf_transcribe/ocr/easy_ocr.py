"""EasyOCR transcriber (Arabic).

EasyOCR with the ``ar`` language model. Tested in the survey at
notebooks/00_model_survey.ipynb; produces real Arabic with some typos.
Useful as a CPU-friendly fallback when surya is unavailable or too slow.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, TableCell, TableGrid


def _crop(image: PILImage, bbox: BBox) -> PILImage:
    return image.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)))


class EasyOCRTranscriber:
    """OCR transcriber backed by EasyOCR with Arabic weights."""

    def __init__(
        self,
        *,
        device: str = "auto",
        disable_formula: bool = False,
        batch_size: int | None = None,
    ) -> None:
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._gpu = device == "cuda"
        # ``disable_formula`` / ``batch_size`` are accepted for API
        # symmetry with the surya adapter. EasyOCR does not emit LaTeX,
        # and its public API does not surface a batch-size knob — both
        # are no-ops here.
        self._disable_formula = disable_formula
        self._batch_size = batch_size
        self._reader: object | None = None

    def _load(self) -> object:
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelDownloadError(
                "easyocr-ara backend requires the 'easyocr' package; install with: "
                "pip install easyocr"
            ) from exc
        try:
            self._reader = easyocr.Reader(["ar"], gpu=self._gpu)
        except Exception as exc:
            raise ModelDownloadError(f"failed to initialise easyocr: {exc}") from exc
        return self._reader

    def _ocr_image(self, image: PILImage) -> str:
        import numpy as np

        reader = self._load()
        try:
            results = reader.readtext(  # type: ignore[attr-defined]
                np.array(image), detail=1, paragraph=True
            )
        except Exception as exc:
            raise OCRTranscriptionError(f"easyocr failed: {exc}") from exc
        return "\n".join(r[1] for r in results)

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

    def transcribe_page(
        self, regions: Sequence[Region], page_image: PILImage
    ) -> list[Region]:
        return [self.transcribe(r, page_image) for r in regions]


__all__ = ["EasyOCRTranscriber"]
