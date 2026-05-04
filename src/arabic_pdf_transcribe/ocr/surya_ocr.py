"""Surya OCR transcriber.

Surya (datalab-to/surya) is a multilingual line-level OCR model that
emerged as the body-text winner in the model survey at
notebooks/00_model_survey.ipynb. It is the default OCR backend.

Surya owns its own line detection internally, so this adapter crops the
region from the page, runs surya, and joins the recognised lines with
newlines. ``role == FIGURE`` is excluded by the pipeline before we are
called, so we do not special-case it here.

Performance knobs
-----------------

* ``batch_size`` — forwarded to surya's ``recognition_batch_size`` and
  ``detection_batch_size`` so callers can tune GPU throughput from the
  CLI without touching environment variables.
* :meth:`transcribe_page` — single batched call across every non-figure
  region on a page, materially faster than per-region calls when the
  page has many regions (one model invocation, one warm-up cost).

Formula handling
----------------

Surya emits ``<math>...</math>`` markup for math-like crops by default.
When ``disable_formula=True`` the adapter passes ``math_mode=False`` to
surya and runs a defensive regex pass that strips any residual math
markup or stray LaTeX delimiters (``$...$``, ``\\(...\\)``, ``\\[...\\]``)
so the output is plain text.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, TableCell, TableGrid


def _crop(image: PILImage, bbox: BBox) -> PILImage:
    return image.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)))


# Defensive regexes for ``disable_formula``: surya may still emit math
# fragments even with ``math_mode=False`` for some inputs, and native
# extraction occasionally surfaces inline TeX. Stripping is conservative —
# only the wrappers go; the inner glyphs are kept as plain text.
_MATH_TAG = re.compile(r"<\s*math\b[^>]*>(.*?)<\s*/\s*math\s*>", re.DOTALL | re.IGNORECASE)
_TEX_INLINE = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_TEX_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
_DOLLAR_INLINE = re.compile(r"(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$")
_DOLLAR_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def _strip_math(text: str) -> str:
    text = _MATH_TAG.sub(lambda m: m.group(1), text)
    text = _DOLLAR_DISPLAY.sub(lambda m: m.group(1), text)
    text = _TEX_DISPLAY.sub(lambda m: m.group(1), text)
    text = _TEX_INLINE.sub(lambda m: m.group(1), text)
    text = _DOLLAR_INLINE.sub(lambda m: m.group(1), text)
    return text


class SuryaOCRTranscriber:
    """OCR transcriber backed by surya.

    Loads the surya predictors lazily on the first call. The same
    predictors are reused across regions and pages.
    """

    def __init__(
        self,
        *,
        disable_formula: bool = False,
        batch_size: int | None = None,
    ) -> None:
        self._rec: object | None = None
        self._det: object | None = None
        self._disable_formula = disable_formula
        self._batch_size = batch_size

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

    def _call_surya(self, images: list[PILImage]) -> list[object]:
        rec, det = self._load()
        kwargs: dict[str, object] = {"det_predictor": det}
        kwargs["math_mode"] = not self._disable_formula
        if self._batch_size is not None:
            kwargs["recognition_batch_size"] = self._batch_size
            kwargs["detection_batch_size"] = self._batch_size
        try:
            return rec(images, **kwargs)  # type: ignore[operator,no-any-return]
        except Exception as exc:
            raise OCRTranscriptionError(f"surya failed: {exc}") from exc

    def _join_result(self, result: object) -> str:
        text = "\n".join(line.text for line in result.text_lines)  # type: ignore[attr-defined]
        if self._disable_formula:
            text = _strip_math(text)
        return text

    def _ocr_image(self, image: PILImage) -> str:
        results = self._call_surya([image])
        if not results:
            return ""
        return self._join_result(results[0])

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
        """Batched OCR for every region on one page in a single surya call.

        Tables fall back to the per-cell loop because each cell needs an
        independent crop+result; figures pass through untouched. All
        remaining (text-bearing) regions go to surya as one image batch
        so detection and recognition amortise their warm-up.
        """
        passthrough: dict[int, Region] = {}
        text_indices: list[int] = []
        text_crops: list[PILImage] = []
        for idx, region in enumerate(regions):
            if region.role is RegionRole.FIGURE:
                passthrough[idx] = region
                continue
            if region.role is RegionRole.TABLE and region.table_grid is not None:
                passthrough[idx] = self.transcribe(region, page_image)
                continue
            text_indices.append(idx)
            text_crops.append(_crop(page_image, region.bbox))

        out: list[Region | None] = [None] * len(regions)
        for idx, region in passthrough.items():
            out[idx] = region

        if text_crops:
            results = self._call_surya(text_crops)
            for slot, idx in enumerate(text_indices):
                if slot < len(results):
                    text = self._join_result(results[slot])
                else:
                    text = ""
                out[idx] = regions[idx].with_text(text)

        # ``out`` is fully populated (every input slot was assigned).
        return [r for r in out if r is not None]


__all__ = ["SuryaOCRTranscriber"]
