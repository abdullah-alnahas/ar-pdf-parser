"""Triton-Inference-Server-backed OCR transcriber.

The pipeline talks to a Triton server hosting the ``surya_ocr`` Python
backend (see ``triton/models/surya_ocr/``) over gRPC. The server owns
the predictors; multiple ``instance_group`` copies execute in parallel
on independent CUDA streams, so concurrent CLI / FastAPI clients get
real GPU concurrency without any in-process model lifecycle.

Same :class:`OCRTranscriber` contract as the in-process surya adapter:

* :meth:`transcribe` — single region.
* :meth:`transcribe_page` — every text region on a page in one gRPC
  round-trip.

When ``disable_formula=True`` the adapter forwards ``MATH_MODE=False``
to the backend and runs the same defensive math-stripping regex as the
in-process adapter as a belt-and-braces step.

Usage
-----

Start Triton::

    docker run --gpus=all -p 8001:8001 -v "$PWD/triton/models:/models" \\
        nvcr.io/nvidia/tritonserver:24.10-py3 \\
        tritonserver --model-repository=/models

Then::

    arabic-pdf-transcribe in.pdf -o out.docx \\
        --ocr triton --triton-url localhost:8001 \\
        --max-workers 8 --batch-size 16
"""

from __future__ import annotations

import io
from collections.abc import Sequence

import numpy as np
from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.errors import ModelDownloadError, OCRTranscriptionError
from arabic_pdf_transcribe.ocr.surya_ocr import _strip_math
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, TableCell, TableGrid


def _crop(image: PILImage, bbox: BBox) -> PILImage:
    return image.crop((int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)))


def _png_bytes(image: PILImage) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class TritonOCRTranscriber:
    """Surya-via-Triton OCR transcriber.

    Parameters
    ----------
    url:
        ``host:port`` of the Triton gRPC endpoint. Default
        ``localhost:8001``.
    model_name:
        Name of the model in the Triton repository. Default
        ``surya_ocr``.
    disable_formula:
        Forward ``MATH_MODE=False`` to the backend and strip residual
        math markup from results.
    batch_size:
        Forwarded to the backend as ``RECOGNITION_BATCH_SIZE``; the
        backend uses it for surya's internal recognition + detection
        batch.
    timeout:
        Per-call gRPC timeout, in seconds. Default 600.
    """

    def __init__(
        self,
        *,
        url: str = "localhost:8001",
        model_name: str = "surya_ocr",
        disable_formula: bool = False,
        batch_size: int | None = None,
        timeout: float = 600.0,
    ) -> None:
        self._url = url
        self._model_name = model_name
        self._disable_formula = disable_formula
        self._batch_size = batch_size
        self._timeout = timeout
        self._client: object | None = None

    def _get_client(self) -> object:
        if self._client is not None:
            return self._client
        try:
            import tritonclient.grpc as grpcclient  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ModelDownloadError(
                "triton OCR backend requires 'tritonclient[grpc]'; install with: "
                "pip install 'tritonclient[grpc]'"
            ) from exc
        try:
            client = grpcclient.InferenceServerClient(url=self._url)
            if not client.is_server_live():
                raise ModelDownloadError(f"Triton server at {self._url!r} is not live")
            if not client.is_model_ready(self._model_name):
                raise ModelDownloadError(
                    f"Triton model {self._model_name!r} is not ready on {self._url!r}"
                )
        except ModelDownloadError:
            raise
        except Exception as exc:
            raise ModelDownloadError(
                f"failed to connect to Triton at {self._url!r}: {exc}"
            ) from exc
        self._client = client
        return client

    def _infer(self, images: list[PILImage]) -> list[str]:
        if not images:
            return []
        import tritonclient.grpc as grpcclient  # type: ignore[import-not-found]

        client = self._get_client()
        png_arr = np.array([_png_bytes(im) for im in images], dtype=object)

        in_img = grpcclient.InferInput("IMAGE_PNG", [len(images)], "BYTES")
        in_img.set_data_from_numpy(png_arr)

        in_math = grpcclient.InferInput("MATH_MODE", [1], "BOOL")
        in_math.set_data_from_numpy(np.array([not self._disable_formula], dtype=bool))

        inputs = [in_img, in_math]
        if self._batch_size is not None:
            in_bs = grpcclient.InferInput("RECOGNITION_BATCH_SIZE", [1], "INT32")
            in_bs.set_data_from_numpy(np.array([int(self._batch_size)], dtype=np.int32))
            inputs.append(in_bs)

        out = grpcclient.InferRequestedOutput("TEXT")

        try:
            result = client.infer(  # type: ignore[attr-defined]
                model_name=self._model_name,
                inputs=inputs,
                outputs=[out],
                client_timeout=self._timeout,
            )
        except Exception as exc:
            raise OCRTranscriptionError(f"triton infer failed: {exc}") from exc

        texts_arr = result.as_numpy("TEXT")
        if texts_arr is None:
            raise OCRTranscriptionError("triton response missing TEXT output")
        out_texts: list[str] = []
        for entry in texts_arr:
            text = entry.decode("utf-8") if isinstance(entry, (bytes, bytearray)) else str(entry)
            if self._disable_formula:
                text = _strip_math(text)
            out_texts.append(text)
        return out_texts

    def transcribe(self, region: Region, page_image: PILImage) -> Region:
        if region.role is RegionRole.FIGURE:
            return region
        if region.role is RegionRole.TABLE and region.table_grid is not None:
            cells_flat: list[PILImage] = []
            shape: list[int] = []
            for row in region.table_grid.rows:
                shape.append(len(row))
                for cell in row:
                    cells_flat.append(_crop(page_image, cell.bbox))
            texts = self._infer(cells_flat)
            new_rows: list[tuple[TableCell, ...]] = []
            offset = 0
            for r_idx, row in enumerate(region.table_grid.rows):
                width = shape[r_idx]
                row_texts = texts[offset : offset + width]
                offset += width
                new_cells = [
                    TableCell(text=t, confidence=None, bbox=row[i].bbox)
                    for i, t in enumerate(row_texts)
                ]
                new_rows.append(tuple(new_cells))
            return region.with_table_grid(TableGrid(rows=tuple(new_rows)))
        [text] = self._infer([_crop(page_image, region.bbox)])
        return region.with_text(text)

    def transcribe_page(
        self, regions: Sequence[Region], page_image: PILImage
    ) -> list[Region]:
        passthrough: dict[int, Region] = {}
        text_indices: list[int] = []
        text_crops: list[PILImage] = []
        for idx, region in enumerate(regions):
            if region.role is RegionRole.FIGURE:
                passthrough[idx] = region
                continue
            if region.role is RegionRole.TABLE and region.table_grid is not None:
                # Tables go through the per-region path so the cell shape is
                # preserved; one extra round-trip per table is cheap.
                passthrough[idx] = self.transcribe(region, page_image)
                continue
            text_indices.append(idx)
            text_crops.append(_crop(page_image, region.bbox))

        out: list[Region | None] = [None] * len(regions)
        for idx, region in passthrough.items():
            out[idx] = region

        if text_crops:
            texts = self._infer(text_crops)
            for slot, idx in enumerate(text_indices):
                text = texts[slot] if slot < len(texts) else ""
                out[idx] = regions[idx].with_text(text)

        return [r for r in out if r is not None]


__all__ = ["TritonOCRTranscriber"]
