"""DocLayout-YOLO layout detector.

Wraps the ``doclayout_yolo`` package's ``YOLOv10`` predictor with the
checkpoint from ``juliozhao/DocLayout-YOLO-DocStructBench``. The model
emits PubLayNet-style classes ("title", "plain text", "figure",
"table", etc.) which we map onto :class:`RegionRole`.

DocLayout-YOLO was tested in the model survey but was found to mis-class
Arabic body text as ``figure`` on stylised pages; we still expose it as a
selectable backend for users whose corpora benefit from explicit region
boxes.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL.Image import Image as PILImage

from arabic_pdf_transcribe.errors import ModelDownloadError
from arabic_pdf_transcribe.regions import BBox, Region, RegionRole, RegionSource


# DocLayout-YOLO DocStructBench class names → project RegionRole.
_LABEL_TO_ROLE: dict[str, RegionRole] = {
    "title": RegionRole.HEADING,
    "plain text": RegionRole.PARAGRAPH,
    "abandon": RegionRole.HEADER_FOOTER,
    "figure": RegionRole.FIGURE,
    "figure_caption": RegionRole.CAPTION,
    "table": RegionRole.TABLE,
    "table_caption": RegionRole.CAPTION,
    "table_footnote": RegionRole.CAPTION,
    "isolate_formula": RegionRole.PARAGRAPH,
    "formula_caption": RegionRole.CAPTION,
}

_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"


class DocLayoutYoloDetector:
    """Layout detector backed by DocLayout-YOLO.

    Loads weights lazily on the first ``detect`` call. Subsequent calls
    reuse the same model in-process.
    """

    def __init__(self, *, device: str = "auto", confidence: float = 0.25) -> None:
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._confidence = confidence
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from doclayout_yolo import YOLOv10
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ModelDownloadError(
                "doclayout-yolo backend requires the 'doclayout-yolo' and "
                "'huggingface_hub' packages; install with: "
                "pip install doclayout-yolo huggingface_hub"
            ) from exc
        try:
            ckpt = hf_hub_download(repo_id=_REPO_ID, filename=_FILENAME)
        except Exception as exc:
            raise ModelDownloadError(
                f"failed to download {_REPO_ID}/{_FILENAME}: {exc}"
            ) from exc
        self._model = YOLOv10(ckpt)
        return self._model

    def detect(self, page_image: PILImage, page_index: int) -> Sequence[Region]:
        model = self._load()
        results = model.predict(  # type: ignore[attr-defined]
            page_image, imgsz=1024, conf=self._confidence, device=self._device
        )
        regions: list[Region] = []
        for result in results:
            names = result.names
            for box, cls, conf in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
            ):
                label = names[int(cls)]
                role = _LABEL_TO_ROLE.get(label, RegionRole.UNKNOWN)
                x0, y0, x1, y1 = (float(v) for v in box)
                regions.append(
                    Region(
                        page_index=page_index,
                        bbox=BBox(x0=x0, y0=y0, x1=x1, y1=y1),
                        text="",
                        role=role,
                        source=RegionSource.OCR,
                        confidence=float(conf),
                    )
                )
        return regions


__all__ = ["DocLayoutYoloDetector"]
