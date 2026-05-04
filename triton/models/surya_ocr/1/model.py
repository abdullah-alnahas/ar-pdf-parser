"""Triton Python backend wrapping the surya OCR pipeline.

Each model instance owns its own copy of the surya predictors
(``FoundationPredictor`` + ``RecognitionPredictor`` + ``DetectionPredictor``).
``instance_group.count`` in ``config.pbtxt`` controls how many copies
run on the GPU; Triton dispatches concurrent requests across instances
on independent CUDA streams, which is what gives us actual parallelism
(separate from surya's internal batching, which we still use *within*
each request).

Wire format
-----------
Inputs:
    IMAGE_PNG : 1-D TYPE_STRING — N PNG byte buffers, one per image.
    MATH_MODE : optional 1×1 TYPE_BOOL — when False, ``math_mode=False``
                is forwarded to surya so LaTeX-y output is suppressed.
    RECOGNITION_BATCH_SIZE : optional 1×1 TYPE_INT32 — surya's internal
                recognition + detection batch size.

Outputs:
    TEXT : 1-D TYPE_STRING — N transcribed strings, lines joined by ``\n``.

Each request is independent (max_batch_size=0); cross-request batching
is delegated to running multiple ``instance_group`` copies. Within a
request, surya's internal batcher handles the N images.
"""

from __future__ import annotations

import io
import os

from typing import Any

import numpy as np
import triton_python_backend_utils as pb_utils  # type: ignore[import-not-found]
from PIL import Image


class TritonPythonModel:
    rec: Any
    det: Any

    def initialize(self, args: dict) -> None:  # noqa: ARG002 — Triton signature
        # Honour FORCE_CPU=1 from config parameters or environment so we
        # can point a CPU box at this same model dir for smoke tests.
        if os.environ.get("FORCE_CPU") == "1":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor

        foundation = FoundationPredictor()
        self.rec = RecognitionPredictor(foundation)
        self.det = DetectionPredictor()

    def execute(self, requests: list) -> list:
        responses = []
        for request in requests:
            try:
                responses.append(self._run_one(request))
            except Exception as exc:  # noqa: BLE001 — bubble to client as error
                err = pb_utils.TritonError(f"surya_ocr backend failed: {exc}")
                responses.append(pb_utils.InferenceResponse(error=err))
        return responses

    def _run_one(self, request: object) -> object:
        png_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE_PNG")
        if png_tensor is None:
            raise ValueError("missing IMAGE_PNG input")
        png_bytes = png_tensor.as_numpy()
        # ``as_numpy()`` for TYPE_STRING returns object dtype with bytes/str.
        images: list[Image.Image] = []
        for entry in png_bytes:
            buf = entry if isinstance(entry, (bytes, bytearray)) else entry.encode("latin-1")
            images.append(Image.open(io.BytesIO(buf)).convert("RGB"))

        math_mode = True
        math_tensor = pb_utils.get_input_tensor_by_name(request, "MATH_MODE")
        if math_tensor is not None:
            math_mode = bool(math_tensor.as_numpy()[0])

        kwargs: dict = {"det_predictor": self.det, "math_mode": math_mode}
        bs_tensor = pb_utils.get_input_tensor_by_name(request, "RECOGNITION_BATCH_SIZE")
        if bs_tensor is not None:
            bs = int(bs_tensor.as_numpy()[0])
            if bs > 0:
                kwargs["recognition_batch_size"] = bs
                kwargs["detection_batch_size"] = bs

        if not images:
            texts: list[str] = []
        else:
            results = self.rec(images, **kwargs)
            texts = ["\n".join(line.text for line in r.text_lines) for r in results]

        out_arr = np.array(texts, dtype=object)
        out_tensor = pb_utils.Tensor("TEXT", out_arr)
        return pb_utils.InferenceResponse(output_tensors=[out_tensor])

    def finalize(self) -> None:
        # Drop predictor refs so CUDA frees on shutdown.
        self.rec = None  # type: ignore[assignment]
        self.det = None  # type: ignore[assignment]
