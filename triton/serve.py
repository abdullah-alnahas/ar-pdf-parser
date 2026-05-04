"""PyTriton launcher hosting the surya OCR model.

PyTriton wraps a Python function in a real Triton server (same gRPC
wire format as the C++ build) without the full model-repository /
config.pbtxt / docker dance. The CLI's :class:`TritonOCRTranscriber`
talks to this server unchanged on ``localhost:8001``.

Why this exists alongside ``triton/models/``
--------------------------------------------
* ``triton/models/`` + ``Dockerfile`` + ``run.sh`` — production path.
  Container-isolated, multi-instance, Model Analyzer compatible.
  Needs nvidia-container-toolkit installed on the host.
* ``triton/serve.py`` (this file) — local-dev / single-host path. No
  docker required. Runs in the same Python env as the CLI.

Run::

    pip install '.[ml,triton-server]'
    python triton/serve.py --instances 2 --port 8001

Then point the CLI at it::

    arabic-pdf-transcribe in.pdf -o out.docx \\
        --ocr triton --triton-url localhost:8001 \\
        --max-workers 4 --batch-size 32

Wire format mirrors ``triton/models/surya_ocr/config.pbtxt``:

* IMAGE_PNG : 1-D bytes — one PNG buffer per region
* MATH_MODE : 1×1 bool — ``False`` ⇒ surya runs without LaTeX
* RECOGNITION_BATCH_SIZE : 1×1 int32 (optional)
* TEXT      : 1-D bytes — one transcribed string per input image
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
from typing import Any

import numpy as np

# Lazy: import surya only after CLI parse so ``--help`` is fast.

logger = logging.getLogger("apt.triton.serve")


def _make_predictors() -> tuple[Any, Any]:
    from surya.detection import DetectionPredictor
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor

    foundation = FoundationPredictor()
    rec = RecognitionPredictor(foundation)
    det = DetectionPredictor()
    return rec, det


def _decode_png_array(arr: np.ndarray) -> list[Any]:
    from PIL import Image

    images = []
    for entry in arr:
        buf = entry if isinstance(entry, (bytes, bytearray)) else bytes(entry)
        images.append(Image.open(io.BytesIO(buf)).convert("RGB"))
    return images


def build_infer_fn(rec: Any, det: Any):
    """Return the ``infer_fn`` PyTriton invokes per request batch."""

    def infer_fn(requests: list[dict]) -> list[dict]:
        responses = []
        for req in requests:
            try:
                png_in = req["IMAGE_PNG"]
                # PyTriton hands inputs as 2-D arrays with a leading
                # batch dim of 1 when max_batch_size>0; we configured
                # max_batch_size=0 so no leading batch dim is added.
                images = _decode_png_array(png_in.reshape(-1))

                math_mode = True
                if "MATH_MODE" in req:
                    math_mode = bool(req["MATH_MODE"].reshape(-1)[0])

                kwargs: dict[str, Any] = {
                    "det_predictor": det,
                    "math_mode": math_mode,
                }
                if "RECOGNITION_BATCH_SIZE" in req:
                    bs = int(req["RECOGNITION_BATCH_SIZE"].reshape(-1)[0])
                    if bs > 0:
                        kwargs["recognition_batch_size"] = bs
                        kwargs["detection_batch_size"] = bs

                if not images:
                    texts: list[str] = []
                else:
                    results = rec(images, **kwargs)
                    texts = [
                        "\n".join(line.text for line in r.text_lines) for r in results
                    ]

                out_arr = np.array([t.encode("utf-8") for t in texts], dtype=object)
                responses.append({"TEXT": out_arr})
            except Exception as exc:  # noqa: BLE001
                logger.exception("infer_fn failure")
                err = np.array([f"ERROR: {exc}".encode()], dtype=object)
                responses.append({"TEXT": err})
        return responses

    return infer_fn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="triton-serve",
        description="PyTriton server hosting surya OCR on gRPC + HTTP.",
    )
    parser.add_argument(
        "--port", type=int, default=8001, help="gRPC port (default 8001)."
    )
    parser.add_argument(
        "--http-port", type=int, default=8000, help="HTTP port (default 8000)."
    )
    parser.add_argument(
        "--metrics-port", type=int, default=8002, help="Metrics port (default 8002)."
    )
    parser.add_argument(
        "--model-name", type=str, default="surya_ocr", help="Triton model name."
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help=(
            "Number of model instances on the GPU. Bump (2-4) for real "
            "concurrent execution on independent CUDA streams; each "
            "instance holds its own surya predictor copy."
        ),
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU (CUDA_VISIBLE_DEVICES='')."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        from pytriton.model_config import ModelConfig, Tensor
        from pytriton.triton import Triton, TritonConfig
    except ImportError:
        sys.stderr.write(
            "error: pytriton is not installed; run: pip install nvidia-pytriton\n"
        )
        return 2

    logger.info("loading surya predictors (one set per instance)…")
    # PyTriton spawns the infer_fn in this process; instance_count
    # controls how many parallel calls it can dispatch. We reuse one
    # predictor set across instances — surya's torch ops release the
    # GIL during forward, so concurrent calls overlap on the GPU
    # naturally.
    rec, det = _make_predictors()
    infer_fn = build_infer_fn(rec, det)

    triton_config = TritonConfig(
        grpc_port=args.port,
        http_port=args.http_port,
        metrics_port=args.metrics_port,
        log_verbose=1 if args.log_level == "DEBUG" else 0,
    )

    with Triton(config=triton_config) as triton:
        triton.bind(
            model_name=args.model_name,
            infer_func=infer_fn,
            inputs=[
                Tensor(name="IMAGE_PNG", dtype=bytes, shape=(-1,)),
                Tensor(name="MATH_MODE", dtype=np.bool_, shape=(1,), optional=True),
                Tensor(
                    name="RECOGNITION_BATCH_SIZE",
                    dtype=np.int32,
                    shape=(1,),
                    optional=True,
                ),
            ],
            outputs=[
                Tensor(name="TEXT", dtype=bytes, shape=(-1,)),
            ],
            config=ModelConfig(
                # PyTriton requires max_batch_size>=1. Each client
                # request already carries N images in IMAGE_PNG, so
                # cross-request batching is unnecessary; we set
                # max_batch_size=1 and let surya batch *within* a
                # request via recognition_batch_size. Cross-request
                # parallelism comes from --instances.
                max_batch_size=1,
            ),
            strict=True,
        )
        logger.info(
            "Triton bound on grpc=%d http=%d metrics=%d (model=%s, instances=%d)",
            args.port,
            args.http_port,
            args.metrics_port,
            args.model_name,
            args.instances,
        )
        triton.serve()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
