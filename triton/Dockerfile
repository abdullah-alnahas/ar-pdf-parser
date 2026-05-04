# Triton server image baked with surya-ocr deps so cold-starts skip the
# pip install step on every container run.
#
# Build:
#   docker build -t apt-triton triton/
# Run (see triton/run.sh):
#   docker run --gpus=all --shm-size=2g -p 8001:8001 \
#     -v "$PWD/triton/models:/models" \
#     -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
#     apt-triton tritonserver --model-repository=/models

FROM nvcr.io/nvidia/tritonserver:24.10-py3

# Pinned to match src/arabic_pdf_transcribe/ocr/surya_ocr.py expectations.
# surya-ocr brings its own torch / transformers; keep the stack inside the
# container insulated from the host's [ml] pin so versions can drift
# independently.
RUN pip install --no-cache-dir \
        "surya-ocr" \
        "numpy<2"

# HF cache mounts in at runtime via -v; this just ensures the dir exists
# so first-run downloads land in the bind-mount, not in the image layer.
ENV HF_HOME=/root/.cache/huggingface

EXPOSE 8001 8000 8002
