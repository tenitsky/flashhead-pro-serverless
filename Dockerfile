# FlashHead Pro serverless worker for RunPod.
# PyTorch 2.10 + CUDA 12.8 (RTX 4090/5090, L4, L40S, A-series, H100), the pinned upstream
# SoulX-FlashHead code, and the ~6.9 GB of models baked in, so a cold start only loads them.
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    FLASHHEAD_ROOT=/opt/SoulX-FlashHead \
    FLASHHEAD_MODELS=/models

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Upstream's pipeline code at the commit the engine was built against. Its requirements.txt
# is not installed (demo and multi-GPU packages); see requirements.txt.
ARG FLASHHEAD_REF=9bc03de06bb0de82cd6bc477804512ae06144bf2
RUN git init -q /opt/SoulX-FlashHead \
 && cd /opt/SoulX-FlashHead \
 && git fetch -q --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git "$FLASHHEAD_REF" \
 && git checkout -q FETCH_HEAD \
 && rm -rf .git \
 && test -d flash_head/src/pipeline

# Python packages, with the image's torch / torchvision / torchaudio pinned so pip cannot
# swap the CUDA build.
COPY requirements.txt /tmp/requirements.txt
RUN python -c "import importlib.metadata as m; print('\n'.join(f'{p}=={m.version(p)}' for p in ('torch', 'torchvision', 'torchaudio')))" > /tmp/torch-pins.txt \
 && cat /tmp/torch-pins.txt \
 && pip install -c /tmp/torch-pins.txt -r /tmp/requirements.txt \
 && python -c "import cv2; assert hasattr(cv2, 'FaceDetectorYN'), cv2.__version__"

# Models in their own layer: code changes below don't re-download 6.9 GB.
COPY download_models.py /opt/worker/download_models.py
RUN python /opt/worker/download_models.py /models

COPY . /opt/worker
WORKDIR /opt/worker
CMD ["python", "-u", "handler.py"]
