# syntax=docker/dockerfile:1.7
# One Dockerfile, three stages. `app` and `tts` both build FROM `base`, so the
# multi-GB torch/CUDA layer is stored once and shared by both images.

FROM python:3.11-slim-bookworm AS base
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    HOME=/tmp/home \
    XDG_CACHE_HOME=/assets/cache \
    HF_HOME=/assets/models/hf \
    TORCH_HOME=/assets/models/torch \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg rubberband-cli libsndfile1 git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv==0.8.22 && uv venv /opt/venv --python /usr/local/bin/python3.11
RUN uv pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
RUN mkdir -p /tmp/home && chmod 1777 /tmp/home


# ---------------------------------------------------------------- TTS service
FROM base AS tts
ARG INDEXTTS_REF=d9e41aac89fd00b3d71497fddb287b7f24613712
WORKDIR /opt/index-tts
RUN git init -q . && git remote add origin https://github.com/index-tts/index-tts.git \
    && git fetch -q --depth 1 origin ${INDEXTTS_REF} && git checkout -q FETCH_HEAD \
    && rm -rf .git examples tests archive
# Install IndexTTS's locked deps into the shared venv (torch already satisfied by base).
RUN uv export --frozen --no-dev --no-hashes --no-emit-project -o /tmp/req.txt \
    && uv pip install -r /tmp/req.txt \
        --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
    && uv pip install --no-deps -e . \
    && uv pip install "fastapi>=0.115" "uvicorn[standard]>=0.30"
# Build the text-normalizer FST caches now: at runtime the container is non-root and
# IndexTTS would try to write them inside site-packages.
RUN python -c "from indextts.utils.front import TextNormalizer; TextNormalizer().load()" \
    && chmod -R a+rwX /opt/index-tts/indextts/utils /opt/venv/lib/python3.11/site-packages/tn
COPY services/tts/server.py /opt/tts/server.py
WORKDIR /opt/tts
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]


# ---------------------------------------------------------------- Main app (CLI + web UI + pipeline)
FROM base AS app
COPY requirements-app.txt /tmp/requirements-app.txt
RUN uv pip install -r /tmp/requirements-app.txt
# ctranslate2 (faster-whisper) needs cuBLAS/cuDNN; reuse the copies torch ships.
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.11/site-packages/nvidia/cublas/lib:/opt/venv/lib/python3.11/site-packages/nvidia/cudnn/lib
WORKDIR /app
COPY pyproject.toml /app/
COPY models /app/models
COPY subarashii /app/subarashii
RUN uv pip install --no-deps -e /app
EXPOSE 7860
CMD ["subarashii", "serve", "--host", "0.0.0.0", "--port", "7860"]
