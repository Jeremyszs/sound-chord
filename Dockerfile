# CPU image: python:3.12-slim + ffmpeg + CPU-only torch, models baked in at build time.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE} AS base
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models TORCH_HOME=/app/models/torch

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
# CPU-only torch keeps the image around 1.5 GB instead of 5+ GB with CUDA.
RUN pip install --index-url ${TORCH_INDEX_URL} torch torchaudio

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

COPY scripts ./scripts
# Downloads and verifies BTC (converted to safetensors) and Beat This! weights,
# so containers start without network access.
RUN python scripts/fetch_models.py --models-dir /app/models

RUN useradd --system --uid 10001 app && chown -R app /app/models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/v1/health').status != 200)"
CMD ["chordanalyzer-api"]
