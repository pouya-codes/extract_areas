# syntax=docker/dockerfile:1.7
#
# One Dockerfile, one image per service. Build a target with:
#
#   docker build -f docker/Dockerfile --target gateway          -t aim-inference-gateway .
#   docker build -f docker/Dockerfile --target worker-hovernet  -t aim-inference-worker-hovernet .
#   docker build -f docker/Dockerfile --target worker-deepliif  -t aim-inference-worker-deepliif .
#   docker build -f docker/Dockerfile --target worker-ec        -t aim-inference-worker-ec .
#   docker build -f docker/Dockerfile --target worker-pdac      -t aim-inference-worker-pdac .
#   docker build -f docker/Dockerfile --target worker-stub      -t aim-inference-worker-stub .
#   docker build -f docker/Dockerfile --target worker-base      -t aim-inference-worker-base .
#   docker build -f docker/Dockerfile --target tissue-api       -t aim-inference-tissue-api .
#
# or ``docker/build.sh`` for all of them. Model weights are NOT in the images;
# they are mounted at /models (see deploy/k8s/base/config.container.json).
#
# worker-base is the published shared runtime (CUDA + torch + worker/). Per-model
# GitHub repos build FROM aimlabacr.azurecr.io/aimlab/aim-inference-worker-base.

ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126

# --------------------------------------------------------------------------- #
# gateway: plain Python, no CUDA
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS gateway
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY gateway/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY inference_platform/ /app/inference_platform/
COPY gateway/ /app/gateway/
COPY deploy/k8s/base/config.container.json /app/config.json
RUN useradd --uid 10001 --no-create-home app
USER 10001
ENV GATEWAY_PORT=8080 LOG_FORMAT=json
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/readyz').status==200 else 1)"
CMD ["python", "-m", "gateway"]

# --------------------------------------------------------------------------- #
# worker-stub: no GPU, no ML runtime; CI / cluster smoke tests
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS worker-stub
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY worker/requirements/stub.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY inference_platform/ /app/inference_platform/
COPY worker/ /app/worker/
COPY models/__init__.py models/base_model.py models/model_registry.py /app/models/
COPY slide_region.py progress_report.py config.py /app/
COPY deploy/k8s/base/config.container.json /app/config.json
RUN useradd --uid 10001 --no-create-home app
USER 10001
ENV MODEL_NAME=stub WORKER_HTTP_PORT=9100 LOG_FORMAT=json
EXPOSE 9100
CMD ["python", "-m", "worker.main"]

# --------------------------------------------------------------------------- #
# base: CUDA runtime + Python 3.10 + torch + everything every GPU worker needs
# --------------------------------------------------------------------------- #
FROM ${CUDA_IMAGE} AS base
ARG TORCH_INDEX
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mplconfig
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev build-essential git \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libopenslide0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python
RUN pip install --upgrade pip && pip install --index-url ${TORCH_INDEX} torch==2.7.1 torchvision==0.22.1
COPY worker/requirements/common.txt /tmp/common.txt
RUN pip install -r /tmp/common.txt
WORKDIR /app
COPY inference_platform/ /app/inference_platform/
COPY worker/ /app/worker/
COPY models/__init__.py models/base_model.py models/model_registry.py /app/models/
COPY slide_region.py progress_report.py auto_batch.py config.py /app/
COPY src/ /app/src/
COPY deploy/k8s/base/config.container.json /app/config.json
RUN useradd --uid 10001 --no-create-home app && mkdir -p /tmp/mplconfig && chown 10001 /tmp/mplconfig
ENV WORKER_HTTP_PORT=9100 LOG_FORMAT=json DEVICE=auto
EXPOSE 9100
CMD ["python", "-m", "worker.main"]

# --------------------------------------------------------------------------- #
# worker-base: published alias of ``base`` for per-model repos
# (stays root so child Dockerfiles can pip install before USER 10001)
# --------------------------------------------------------------------------- #
FROM base AS worker-base

# --------------------------------------------------------------------------- #
# worker-hovernet
# --------------------------------------------------------------------------- #
FROM base AS worker-hovernet
COPY worker/requirements/hovernet.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY module/hovernet/ /app/module/hovernet/
COPY models/hovernet_model.py /app/models/
USER 10001
ENV MODEL_NAME=hovernet

# --------------------------------------------------------------------------- #
# worker-ec  (EC cancer VarMIL)
# --------------------------------------------------------------------------- #
FROM base AS worker-ec
COPY worker/requirements/ec.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY models/ec_cancer_model.py /app/models/
USER 10001
ENV MODEL_NAME=ec_cancer

# --------------------------------------------------------------------------- #
# worker-pdac  (GigaPath fine-tune; needs HF_TOKEN for the encoder weights)
# --------------------------------------------------------------------------- #
FROM base AS worker-pdac
COPY worker/requirements/pdac.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY module/pdac_inference/ /app/module/pdac_inference/
COPY models/pdac_model.py /app/models/
USER 10001
ENV MODEL_NAME=pdac HF_HOME=/tmp/hf

# --------------------------------------------------------------------------- #
# worker-deepliif  (needs a JDK: DeepLIIF imports bioformats via javabridge)
# --------------------------------------------------------------------------- #
FROM base AS worker-deepliif
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-11-jdk-headless \
    && rm -rf /var/lib/apt/lists/*
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
COPY worker/requirements/deepliif.txt /tmp/model.txt
RUN pip install --no-build-isolation -r /tmp/model.txt
COPY module/DeepLiff/ /app/module/DeepLiff/
COPY models/deepliif_model.py /app/models/
USER 10001
ENV MODEL_NAME=deepliif

# --------------------------------------------------------------------------- #
# tissue-api  (slide-mask / tissue-region helpers: SAM3, no region models)
# --------------------------------------------------------------------------- #
FROM base AS tissue-api
COPY worker/requirements/tissue.txt /tmp/model.txt
RUN pip install -r /tmp/model.txt
COPY app_refactored.py mask_generator.py inference_jobs.py /app/
COPY deploy/k8s/base/config.tissue.json /app/config.json
USER 10001
ENV AI_REPLICA_POOL=general AI_REPLICA_PORT=8000 HF_HOME=/tmp/hf TISSUE_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10m CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["sh", "-c", "exec uvicorn app_refactored:app --host 0.0.0.0 --port ${TISSUE_PORT} --workers 1 --timeout-keep-alive 75"]
