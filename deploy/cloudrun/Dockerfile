# ForgeLM demonstration -- CPU inference container for Cloud Run.
#
# The base model is baked into the image at the pinned revision rather than
# downloaded on boot. A cold start on Cloud Run already pays for pulling the
# image and materialising ~2 GB of float32 weights; adding a 1 GB download from
# the Hub on top of that is the difference between a slow first request and a
# startup probe that times out. It also means the container never needs the
# network at runtime, which is why HF_HUB_OFFLINE is set below: if the bake
# ever silently failed, the app fails loudly at boot instead of quietly
# reaching for the internet in production.

FROM python:3.12-slim

# libgomp1 is OpenMP, which the CPU torch wheels link against for their thread
# pool. It is not in python:*-slim and the failure without it is an obscure
# import error, not a missing-package message.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# Dependencies first: this layer is the expensive one and only needs rebuilding
# when requirements.txt changes, not when app.py does.
COPY requirements.txt ./
# gradio is deliberately absent from requirements.txt -- on Hugging Face the
# runtime installs it from the README's sdk_version, and pinning it there would
# conflict. In a container there is no such runtime, so it is pinned here, to
# the version this app was verified against.
RUN pip install -r requirements.txt "gradio==6.25.0"

# The model id and revision live in the package, so the bake cannot drift from
# what the application will ask for at runtime.
COPY src ./src
RUN python -c "\
import sys; sys.path.insert(0, 'src'); \
from forgelm.modeling import BASE_MODEL_ID, BASE_MODEL_REVISION; \
from huggingface_hub import snapshot_download; \
p = snapshot_download(BASE_MODEL_ID, revision=BASE_MODEL_REVISION, \
    ignore_patterns=['*.pth', '*.msgpack', '*.h5', '*.onnx', '*.gguf']); \
print('baked', BASE_MODEL_ID, BASE_MODEL_REVISION[:12], '->', p)"

COPY artifacts ./artifacts
COPY app.py ./

# Fail loudly rather than reaching for the network if the bake above regressed.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Cloud Run injects PORT and app.py prefers it; 8080 is the platform default
# and keeps `docker run -p 8080:8080` working locally without extra flags.
ENV PORT=8080
EXPOSE 8080

# Nothing here needs root, and the baked cache only needs to be readable.
RUN useradd --create-home --uid 1000 forgelm \
    && chown -R forgelm:forgelm /app /opt/hf
USER forgelm

CMD ["python", "app.py"]
