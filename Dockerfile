# CarbIQ — production image (Azure Container Apps, or any Docker host).
#
# CPU-only inference. Both model towers are downloaded at BUILD time and baked
# into the image, so the server never downloads weights at request time and a
# restart never re-fetches them.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf_cache \
    YOLO_CONFIG_DIR=/app/.ultralytics \
    FLASK_ENV=production \
    INFERENCE_DEVICE=cpu

WORKDIR /app

# OpenCV runtime libraries (ultralytics pulls the non-headless opencv-python,
# which needs libGL + glib on a slim Debian base).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# CPU build of torch FIRST — otherwise ultralytics pulls the ~2.5 GB CUDA build
# from PyPI. Same versions as the development venv (2.5.1 / 0.20.1).
RUN pip install torch==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Run as an unprivileged user (UID 1000). Everything the app
# writes at runtime (uploads, SQLite file, caches) lives under /app, so that
# user owns /app from here on.
RUN useradd -m -u 1000 user && chown user /app
USER user

# Stage-2 ViT (~343 MB) into the image's HF cache. Its own layer, so code
# changes do not re-download it.
RUN python -c "from transformers import pipeline; \
pipeline('image-classification', model='edwinpalegre/ee8225-group4-vit-trashnet-enhanced')"

COPY --chown=user . .

# Stage-1 waste detector (~6 MB) via the app's own registered source.
RUN python -c "from app.services.detection_service import _fetch_weights, _WEIGHT_SOURCES as W; \
_fetch_weights('models/yolov8n-waste-det.pt', W['yolov8n-waste-det.pt'])"

# Listens on 7860 by default; hosts can inject $PORT.
EXPOSE 7860

# ONE worker: each worker holds its own copy of both models (~0.9 GB peak).
# Threads serve concurrent requests inside that worker.
CMD ["sh", "-c", "exec gunicorn run:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120"]
