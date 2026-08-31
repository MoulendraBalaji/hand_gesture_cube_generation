# GestureForge — best-effort container image (Linux focus).
#
# NOTE ON WEBCAM PASSTHROUGH (read the README "Docker" section):
#   * Linux: pass the host webcam in with `--device=/dev/video0` (and typically
#     `--net=host` for OpenCV's V4L2 backend).
#   * Windows / macOS: webcam passthrough into Docker is awkward and not well
#     supported, so running the GUI live from a container is not recommended on
#     those hosts.  The image is still useful for running the headless smoke
#     test and the non-GUI pytest suite in CI-style environments.
#
# Build:  docker build -t gestureforge .
# Smoke:  docker run --rm gestureforge python main.py --headless-log-only
# Tests:  docker run --rm gestureforge python -m pytest tests/ -q

FROM python:3.11-slim

# System libs required by OpenCV/MediaPipe/V4L2.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libv4l-0 libx11-6 libxext6 \
    libsm6 libxrender1 libfontconfig1 libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml requirements.txt ./
COPY gestureforge ./gestureforge
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e ".[ml]" \
    && pip install --no-cache-dir ".[report]" || true

# Headless smoke test by default (no display attached).
CMD ["python", "main.py", "--headless-log-only"]
