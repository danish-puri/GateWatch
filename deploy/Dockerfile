# Headless image for GCM GateWatch. CPU-only (no CUDA); uses opencv-python-headless
# so it needs no display. Scaffold -- refine base image and system deps when the
# perception stack is wired up.
FROM python:3.12-slim

# ffmpeg/libGL are what opencv + RTSP decoding need at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev

# config.yaml and .env are mounted at run time, not baked in
EXPOSE 8080
CMD ["uv", "run", "gcm-gatewatch"]
