# =====================================================================
# SMART DAIRY | AWS Graviton4 (ARM64) & COOL Container Image
# OpenCV AI Competition 2026, powered by AWS
# Build for AWS Graviton (linux/arm64) or hybrid x86_64:
#   docker buildx build --platform linux/arm64 -t smart-dairy-opencv26:graviton .
# =====================================================================
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    AWS_DEFAULT_REGION=us-east-1 \
    PORT=8000 \
    OPENCV_OPENCL_RUNTIME="" \
    OMP_NUM_THREADS=4

# System libraries required for OpenCV 5 / COOL & video streaming
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python3", "web_app.py", "--host", "0.0.0.0", "--port", "8000"]

