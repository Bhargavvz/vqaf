# ============================================================
# Medical VQA System - Docker Configuration
# Multi-stage build with CUDA support for GPU inference
# ============================================================

# Stage 1: Base with CUDA
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ============================================================
# Stage 2: Dependencies
# ============================================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install flash-attn separately (needs CUDA)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || \
    echo "Flash attention installation skipped (may not be compatible)"

# ============================================================
# Stage 3: Application
# ============================================================
FROM dependencies AS app

WORKDIR /app

# Copy application code
COPY . /app/medical_vqa/

# Create necessary directories
RUN mkdir -p /app/data /app/outputs /app/cache

# Set environment variables
ENV VQA_CONFIG=/app/medical_vqa/config.yaml
ENV VQA_ADAPTER_PATH=/app/outputs/final_model
ENV TRANSFORMERS_CACHE=/app/cache
ENV HF_HOME=/app/cache

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command: Start the API server
CMD ["python", "-m", "uvicorn", "medical_vqa.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
