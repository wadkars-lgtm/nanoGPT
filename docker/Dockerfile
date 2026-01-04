# CUDA 12.8 runtime base (Linux) for GPU-enabled containers
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# ---- OS deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Make python/pip point to 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && python -m pip install --upgrade pip setuptools wheel

# ---- Install PyTorch CUDA 12.8 wheels (your requirement) ----
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ---- Copy lockfile and install deps ----
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install -r /tmp/requirements.lock.txt

# ---- Copy project ----
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY bench/ ./bench/
COPY tests/ ./tests/
COPY config/ ./config/

# Install your package (editable) if needed
RUN pip install -e ".[dev,bench]"

CMD ["bash"]
