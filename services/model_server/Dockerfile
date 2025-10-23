# Use CUDA 12.4 base with Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="EfferenceAI <support@efference.ai>"
LABEL description="Efference Model Server - GPU-Powered Inference Engine"

# Set non-interactive mode and timezone BEFORE installing packages
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

# Install Python 3.10 (default in Ubuntu 22.04) and system dependencies
RUN apt-get update && apt-get install -y \
    # Python (3.10 is default in Ubuntu 22.04)
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    # OpenCV dependencies
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    # Git for installing packages from GitHub
    git \
    # AWS CLI for S3 downloads
    awscli \
    # Download tools
    curl \
    wget \
    # Build tools (needed for some Python packages)
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Reset to default
ENV DEBIAN_FRONTEND=dialog

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /tmp/requirements.txt

# Install base dependencies (everything except efference-rgbd)
RUN grep -v "efference-rgbd" /tmp/requirements.txt > /tmp/requirements-base.txt && \
    pip install --no-cache-dir -r /tmp/requirements-base.txt

# Install efference-rgbd using GitHub token from build secret
RUN --mount=type=secret,id=github_token \
    GITHUB_TOKEN=$(cat /run/secrets/github_token) && \
    pip install --no-cache-dir "git+https://${GITHUB_TOKEN}@github.com/EfferenceAI/efference-rgbd.git@main"

# Verify installations
RUN python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python3 -c "import efference_rgbd; print('efference-rgbd: installed')" && \
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Copy application code
COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create weights directory
RUN mkdir -p /app/weights

# Expose port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_NAME="d435"
ENV MODEL_S3_URI=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]