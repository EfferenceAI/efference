FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

LABEL maintainer="EfferenceAI <support@efference.ai>"
LABEL description="Efference Model Server - GPU-Powered Inference Engine"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # OpenCV dependencies
    libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    # Git for installing packages from GitHub
    git \
    # AWS CLI for S3 downloads
    awscli \
    # Curl for healthcheck
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Verify efference-rgbd installed correctly
RUN python -c "import efference_rgbd; print(f'efference-rgbd version: {efference_rgbd.__version__ if hasattr(efference_rgbd, \"__version__\") else \"installed\"}')"

# Copy application code
COPY app/ ./app/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create weights directory
RUN mkdir -p /app/weights

# Expose port
EXPOSE 8000

# Environment variables (can be overridden at runtime)
ENV MODEL_NAME="rgbd"
ENV MODEL_S3_URI=""
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application through entrypoint
ENTRYPOINT ["./entrypoint.sh"]