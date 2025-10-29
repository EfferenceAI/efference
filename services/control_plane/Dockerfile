FROM python:3.12-slim

LABEL maintainer="EfferenceAI <support@efference.ai>"
LABEL description="Efference Control Plane - Public API Gateway"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For healthcheck
    curl \
    # For PostgreSQL
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY main.py .

# Create data directory for SQLite fallback
RUN mkdir -p /app/data

# Non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Environment variables with defaults
ENV DATABASE_URL=""
ENV MODEL_SERVER_URL="http://model-server:8000/infer"
ENV REQUEST_TIMEOUT="300"
ENV MAX_FILE_SIZE="20000000"
ENV ADMIN_KEY=""

# Health check
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]