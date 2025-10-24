#!/bin/bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print banner
echo "======================================================================="
echo "           EFFERENCE MODEL SERVER - STARTUP                            "
echo "======================================================================="

# Validate MODEL_NAME (determines which weights to use)
if [ -z "$MODEL_NAME" ]; then
  log_warn "MODEL_NAME not set, defaulting to 'd435'"
  export MODEL_NAME="d435"
fi

# Determine weight filename based on model name
case "$MODEL_NAME" in
  "rgbd"|"d435"|"rgbd_d435")
    WEIGHT_FILE="d435.pth"
    ;;
  "rgbd_d405"|"d405")
    WEIGHT_FILE="d405.pth"
    ;;
  *)
    log_warn "Unknown MODEL_NAME '$MODEL_NAME', defaulting to d435.pth"
    WEIGHT_FILE="d435.pth"
    ;;
esac

MODEL_LOCAL_PATH="/app/weights/$WEIGHT_FILE"
MAX_RETRIES=3
RETRY_DELAY=5

log_info "Configuration:"
log_info "  MODEL_NAME: $MODEL_NAME"
log_info "  WEIGHT_FILE: $WEIGHT_FILE"
log_info "  LOCAL_PATH: $MODEL_LOCAL_PATH"

# Create weights directory
mkdir -p /app/weights

# Check if model already exists (for caching)
if [ -f "$MODEL_LOCAL_PATH" ]; then
  log_warn "Model weights already exist at $MODEL_LOCAL_PATH"
  FILE_SIZE=$(stat -c%s "$MODEL_LOCAL_PATH" 2>/dev/null || stat -f%z "$MODEL_LOCAL_PATH" 2>/dev/null)
  log_info "Existing file size: $((FILE_SIZE / 1048576))MB"
  log_info "Using existing weights, skipping download"
  
  # Skip download, go straight to server start
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

# Determine download method based on MODEL_S3_URI format
if [ -z "$MODEL_S3_URI" ]; then
  # Use public Efference weights if no custom URI provided
  log_info "No MODEL_S3_URI provided, using public Efference weights"
  MODEL_URL="https://efference-weights.s3.us-west-1.amazonaws.com/public/$WEIGHT_FILE"
  DOWNLOAD_METHOD="wget"
else
  MODEL_URL="$MODEL_S3_URI"
  
  # Detect if it's an HTTPS URL or S3 URI
  if [[ "$MODEL_URL" =~ ^https?:// ]]; then
    DOWNLOAD_METHOD="wget"
    log_info "Detected public HTTPS URL"
  elif [[ "$MODEL_URL" =~ ^s3:// ]]; then
    DOWNLOAD_METHOD="aws"
    log_info "Detected private S3 URI (requires AWS credentials)"
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
      log_error "AWS credentials not configured for S3 URI!"
      log_error "Either use public HTTPS URL or configure IAM role."
      exit 1
    fi
  else
    log_error "Invalid MODEL_S3_URI format: $MODEL_URL"
    log_error "Expected: https://... or s3://..."
    exit 1
  fi
fi

log_info "Download URL: $MODEL_URL"
log_info "Download method: $DOWNLOAD_METHOD"

# Download with retry logic
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  log_info "Download attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
  
  SUCCESS=false
  
  if [ "$DOWNLOAD_METHOD" = "wget" ]; then
    # Public HTTPS download
    if wget -q --show-progress -O "$MODEL_LOCAL_PATH" "$MODEL_URL"; then
      SUCCESS=true
    fi
  elif [ "$DOWNLOAD_METHOD" = "aws" ]; then
    # Private S3 download
    if aws s3 cp "$MODEL_URL" "$MODEL_LOCAL_PATH" --no-progress; then
      SUCCESS=true
    fi
  fi
  
  if [ "$SUCCESS" = true ]; then
    log_info "✓ Model downloaded successfully!"
    
    # Verify file size
    FILE_SIZE=$(stat -c%s "$MODEL_LOCAL_PATH" 2>/dev/null || stat -f%z "$MODEL_LOCAL_PATH" 2>/dev/null)
    FILE_SIZE_MB=$((FILE_SIZE / 1048576))
    log_info "Model file size: ${FILE_SIZE_MB}MB"
    
    # Verify file is not corrupted (basic check)
    if [ "$FILE_SIZE" -lt 1000000 ]; then
      log_error "Downloaded file is suspiciously small (<1MB). May be corrupted."
      exit 1
    fi
    
    break
  else
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      log_warn "Download failed. Retrying in ${RETRY_DELAY}s..."
      sleep $RETRY_DELAY
    else
      log_error "Failed to download model after $MAX_RETRIES attempts!"
      exit 1
    fi
  fi
done

echo "======================================================================="
log_info "Starting Uvicorn server..."
log_info "Listening on 0.0.0.0:8000"
log_info "Model variant: $MODEL_NAME"
echo "======================================================================="

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info