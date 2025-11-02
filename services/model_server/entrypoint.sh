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

# Validate MODEL_NAME (determines which weights to load initially)
if [ -z "$MODEL_NAME" ]; then
  log_warn "MODEL_NAME not set, defaulting to 'd435'"
  export MODEL_NAME="d435"
fi

log_info "Configuration:"
log_info "  MODEL_NAME: $MODEL_NAME (initial model to load)"

# Create weights directory
mkdir -p /app/weights

# Define all model weights to download
declare -a WEIGHT_FILES=("d435.pth" "d405.pth")
MAX_RETRIES=3
RETRY_DELAY=5

# Check if all models already exist (for caching)
ALL_EXIST=true
for WEIGHT_FILE in "${WEIGHT_FILES[@]}"; do
  if [ ! -f "/app/weights/$WEIGHT_FILE" ]; then
    ALL_EXIST=false
    break
  fi
done

if [ "$ALL_EXIST" = true ]; then
  log_info "All model weights already exist, skipping download"
  for WEIGHT_FILE in "${WEIGHT_FILES[@]}"; do
    FILE_SIZE=$(stat -c%s "/app/weights/$WEIGHT_FILE" 2>/dev/null || stat -f%z "/app/weights/$WEIGHT_FILE" 2>/dev/null)
    log_info "  $WEIGHT_FILE: $((FILE_SIZE / 1048576))MB"
  done
  
  # Skip download, go straight to server start
  exec uvicorn app.main:app --host 0.0.0.0 --port 8000
fi

# Determine download method based on MODEL_S3_URI format
if [ -z "$MODEL_S3_URI" ]; then
  # Use public Efference weights if no custom URI provided
  log_info "No MODEL_S3_URI provided, using public Efference weights"
  BASE_URL="https://efference-weights.s3.us-west-1.amazonaws.com/public"
  DOWNLOAD_METHOD="wget"
else
  # Check if MODEL_S3_URI is a base URL or specific file
  if [[ "$MODEL_S3_URI" =~ \.(pth|pt)$ ]]; then
    log_error "MODEL_S3_URI should be a base URL/path, not a specific file"
    log_error "Example: https://efference-weights.s3.us-west-1.amazonaws.com/public"
    log_error "        or s3://my-bucket/models"
    exit 1
  fi
  
  BASE_URL="$MODEL_S3_URI"
  
  # Detect if it's an HTTPS URL or S3 URI
  if [[ "$BASE_URL" =~ ^https?:// ]]; then
    DOWNLOAD_METHOD="wget"
    log_info "Detected public HTTPS URL"
  elif [[ "$BASE_URL" =~ ^s3:// ]]; then
    DOWNLOAD_METHOD="aws"
    log_info "Detected private S3 URI (requires AWS credentials)"
    
    # Verify AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
      log_error "AWS credentials not configured for S3 URI!"
      log_error "Either use public HTTPS URL or configure IAM role."
      exit 1
    fi
  else
    log_error "Invalid MODEL_S3_URI format: $BASE_URL"
    log_error "Expected: https://... or s3://..."
    exit 1
  fi
fi

log_info "Base URL: $BASE_URL"
log_info "Download method: $DOWNLOAD_METHOD"
log_info "Downloading all model weights..."

# Download all weight files
for WEIGHT_FILE in "${WEIGHT_FILES[@]}"; do
  MODEL_LOCAL_PATH="/app/weights/$WEIGHT_FILE"
  MODEL_URL="$BASE_URL/$WEIGHT_FILE"
  
  log_info ""
  log_info "Downloading $WEIGHT_FILE..."
  log_info "  Source: $MODEL_URL"
  log_info "  Target: $MODEL_LOCAL_PATH"
  
  # Download with retry logic
  RETRY_COUNT=0
  while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    log_info "  Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES"
    
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
      log_info "  ✓ $WEIGHT_FILE downloaded successfully!"
      
      # Verify file size
      FILE_SIZE=$(stat -c%s "$MODEL_LOCAL_PATH" 2>/dev/null || stat -f%z "$MODEL_LOCAL_PATH" 2>/dev/null)
      FILE_SIZE_MB=$((FILE_SIZE / 1048576))
      log_info "  File size: ${FILE_SIZE_MB}MB"
      
      # Verify file is not corrupted (basic check)
      if [ "$FILE_SIZE" -lt 1000000 ]; then
        log_error "  Downloaded file is suspiciously small (<1MB). May be corrupted."
        exit 1
      fi
      
      break
    else
      RETRY_COUNT=$((RETRY_COUNT + 1))
      if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        log_warn "  Download failed. Retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
      else
        log_error "  Failed to download $WEIGHT_FILE after $MAX_RETRIES attempts!"
        exit 1
      fi
    fi
  done
done

log_info ""
log_info "✓ All model weights downloaded successfully!"

echo "======================================================================="
log_info "Starting Uvicorn server..."
log_info "Listening on 0.0.0.0:8000"
log_info "Model variant: $MODEL_NAME"
echo "======================================================================="

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --log-level info