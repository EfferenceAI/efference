"""Basic usage example for Efference SDK."""

import os
from efference import EfferenceClient

# Get API key from environment
api_key = os.getenv("EFFERENCE_API_KEY", "sk_test_demo_key")

# Initialize client (point to local development server)
client = EfferenceClient(
    api_key=api_key,
    base_url="http://localhost:8000",  # Local development
    timeout=600.0
)

# Process a video
try:
    result = client.videos.process("sample_video.mp4")
    print("Success!")
    print(f"Status: {result['status']}")
    print(f"Credits deducted: {result.get('credits_deducted', 'N/A')}")
    print(f"Credits remaining: {result.get('credits_remaining', 'N/A')}")
    print(f"Inference result: {result.get('inference_result', {})}")
except Exception as e:
    print(f"Error: {e}")