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

print("=== Basic Video Processing ===")
try:
    result = client.videos.process("sample_video.mp4")
    print("Single frame processing successful!")
    print(f"Status: {result['status']}")
    print(f"Credits deducted: {result.get('credits_deducted', 'N/A')}")
    print(f"Credits remaining: {result.get('credits_remaining', 'N/A')}")
    print(f"Inference result: {result.get('inference_result', {})}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Batch Video Processing ===")
try:
    result = client.videos.process_batch(
        "sample_video.mp4", 
        max_frames=50,
        frame_skip=2
    )
    print("Batch processing successful!")
    print(f"Frames processed: {result.get('frames_processed', 'N/A')}")
    print(f"Credits deducted: {result.get('credits_deducted', 'N/A')}")
    print(f"Credits remaining: {result.get('credits_remaining', 'N/A')}")
    billing = result.get('billing_info', {})
    if billing:
        print(f"Billing breakdown: base={billing.get('base_cost')}, frames={billing.get('frame_cost')}, size={billing.get('size_cost')}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Model Management ===")
try:
    # List available models
    models = client.models.list()
    print(f"Available models: {models.get('available_models', [])}")
    print(f"Current active model: {models.get('active_model', 'Unknown')}")
    
    # Switch model
    switch_result = client.models.switch("d405")
    print(f"Model switch result: {switch_result}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== Live Streaming ===")
try:
    # Check stream status
    status = client.streaming.status()
    print(f"Stream status: {status}")
    
    # Start stream (only if hardware available)
    start_result = client.streaming.start("realsense")
    print(f"Stream start result: {start_result}")
    
    # Get a frame with inference
    frame = client.streaming.get_frame(run_inference=True)
    print(f"Frame captured, count: {frame.get('frame_data', {}).get('frame_count', 'N/A')}")
    if frame.get('credits_deducted'):
        print(f"Inference credits deducted: {frame['credits_deducted']}")
    
    # Stop stream
    stop_result = client.streaming.stop()
    print(f"Stream stop result: {stop_result}")
    
except Exception as e:
    print(f"Streaming error (expected if no hardware): {e}")

print("\n=== RGBD Image Processing ===")
try:
    result = client.images.process_rgbd(
        "color.png",
        save_visualization="depth_colored.png",
        save_3panel="comparison.png"
    )
    print("RGBD processing successful!")
    print(f"Depth range: {result['inference_result']['output']['min']:.2f}-{result['inference_result']['output']['max']:.2f}m")
except Exception as e:
    print(f"Error: {e}")