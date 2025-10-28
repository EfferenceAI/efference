# Efference SDK Tutorial

Complete step-by-step guide to using the Efference Python SDK for 3D vision tasks.

## Prerequisites

- Python 3.9 or higher
- An Efference API account and API key
- Sample video or image files (optional)

---

## Step 1: Installation and Setup

### Install the SDK

```bash
pip install efference
```

Verify installation:

```bash
python -c "from efference import EfferenceClient; print('Installation successful!')"
```

### Get Your API Key

1. Log in to your Efference account
2. Navigate to API Settings
3. Create a new API key
4. Copy the key (format: `sk_live_xxxx` or `sk_test_xxxx`)

### Initialize the Client

```python
from efference import EfferenceClient

api_key = "sk_live_your_api_key_here"
client = EfferenceClient(api_key=api_key)

print("Client initialized successfully!")
```

---

## Step 2: Simple Video Processing

Processing a single video file extracts one frame and runs inference.

### Basic Usage

```python
result = client.videos.process("path/to/video.mp4")
```

### Accessing Results

```python
print(f"Status: {result['status']}")
print(f"File size: {result['file_size_bytes'] / 1e6:.2f} MB")
print(f"Credits used: {result['credits_deducted']:.2f}")
print(f"Credits remaining: {result['credits_remaining']:.2f}")

print("Inference results:")
print(result['inference_result'])
```

### Full Example

```python
from efference import EfferenceClient

client = EfferenceClient(api_key="sk_live_your_key")

try:
    result = client.videos.process("sample_video.mp4")
    
    if result['status'] == 'success':
        print("Video processed successfully!")
        
        output = result['inference_result']['output']
        print(f"Depth statistics:")
        print(f"  Min: {output['min']:.2f}m")
        print(f"  Max: {output['max']:.2f}m")
        print(f"  Mean: {output['mean']:.2f}m")
        
        print(f"\nCredit information:")
        print(f"  Deducted: {result['credits_deducted']:.2f}")
        print(f"  Remaining: {result['credits_remaining']:.2f}")
    else:
        print(f"Processing failed: {result['status']}")
        
except FileNotFoundError:
    print("Video file not found")
except Exception as e:
    print(f"Error: {e}")
```

---

## Step 3: Batch Processing

Batch processing analyzes multiple frames from a video.

### Understanding Batch Processing

- `process()` - Extracts and analyzes a single frame
- `process_batch()` - Analyzes multiple frames
- `max_frames` - Limit the number of frames to process
- `frame_skip` - Process every Nth frame (reduces cost and time)

### Basic Batch Processing

```python
result = client.videos.process_batch(
    "long_video.mp4",
    max_frames=50,
    frame_skip=1
)
```

### Interpreting Results

```python
print(f"Frames processed: {result['frames_processed']}")
print(f"Total frames in video: {result['video_metadata']['frame_count']}")
print(f"Video duration: {result['video_metadata']['frame_count'] / result['video_metadata']['fps']:.2f}s")
print(f"Credits used: {result['credits_deducted']:.2f}")
```

### Process Every Other Frame

```python
result = client.videos.process_batch(
    "video.mp4",
    max_frames=100,
    frame_skip=2
)

print(f"Processed {result['frames_processed']} frames")
print(f"Saved {result['frames_processed']} * 2 = {result['frames_processed'] * 2} frames worth of cost")
```

### Full Batch Example

```python
from efference import EfferenceClient

client = EfferenceClient(api_key="sk_live_your_key")

try:
    result = client.videos.process_batch(
        "video.mp4",
        max_frames=30,
        frame_skip=1
    )
    
    print("Batch Processing Results")
    print("========================")
    print(f"Frames processed: {result['frames_processed']}")
    print(f"Video info:")
    print(f"  FPS: {result['video_metadata']['fps']}")
    print(f"  Total frames: {result['video_metadata']['frame_count']}")
    print(f"  Duration: {result['video_metadata']['frame_count'] / result['video_metadata']['fps']:.1f}s")
    
    print(f"\nFrame-by-frame results:")
    for idx, frame in enumerate(result['batch_results']):
        output = frame['inference_result']['output']
        print(f"  Frame {frame['frame_index']}: depth {output['min']:.2f}m-{output['max']:.2f}m")
    
    print(f"\nCredits used: {result['credits_deducted']:.2f}")
    
except Exception as e:
    print(f"Error: {e}")
```

---

## Step 4: Image Processing with Depth

Process RGB images with optional depth maps for depth estimation and correction.

### RGB-Only Processing

Estimate depth from RGB alone:

```python
result = client.images.process_rgbd(
    rgb_path="color.png"
)
```

### RGB + Depth Processing

Refine depth using sensor measurements:

```python
result = client.images.process_rgbd(
    rgb_path="color.png",
    depth_path="depth_from_sensor.png",
    depth_scale=1000.0
)
```

### Understanding Depth Scale

Depth scale converts raw sensor values to meters:

| Sensor | Scale | Notes |
|--------|-------|-------|
| RealSense | 1000.0 | Raw values in mm |
| Kinect | 1000.0 | Raw values in mm |
| Custom | Varies | ?? |

### Configurable Parameters

```python
result = client.images.process_rgbd(
    rgb_path="color.png",
    depth_path="depth.png",
    depth_scale=1000.0,      # mm to m conversion
    input_size=518,           # Model input resolution
    max_depth=25.0           # Max depth for visualization
)
```

### Getting Depth Statistics

```python
result = client.images.process_rgbd("color.png", "depth.png")

output = result['inference_result']['output']
print(f"Depth statistics:")
print(f"  Min depth: {output['min']:.2f}m")
print(f"  Max depth: {output['max']:.2f}m")
print(f"  Mean depth: {output['mean']:.2f}m")
print(f"  Valid depth: {output['has_valid_depth']}")
```

### Saving Visualizations

```python
result = client.images.process_rgbd(
    rgb_path="color.png",
    depth_path="depth.png",
    save_visualization="output/depth_colored.png",
    save_3panel="output/comparison.png"
)
```

This creates:
1. `depth_colored.png` - Colorized depth (blue=near, red=far)
2. `comparison.png` - 3-panel view (RGB | Original | Corrected)

### Full Image Processing Example

```python
from efference import EfferenceClient

client = EfferenceClient(api_key="sk_live_your_key")

try:
    result = client.images.process_rgbd(
        rgb_path="input/color.png",
        depth_path="input/depth_raw.png",
        depth_scale=1000.0,
        input_size=518,
        max_depth=25.0,
        save_visualization="output/depth.png",
        save_3panel="output/comparison.png"
    )
    
    print("Image processing complete!")
    
    output = result['inference_result']['output']
    print(f"Depth statistics:")
    print(f"  Range: {output['min']:.2f}m - {output['max']:.2f}m")
    print(f"  Mean: {output['mean']:.2f}m")
    
    print(f"\nVisualizations saved:")
    print(f"  Single: output/depth.png")
    print(f"  Comparison: output/comparison.png")
    
    print(f"\nCredits used: {result['credits_deducted']:.2f}")
    
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Error: {e}")
```

---

## Step 5: Visualization

Display depth results using matplotlib.


### Single Depth Visualization

```python
result = client.images.process_rgbd("color.png")

fig = client.images.visualize_depth(
    result,
    mode="single",
    show=True
)
```

### 3-Panel Comparison

Shows RGB | Original Depth | Corrected Depth side by side:

```python
fig = client.images.visualize_depth(
    result,
    mode="3panel",
    show=True
)
```

### Save Visualizations

```python
import matplotlib.pyplot as plt

result = client.images.process_rgbd("color.png", "depth.png")

fig = client.images.visualize_depth(result, mode="3panel", show=False)
fig.savefig("visualization.png", dpi=150, bbox_inches='tight')
plt.close(fig)
```

### Complete Visualization Example

```python
from efference import EfferenceClient
import matplotlib.pyplot as plt

client = EfferenceClient(api_key="sk_live_your_key")

result = client.images.process_rgbd(
    "color.png",
    "depth.png",
    depth_scale=1000.0
)

# Save single visualization
fig1 = client.images.visualize_depth(result, mode="single", show=False)
fig1.savefig("depth_single.png", dpi=150)

# Save 3-panel visualization
fig2 = client.images.visualize_depth(result, mode="3panel", show=False)
fig2.savefig("depth_comparison.png", dpi=150)

print("Visualizations saved!")
```

---

## Step 6: Error Handling

Properly handle errors for robust applications.

### Authentication Errors

```python
from efference import EfferenceClient
import httpx

client = EfferenceClient(api_key="invalid_key")

try:
    result = client.videos.process("video.mp4")
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Authentication failed. Check your API key.")
    elif e.response.status_code == 402:
        print("Insufficient credits. Purchase more credits.")
    else:
        print(f"HTTP error: {e.response.status_code}")
```

### File Errors

```python
try:
    result = client.videos.process("nonexistent.mp4")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Invalid file path: {e}")
```

### Connection Errors

```python
try:
    result = client.videos.process("video.mp4")
except httpx.TimeoutException:
    print("Request timed out.")
except httpx.RequestError as e:
    print(f"Connection error: {e}")
```

### Comprehensive Error Handler

```python
from efference import EfferenceClient
import httpx

def safe_process_video(client, video_path):
    try:
        print(f"Processing: {video_path}")
        result = client.videos.process(video_path)
        
        if result['status'] == 'success':
            print(f"Success! Credits used: {result['credits_deducted']:.2f}")
            return result
        else:
            print(f"Processing failed: {result.get('detail', 'Unknown error')}")
            return None
            
    except FileNotFoundError as e:
        print(f"File error: {e}")
        return None
        
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if status == 401:
            print("Authentication failed.")
        elif status == 402:
            print("Insufficient credits.")
        elif status == 413:
            print("File too large (max 500MB).")
        elif status == 429:
            print("Rate limited. Wait before retrying.")
        elif status >= 500:
            print(f"Server error: {status}")
        else:
            print(f"HTTP error: {status}")
        return None
        
    except httpx.TimeoutException:
        print("Request timed out.")
        return None
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

client = EfferenceClient(api_key="sk_live_your_key")
result = safe_process_video(client, "video.mp4")
```

---

## Step 7: Production Patterns

Patterns for production-ready applications.

### Retry Logic

```python
import time

def process_with_retry(client, video_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.videos.process(video_path)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
            time.sleep(wait_time)
```

### Batch Processing Loop

```python
def process_multiple_videos(client, video_files):
    results = []
    
    for video_file in video_files:
        try:
            result = client.videos.process(video_file)
            results.append({
                "file": video_file,
                "status": "success",
                "data": result
            })
        except Exception as e:
            results.append({
                "file": video_file,
                "status": "failed",
                "error": str(e)
            })
    
    return results
```

### Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting video processing")
result = client.videos.process("video.mp4")
logger.info(f"Processing complete. Credits used: {result['credits_deducted']}")
```

### Configuration Management

Create `.env` file:

```bash
EFFERENCE_API_KEY=sk_live_your_key
EFFERENCE_API_URL=https://api.efference.ai
```

Use in code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("EFFERENCE_API_KEY")
api_url = os.getenv("EFFERENCE_API_URL", "https://api.efference.ai")

client = EfferenceClient(api_key=api_key)
```

### Unit Testing

```python
import unittest

class TestEfferenceClient(unittest.TestCase):
    def setUp(self):
        self.client = EfferenceClient(api_key="sk_test_your_key")
    
    def test_video_processing(self):
        result = self.client.videos.process("test_video.mp4")
        self.assertEqual(result['status'], 'success')
        self.assertIn('credits_deducted', result)
    
    def test_image_processing(self):
        result = self.client.images.process_rgbd("test_color.png")
        self.assertEqual(result['status'], 'success')
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'efference'` | Run: `pip install efference` |
| `Authentication failed (401)` | Check API key is correct and active |
| `Insufficient credits (402)` | Purchase credits from account dashboard |
| `File too large (413)` | Split video into smaller files |
| `Request timed out` | Increase timeout: `client = EfferenceClient(api_key=key, timeout=600)` |

---


## Support

For additional help:

- **GitHub Issues**: https://github.com/EfferenceAI/efference/issues
- **Documentation**: https://docs.efference.ai (placeholder for now. We can just create a route on the api.efference.ai to contain the full documentation.)
- **Email**: support@efference.ai
