"""Constants for credit calculation and other configurations."""

# Credit costs for different operations
CREDIT_COSTS = {
    "video_inference": 1.0,  # Base cost per video
    "per_frame": 0.01,  # Cost per frame processed
    "per_mb": 0.1,  # Cost per MB of data
    "image_rgbd_base": 0.5,  # Base cost for single image RGBD
}


def calculate_credits(video_metadata: dict) -> float:
    """
    Calculate credits needed for video processing.
    
    Args:
        video_metadata: Dictionary with video metadata (frame_count, fps, resolution, etc.)
    
    Returns:
        Credits amount to charge for processing
    """
    # Base credit cost
    base_credits = 1.0
    
    # Get video parameters
    frame_count = video_metadata.get("frame_count", 0)
    fps = video_metadata.get("fps", 30)
    width = video_metadata.get("width", 0)
    height = video_metadata.get("height", 0)
    
    # Calculate duration in seconds
    duration_seconds = frame_count / max(fps, 1)
    
    # Resolution factor (higher resolution costs more)
    resolution = width * height
    resolution_factor = 1.0
    if resolution > 1920 * 1080:  # 4K
        resolution_factor = 2.0
    elif resolution > 1280 * 720:  # 1080p
        resolution_factor = 1.5
    
    # Duration factor (minimum 0.1 credits per second)
    duration_factor = max(duration_seconds * 0.1, 0.1)
    
    # Calculate total credits
    total_credits = base_credits * resolution_factor * duration_factor
    
    return round(total_credits, 2)


    ## Frame count * height*width
    ## $5 per gpu hour used