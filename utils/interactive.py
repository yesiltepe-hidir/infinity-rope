import torch
from typing import List
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def add_subtitles(video: torch.Tensor, subtitles: List[str]) -> torch.Tensor:
    """
    Add subtitles to a video tensor by dividing frames equally among subtitles.
    
    Args:
        video: Tensor of shape [b, t, h, w, c] with values in [0, 255]
        subtitles: List of strings to display as subtitles
    
    Returns:
        Video tensor with subtitles rendered on frames
    """
    if len(subtitles) == 0:
        return video
    
    b, t, h, w, c = video.shape
    frames_per_subtitle = t // len(subtitles)
    
    # Convert to numpy for easier manipulation
    video_np = video.numpy().astype(np.uint8)
    
    # Process each batch
    for batch_idx in range(b):
        # Process each subtitle
        for subtitle_idx, subtitle in enumerate(subtitles):
            start_frame = subtitle_idx * frames_per_subtitle
            end_frame = (subtitle_idx + 1) * frames_per_subtitle if subtitle_idx < len(subtitles) - 1 else t
            
            # Apply subtitle to frames in this range
            for frame_idx in range(start_frame, end_frame):
                # Convert frame to PIL Image
                frame = video_np[batch_idx, frame_idx]
                img = Image.fromarray(frame)
                
                # Create draw object
                draw = ImageDraw.Draw(img)
                
                # Try to use a default font, fallback to default if not available
                try:
                    # Try to use a larger font size
                    font_size = max(24, h // 20)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        font = None
                
                # Calculate text position (bottom center with padding)
                text = subtitle.strip()
                if text:  # Only draw if subtitle is not empty
                    # Get text bounding box
                    if font:
                        bbox = draw.textbbox((0, 0), text, font=font)
                    else:
                        bbox = draw.textbbox((0, 0), text)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Position text at bottom center with padding
                    x = (w - text_width) // 2
                    y = h - text_height - 20  # 20 pixels padding from bottom
                    
                    # Draw text with outline for better visibility
                    # Draw outline (black)
                    for adj in range(-2, 3):
                        for adj2 in range(-2, 3):
                            if adj != 0 or adj2 != 0:
                                if font:
                                    draw.text((x + adj, y + adj2), text, font=font, fill=(0, 0, 0))
                                else:
                                    draw.text((x + adj, y + adj2), text, fill=(0, 0, 0))
                    
                    # Draw main text (white)
                    if font:
                        draw.text((x, y), text, font=font, fill=(255, 255, 255))
                    else:
                        draw.text((x, y), text, fill=(255, 255, 255))
                
                # Convert back to numpy array
                video_np[batch_idx, frame_idx] = np.array(img)
    
    # Convert back to torch tensor
    return torch.from_numpy(video_np).float()
