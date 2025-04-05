"""
Image generation and handling utilities for Satoshi AI API
"""
import logging
import tempfile
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Dict, Any, Optional

# Get logger for this module
logger = logging.getLogger(__name__)


def create_text_image(text: str, width: int = 800, height: int = 400) -> Tuple[BytesIO, str]:
    """
    Create an image with text when image generation fails.
    
    Args:
        text: Text to render on image
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (image buffer, temporary file path)
    """
    try:
        # Create a simple image with the text response
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Use default font if TrueType not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()

        # Add text to image with wrapping
        import textwrap
        wrapped_text = textwrap.fill(text, width=80)
        draw.text((10, 10), wrapped_text, fill=(0, 0, 0), font=font)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            img.save(temp, format="PNG")
            temp_path = temp.name

        # Create buffer for streaming
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return buffer, temp_path
    except Exception as e:
        logger.error(f"Error creating text image: {e}")
        # Create a minimal fallback image with error message
        img = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), f"Error creating image: {str(e)}", fill=(255, 0, 0))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
            img.save(temp, format="PNG")
            temp_path = temp.name
            
        # Create buffer for streaming
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        return buffer, temp_path


def download_image(image_url: str) -> Tuple[Optional[BytesIO], Optional[str]]:
    """
    Download an image from a URL.
    
    Args:
        image_url: URL of the image
        
    Returns:
        Tuple of (image buffer, temporary file path) or (None, None) if download fails
    """
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = BytesIO(response.content)
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
                temp.write(response.content)
                temp_path = temp.name
                
            return image_data, temp_path
        else:
            logger.error(f"Failed to download image: HTTP {response.status_code}")
            return None, None
    except Exception as e:
        logger.error(f"Error downloading image: {e}")
        return None, None


def extract_core_description(image_description: str, max_length: int = 200) -> str:
    """
    Extract the core description from a longer image description.
    
    Args:
        image_description: Full image description
        max_length: Maximum length of core description
        
    Returns:
        Core description
    """
    if len(image_description) > max_length:
        return image_description[:max_length] + "..."
    return image_description 