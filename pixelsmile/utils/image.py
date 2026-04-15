import math
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF

def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int, None]:
    """Calculate dimensions that fit the target area while maintaining aspect ratio"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    # Round to multiples of 32 for VAE compatibility.
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    
    return width, height, None

def resize_with_padding(
    image: Image.Image,
    target_size: tuple,
    padding_mode: str = "reflect",
    fill_value: int | tuple = 0,
):
    """
    Resize image with aspect ratio preserved, then pad to target size.

    Args:
        image (PIL.Image)
        target_size (tuple): (width, height)
        padding_mode (str): "constant" | "reflect" | "edge"
        fill_value: padding value for constant mode
    """
    w, h = image.size
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top

    image = TF.pad(
        image,
        padding=(pad_left, pad_top, pad_right, pad_bottom),
        padding_mode=padding_mode,
        fill=fill_value,
    )

    return image

def resize_with_crop(
    image: Image.Image,
    target_size: tuple,
    box: list = None,  # [x1, y1, x2, y2] coordinates in the original image
    vertical_bias: float = 0.5,
):
    """Crop after resize, optionally biasing around a face box."""
    w, h = image.size
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    if box:
        nx1, ny1, nx2, ny2 = [c * scale for c in box]
        box_cx = (nx1 + nx2) / 2
        box_cy = (ny1 + ny2) / 2
        box_w = nx2 - nx1
        box_h = ny2 - ny1

        if box_w <= target_w:
            left = box_cx - target_w / 2
        else:
            left = box_cx - target_w / 2
        
        if box_h <= target_h:
            min_top_to_contain = ny2 - target_h
            max_top_to_contain = ny1
            
            # Keep the box fully inside crop while applying vertical bias.
            top = max_top_to_contain - (max_top_to_contain - min_top_to_contain) * vertical_bias
        else:
            top = box_cy - target_h / 2
    else:
        left = (new_w - target_w) // 2
        max_top = new_h - target_h
        top = max_top * vertical_bias

    left = max(0, min(left, new_w - target_w))
    top = max(0, min(top, new_h - target_h))

    right = left + target_w
    bottom = top + target_h
    
    image = image.crop((int(left), int(top), int(right), int(bottom)))

    return image

def resize(
    image: Image.Image,
    target_size: tuple,
    resize_mode: str,
    box:list = None
):
    if resize_mode == "padding":
        return resize_with_padding(image, target_size)
    elif resize_mode == "crop":
        return resize_with_crop(image, target_size, box)
    elif resize_mode == "direct":
        return image.resize(target_size, resample=Image.LANCZOS)
    else:
        raise ValueError(f"Resize mode error: {resize_mode}")

def scale_fun(x: float) -> float:
    y = max(0.0, min(1.0, x))
    if x <= 0.6:
        y = 0.5 * x
    elif x <= 0.8:
        y = x - 0.3
    elif x <= 0.9:
        y = 3.0 * x - 1.9
    else:
        y = 2.0 * x - 1.0
    return y

def scale_scores(data):
    if "scores" in data:
        for key, score in data["scores"].items():
            data["scores"][key] = scale_fun(score)
            
    return data
    
