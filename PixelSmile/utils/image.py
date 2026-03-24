import math
from typing import Tuple
from PIL import Image
import torchvision.transforms.functional as TF

def calculate_dimensions(target_area: int, ratio: float) -> Tuple[int, int, None]:
    """Calculate dimensions that fit the target area while maintaining aspect ratio"""
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    # Round to nearest 32 for VAE compatibility
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

    # 1. scale
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 2. resize
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # 3. padding
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
    box: list = None,  # [x1, y1, x2, y2] 原始图像坐标
    vertical_bias: float = 0.5,
):
    """
    基于人脸框的安全裁剪：
    - 如果提供 boxes，裁剪区域会尽可能移动以包含该区域。
    - 如果 boxes 范围超过目标尺寸，则以 boxes 中心为裁剪中心。
    """
    w, h = image.size
    target_w, target_h = target_size

    # 1. 计算缩放比例并调整图像
    scale = max(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)

    # 2. 将原始 boxes 坐标映射到新图尺寸
    if box:

        # 缩放坐标
        nx1, ny1, nx2, ny2 = [c * scale for c in box]
        box_cx = (nx1 + nx2) / 2
        box_cy = (ny1 + ny2) / 2
        box_w = nx2 - nx1
        box_h = ny2 - ny1

        # --- 水平方向 (left) ---
        if box_w <= target_w:
            # box 比目标窄，尽量让 box 居中在裁剪区，但不能越界
            left = box_cx - target_w / 2
        else:
            # box 比目标宽，只能以中心对齐
            left = box_cx - target_w / 2
        
        # --- 垂直方向 (top) ---
        if box_h <= target_h:
            # box 比目标矮，此时可以结合 vertical_bias
            # 这里的逻辑是：计算能完全包含 box 的 top 取值范围
            # min_top: 裁剪框下沿刚好压在 box 下沿 (ny2)
            # max_top_for_box: 裁剪框上沿刚好压在 box 上沿 (ny1)
            min_top_to_contain = ny2 - target_h
            max_top_to_contain = ny1
            
            # 在这个安全范围内，根据 vertical_bias 取值
            # 0.0 靠近 max_top_for_box (偏上), 1.0 靠近 min_top_to_contain (偏下)
            top = max_top_to_contain - (max_top_to_contain - min_top_to_contain) * vertical_bias
        else:
            # box 比目标高，直接中心对齐
            top = box_cy - target_h / 2
    else:
       
        # 无 boxes，执行原有的逻辑
        left = (new_w - target_w) // 2
        max_top = new_h - target_h
        top = max_top * vertical_bias

    # 3. 最终安全钳位 (Safety Clamp)
    # 确保 left 在 [0, new_w - target_w]
    left = max(0, min(left, new_w - target_w))
    # 确保 top 在 [0, new_h - target_h]
    top = max(0, min(top, new_h - target_h))

    right = left + target_w
    bottom = top + target_h
    
    # 转换为整数坐标进行裁剪
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
    
