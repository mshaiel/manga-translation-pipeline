"""Image processing and bounding box utility functions.

Provides helper functions for image loading, grayscale-to-RGB conversion,
coordinate transformations (normalized [0, 1] <-> pixel coordinates), and
bounding box cropping required by Magi and manga-ocr.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image


def load_pil_image(image_source: str | Path | Image.Image) -> Image.Image:
    """Load an image as an RGB PIL Image.

    Args:
        image_source: File path (str or Path) or an existing PIL Image.

    Returns:
        A PIL Image in 'RGB' mode.

    Raises:
        FileNotFoundError: If the file path does not exist.
        ValueError: If the input cannot be decoded into a valid PIL Image.
    """
    if isinstance(image_source, Image.Image):
        if image_source.mode != "RGB":
            return image_source.convert("RGB")
        return image_source

    path = Path(image_source)
    if not path.exists():
        raise FileNotFoundError(f"Manga image not found at: {path}")

    with Image.open(path) as img:
        return img.convert("RGB")


def load_image_for_magi(image_source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an image and format it specifically for Magi v2 inference.

    Magi v2 expects images to be read as grayscale ('L') and then converted
    to RGB ('RGB') as a uint8 numpy array of shape (H, W, 3).

    Args:
        image_source: File path (str or Path), PIL Image, or numpy array.

    Returns:
        A numpy array of shape (H, W, 3) with uint8 dtype.
    """
    if isinstance(image_source, np.ndarray):
        if image_source.ndim == 2:
            # Grayscale (H, W) -> stack to (H, W, 3)
            return np.stack([image_source] * 3, axis=-1).astype(np.uint8)
        if image_source.ndim == 3 and image_source.shape[2] == 3:
            return image_source.astype(np.uint8)
        # Fallback to PIL Image conversion
        pil_img = Image.fromarray(image_source)
    elif isinstance(image_source, Image.Image):
        pil_img = image_source
    else:
        path = Path(image_source)
        if not path.exists():
            raise FileNotFoundError(f"Manga image not found at: {path}")
        pil_img = Image.open(path)

    # Convert to grayscale then RGB as required by Magi v2
    converted_img = pil_img.convert("L").convert("RGB")
    arr = np.array(converted_img, dtype=np.uint8)
    return arr


def bbox_normalized_to_pixel(
    bbox: Sequence[float],
    width: int,
    height: int,
    clip: bool = True,
) -> tuple[int, int, int, int]:
    """Convert normalized [0.0, 1.0] bounding box coordinates to integer pixel coordinates.

    Coordinates are represented as [x1, y1, x2, y2].

    Args:
        bbox: Sequence of 4 floats [x1, y1, x2, y2] normalized between 0.0 and 1.0.
        width: Pixel width of the image.
        height: Pixel height of the image.
        clip: Whether to constrain coordinates strictly within [0, width] and [0, height].

    Returns:
        Tuple of 4 integers: (pixel_x1, pixel_y1, pixel_x2, pixel_y2).
    """
    if len(bbox) != 4:
        raise ValueError(f"Bounding box must contain exactly 4 coordinates, got {len(bbox)}")

    x1 = float(bbox[0]) * width
    y1 = float(bbox[1]) * height
    x2 = float(bbox[2]) * width
    y2 = float(bbox[3]) * height

    # Ensure x1 <= x2 and y1 <= y2
    px1, px2 = min(x1, x2), max(x1, x2)
    py1, py2 = min(y1, y2), max(y1, y2)

    ix1 = int(round(px1))
    iy1 = int(round(py1))
    ix2 = int(round(px2))
    iy2 = int(round(py2))

    if clip:
        ix1 = max(0, min(width, ix1))
        iy1 = max(0, min(height, iy1))
        ix2 = max(0, min(width, ix2))
        iy2 = max(0, min(height, iy2))

    # Guarantee at least 1 pixel width and height if within bounds
    if ix2 <= ix1 and width > 0:
        ix2 = min(width, ix1 + 1)
    if iy2 <= iy1 and height > 0:
        iy2 = min(height, iy1 + 1)

    return ix1, iy1, ix2, iy2


def bbox_pixel_to_normalized(
    bbox: Sequence[int | float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert integer pixel coordinates [x1, y1, x2, y2] to normalized [0.0, 1.0] floats.

    Args:
        bbox: Sequence of 4 numbers [x1, y1, x2, y2] in pixels.
        width: Pixel width of the image.
        height: Pixel height of the image.

    Returns:
        Tuple of 4 floats (norm_x1, norm_y1, norm_x2, norm_y2) clamped to [0.0, 1.0].
    """
    if len(bbox) != 4:
        raise ValueError(f"Bounding box must contain exactly 4 coordinates, got {len(bbox)}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got width={width}, height={height}")

    x1, y1, x2, y2 = bbox
    nx1 = max(0.0, min(1.0, float(x1) / width))
    ny1 = max(0.0, min(1.0, float(y1) / height))
    nx2 = max(0.0, min(1.0, float(x2) / width))
    ny2 = max(0.0, min(1.0, float(y2) / height))

    return min(nx1, nx2), min(ny1, ny2), max(nx1, nx2), max(ny1, ny2)


def crop_image(
    image: Image.Image | np.ndarray,
    bbox: Sequence[int | float],
    normalized: bool = True,
) -> Image.Image:
    """Crop a bounding box region from an image, returning a PIL Image.

    Handles both normalized coordinates (Magi output) and absolute pixel coordinates.

    Args:
        image: PIL Image or numpy array (H, W, C).
        bbox: 4 coordinates [x1, y1, x2, y2].
        normalized: Set to True if bbox coordinates are in [0, 1], False if pixels.

    Returns:
        Cropped PIL Image.
    """
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        pil_img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    width, height = pil_img.size

    if normalized:
        px1, py1, px2, py2 = bbox_normalized_to_pixel(bbox, width, height, clip=True)
    else:
        px1, py1, px2, py2 = (
            max(0, min(width, int(round(bbox[0])))),
            max(0, min(height, int(round(bbox[1])))),
            max(0, min(width, int(round(bbox[2])))),
            max(0, min(height, int(round(bbox[3])))),
        )

    # Ensure non-zero crop area
    if px2 <= px1:
        px2 = min(width, px1 + 1)
    if py2 <= py1:
        py2 = min(height, py1 + 1)

    return pil_img.crop((px1, py1, px2, py2))
