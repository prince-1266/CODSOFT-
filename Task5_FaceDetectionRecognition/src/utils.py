"""
Utility Functions
=================
Image loading and validation helpers.
"""

import os
import cv2
import numpy as np


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_image(path):
    """Load an image from file path. Returns BGR numpy array or None."""
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def resize_image(image, max_size=1024):
    """Resize image if larger than max_size, keeping aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image
    scale = max_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h))


def is_valid_image(filename):
    """Check if filename has a valid image extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS
