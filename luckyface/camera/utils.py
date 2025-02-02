import cv2
import numpy as np
from PIL import Image
from typing import Tuple

def preprocess_image(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL Image and preprocess."""
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize image while maintaining aspect ratio."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def enhance_image(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better analysis."""
    # Basic enhancement
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)