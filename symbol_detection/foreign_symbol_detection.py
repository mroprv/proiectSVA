import numpy as np
import cv2

from symbol_detection.circle_detection import detect_circle
from symbol_detection.cross_detection import detect_x


def detect_foreign(image, crop_fraction: float = 0.10) -> bool:
    has_x = detect_x(image)
    has_o = detect_circle(image)

    if has_x or has_o:
        return False
    h, w = image.shape[:2]
    percent = 0.2
    image = image[
        int(percent * h) : int((1 - percent) * h),
        int(percent * w) : int((1 - percent) * w),
    ]

    # Check if the cell has any meaningful content at all (not empty)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)

    h, w = binary.shape
    dy, dx = max(1, int(h * crop_fraction)), max(1, int(w * crop_fraction))
    cropped = binary[dy : h - dy, dx : w - dx]

    foreground_ratio = np.count_nonzero(cropped) / cropped.size
    return foreground_ratio > 0.02  # cell has content but matched neither X nor O
