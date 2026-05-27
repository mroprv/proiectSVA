import cv2
import numpy as np


def detect_x(image: np.ndarray) -> bool:
    # Normalize to single-channel, white foreground
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(image) > image.size // 2:
        image = cv2.bitwise_not(image)

    # Crop margin to remove stray grid lines
    h, w = image.shape[:2]
    percent = 0.2
    image = image[
        int(percent * h) : int((1 - percent) * h),
        int(percent * w) : int((1 - percent) * w),
    ]

    # Detect line segments
    cell_size = min(image.shape)
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(cell_size * 0.25),
        maxLineGap=int(cell_size * 0.10),
    )

    if lines is None:
        return False

    # Classify segments into +45° and -45° diagonal families
    has_pos, has_neg = False, False
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        if angle <= -90:
            angle += 180
        if 20 <= abs(angle) <= 70:
            if angle > 0:
                has_pos = True
            else:
                has_neg = True

    return has_pos and has_neg
