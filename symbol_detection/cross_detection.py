import cv2
import numpy as np

def detect_x(image: np.ndarray, crop_fraction: float = 0.10) -> bool:
    """
    Returns True if the binarized tic-tac-toe cell contains an X.
    Crops inward by crop_fraction on each side to ignore stray grid lines.
    """
    # Normalize to single-channel, white foreground
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(image) > image.size // 2:
        image = cv2.bitwise_not(image)

    # Crop margin to remove stray grid lines
    h, w = image.shape
    dy, dx = max(1, int(h * crop_fraction)), max(1, int(w * crop_fraction))
    image = image[dy:h-dy, dx:w-dx]

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
        if angle > 90:  angle -= 180
        if angle <= -90: angle += 180
        if 20 <= abs(angle) <= 70:
            if angle > 0: has_pos = True
            else:         has_neg = True

    return has_pos and has_neg