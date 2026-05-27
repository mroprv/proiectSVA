import cv2
import numpy as np

def detect_circle(binary):

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    if area < 50:
        return False

    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return False

    circularity = 4 * np.pi * area / (perimeter ** 2)

    
    return circularity > 0.25