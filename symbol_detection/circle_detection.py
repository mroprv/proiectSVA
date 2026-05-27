import cv2
import numpy as np

def is_circle(binarizedImage) -> bool: #checks if the binarized image contains a circle
    contours, _ = cv2.findContours(binarizedImage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)

    _ , radius = cv2.minEnclosingCircle(c)

    circle_area = np.pi * radius * radius

    ratio = area / circle_area
    if 0.35 < ratio < 0.9:
        return True
    return False