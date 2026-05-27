import cv2
import numpy as np

def is_cross(img) ->bool:
    h,w = img.shape[:2]

    img = img[int(h*0.2):int(h*0.8),int(w*0.2):int(w*0.8)]

    kernel = np.ones((3,3), np.uint8)

    afterClose = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel,iterations=2)

    dilatedImage = cv2.dilate(afterClose,kernel,iterations=2)

    pixelDensity = np.sum(dilatedImage > 0) / dilatedImage.size

    if pixelDensity > 0.2 :
        return True
    return False