import cv2
import numpy as np
def preprocess(rgbImage): #preprocess the image
    #binarize
    gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)

    #apply blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    #close and open
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)

    return thresh