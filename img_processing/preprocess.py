import cv2
import numpy as np

from config import IS_PHOTO_REAL
def preprocess(rgbImage): #preprocess the image
    #binarize
    gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    bg = cv2.GaussianBlur(gray, (55,55), 0)

    norm = cv2.divide(gray, bg, scale=255)

    blur = cv2.GaussianBlur(norm, (5,5), 0)

    _, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imwrite("debug.jpg",thresh)

    #close and open
    kernel = np.ones((3, 3), np.uint8)

    thresh = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations=2)
    thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1+IS_PHOTO_REAL)

    

    return thresh