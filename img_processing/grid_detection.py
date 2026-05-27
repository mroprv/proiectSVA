from config import *

import cv2
import numpy as np
import math

def detect_grid_lines(binarizedImage): #returns 2 arrays that contain the coordinates where the playing grid intersects [x1 x2] [y1 y2]

    lines = cv2.HoughLinesP(binarizedImage,rho=1,theta=np.pi / 180,threshold=100,minLineLength=100,maxLineGap=25)

    if lines is None:
        raise Exception("No grid lines detected")

    vertical = []
    horizontal = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # vertical
        if abs(angle) > 70:
            x_avg = (x1 + x2) // 2
            vertical.append(x_avg)

        # horizontal
        elif abs(angle) < 20:
            y_avg = (y1 + y2) // 2
            horizontal.append(y_avg)

    return process_grids_array(vertical), process_grids_array(horizontal) #processing because the hough transform can detect multiple lines for the same grid line, so we need to filter them out

def process_grids_array(arr): #takes an array, sorts it, and then returns a new array where any subsequent elements that differ from the previous one by more than <LINE_THRESHOLD> are inserted.
    arr.sort()
    arr=[int(x) for x in arr]
    newarr=[arr[0]]
    for i in range(1,len(arr)):
        if arr[i] - arr[i-1] > LINE_THRESHOLD:
            newarr.append(arr[i])
            
    #print(newarr,"/n")            
    return newarr