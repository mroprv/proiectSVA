import cv2

def show(title,img):
    cv2.imshow(title,img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def draw_lines(rgbImage, vertical, horizontal): #draws the detected 2 horizontal,vertical lines onto the image and displays it

    out = rgbImage.copy()
    out=cv2.cvtColor(out,cv2.COLOR_GRAY2BGR)

    for x in vertical:
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 255, 0), 2)

    for y in horizontal:
        cv2.line(out, (0, y), (out.shape[1], y), (255, 0, 0), 2)

    show("lines",out)