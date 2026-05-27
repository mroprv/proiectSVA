from config import *
from img_processing.grid_detection import *
from img_processing.preprocess import *
from img_processing.separate_images import *
from utils.visualization import (show,draw_lines)
from game_logic.build_game_matrix import build_matrix
def main():
    img = cv2.imread(IMAGE_PATH)
    #show("orig",img)
    originalImg = img.copy()
    
    img = preprocess(img)

    

    #crop image to game only
    coords=cv2.findNonZero(img)
    x,y,w,h=cv2.boundingRect(coords)
    img = img[y:y+h, x:x+w]

    #resize to be square
    img=cv2.resize(img,(BOARD_SIZE,BOARD_SIZE))

    #detect intersection indexes
    vertical , horizontal =detect_grid_lines(img)
    vertical=process_grids_array(vertical)
    horizontal=process_grids_array(horizontal)

    #separate into 9 images
    imagini=separate_img(img,vertical,horizontal)

    # imgindex=5
    # show("img",imagini[imgindex])
    
    # print(classify_cell(imagini[imgindex]))

    board=build_matrix(imagini)

    print("Game Matrix:")
    for row in board:
        print(row)



    draw_lines(img, vertical, horizontal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()