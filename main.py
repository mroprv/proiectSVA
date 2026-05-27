from config import *
from img_processing.grid_detection import *
from img_processing.preprocess import *
from img_processing.separate_images import *
from symbol_detection.foreign_symbol_detection import detect_foreign
from utils.visualization import *
from game_logic.build_game_matrix import *


def main():
    # read image
    img = cv2.imread(IMAGE_PATH)
    original = img.copy()
    original = cv2.resize(original, (BOARD_SIZE, BOARD_SIZE))
    # show("orig",img)
    show("original", original)

    # STAGE 1: PREPROCESSING ================================ used: img_processing

    img = preprocess(img)

    # crop image to game only
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]

    # resize to be square
    img = cv2.resize(img, (BOARD_SIZE, BOARD_SIZE))

    # STAGE 2: GRID DETECTION ================================ used: img_processing.grid_detection

    # detect intersection indexes
    vertical, horizontal = detect_grid_lines(img)
    vertical = process_grids_array(vertical)
    horizontal = process_grids_array(horizontal)

    # separate into 9 images
    imagini = separate_img(img, vertical, horizontal)

    # STAGE 3: BUILD GAME MATRIX ================================ used: game_logic,symbol_detection
    board = build_matrix(imagini)

    print("Game Matrix:")
    for row in board:
        print(row)

    draw_separated_images(imagini)

    draw_lines(img, vertical, horizontal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
