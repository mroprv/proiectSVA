# =============================================================================
# main.py — punctul de intrare pentru analiza unei imagini statice
# Orchestrează întregul pipeline: citire imagine → preprocesare →
# localizare grilă → detecție linii → separare celule → clasificare simboluri.
# =============================================================================

from config import *
from img_processing.grid_detection import *
from img_processing.grid_localization import extract_and_deskew_grid
from img_processing.preprocess import *
from img_processing.separate_images import *
from symbol_detection.foreign_symbol_detection import detect_foreign
from utils.visualization import *
from game_logic.build_game_matrix import *


def main():
    # --- Citire și afișare imagine originală ---------------------------------
    img = cv2.imread(IMAGE_PATH)
    original = img.copy()
    original = cv2.resize(original, (BOARD_SIZE, BOARD_SIZE))
    show("original", original)

    # --- ETAPA 1: PREPROCESARE -----------------------------------------------
    # Transformă imaginea color într-o imagine binară (alb/negru) curată,
    # pregătită pentru detecția geometrică a liniilor și a simbolurilor.
    img = preprocess(img)

    # --- ETAPA 1b: LOCALIZARE ȘI ALINIERE GRILĂ ------------------------------
    # Detectează liniile grilei în imaginea preprocesată, corectează înclinarea
    # (deskew), apoi decupează imaginea centrată pe celula de mijloc a tablei.
    img = extract_and_deskew_grid(img)

    # --- Decupare la conținut util -------------------------------------------
    # Elimină eventualele zone negre rămase după rotație (artefacte warpAffine)
    # prin găsirea bounding-box-ului pixelilor nenuli.
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]

    # Redimensionare la dimensiunea standard pentru a uniformiza coordonatele
    # folosite în etapele următoare.
    img = cv2.resize(img, (BOARD_SIZE, BOARD_SIZE))

    # --- ETAPA 2: DETECȚIE LINII GRILĂ ---------------------------------------
    # Aplică transformata Hough probabilistică (HoughLinesP) pentru a detecta
    # cele 2 linii verticale și 2 linii orizontale ale grilei tic-tac-toe.
    # process_grids_array elimină duplicatele cauzate de detecții multiple
    # ale aceleiași linii fizice.
    vertical, horizontal = detect_grid_lines(img)
    vertical = process_grids_array(vertical)
    horizontal = process_grids_array(horizontal)

    # --- Separare în 9 celule ------------------------------------------------
    # Folosind coordonatele liniilor, imaginea este decupată în 9 sub-imagini
    # corespunzătoare celor 9 celule ale tablei (3 rânduri × 3 coloane).
    imagini = separate_img(img, vertical, horizontal)

    # --- ETAPA 3: CONSTRUIRE MATRICE DE JOC ----------------------------------
    # Fiecare celulă este clasificată individual ca "X", "O", "?" sau " "
    # (spațiu = celulă goală). Rezultatul este o matrice 3×3 de șiruri.
    board = build_matrix(imagini)

    print("Game Matrix:")
    for row in board:
        print(row)

    # --- Vizualizare rezultate -----------------------------------------------
    draw_separated_images(imagini)   # afișează cele 9 celule într-un singur canvas
    draw_lines(img, vertical, horizontal)  # suprapune liniile detectate pe imagine

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
