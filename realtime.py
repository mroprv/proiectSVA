# =============================================================================
# realtime.py — detecție tic-tac-toe în timp real prin webcam
#
# Urmează același pipeline ca main.py, aplicat cadru cu cadru pe fluxul
# video al camerei. Afișează side-by-side:
#   - panoul stâng:  cadrul original al camerei
#   - panoul drept:  imaginea binară procesată cu linii și simboluri suprapuse
#
# Gestionarea erorilor per-cadru: dacă grila nu poate fi detectată într-un
# cadru (iluminare proastă, cameră mișcată, ocluzie), se păstrează ultimul
# rezultat valid și se afișează un mesaj de status — bucla nu se întrerupe.
# =============================================================================

import time

import cv2
import numpy as np

from config import BOARD_SIZE, REALTIME_FPS
from game_logic.build_game_matrix import build_matrix
from img_processing.grid_detection import detect_grid_lines, process_grids_array
from img_processing.grid_localization import extract_and_deskew_grid
from img_processing.preprocess import preprocess
from img_processing.separate_images import separate_img

# Rezoluția de captură a camerei (nu afectează BOARD_SIZE intern)
CAPTURE_W = 640
CAPTURE_H = 480

# Culorile simbolurilor în format BGR (Blue-Green-Red, convenția OpenCV)
_SYMBOL_COLOR = {
    "X": (0,   0,   220),  # roșu
    "O": (220, 60,   0),   # albastru
    "?": (0,  165,  255),  # portocaliu
    " ": None,             # celulă goală — nu se desenează nimic
}


# =============================================================================
# Pipeline per cadru
# =============================================================================

def _process_frame(frame: np.ndarray):
    """
    Aplică întregul pipeline de detecție pe un cadru video.
    Ridică excepție la orice eșec, permițând bucla principală să continue
    cu cadrul următor fără a se bloca.

    Pipeline (identic cu main.py):
      preprocess → extract_and_deskew_grid → crop NonZero → resize →
      detect_grid_lines → separate_img → build_matrix
    """
    # Preprocesare: conversie la binar, CLAHE, normalizare, Otsu, morfologie
    img = preprocess(frame)

    # Localizare grilă: deskew + crop centrat pe celula de mijloc
    img = extract_and_deskew_grid(img)

    # Eliminare zone negre reziduale după rotație
    coords = cv2.findNonZero(img)
    if coords is None:
        raise ValueError("imagine goală după localizare")
    x, y, w, h = cv2.boundingRect(coords)
    img = img[y : y + h, x : x + w]
    img = cv2.resize(img, (BOARD_SIZE, BOARD_SIZE))

    # Detecție linii grilă prin HoughLinesP
    vertical, horizontal = detect_grid_lines(img)
    vertical   = process_grids_array(vertical)
    horizontal = process_grids_array(horizontal)

    # Separare în 9 celule și clasificare simboluri
    cells = separate_img(img, vertical, horizontal)
    board = build_matrix(cells)
    return board, img, vertical, horizontal


# =============================================================================
# Funcții de desenare overlay
# =============================================================================

def _annotate(binary: np.ndarray, board, vertical, horizontal) -> np.ndarray:
    """
    Construiește imaginea de afișare pentru panoul drept:
      - Convertește binarul la BGR pentru a putea desena color.
      - Suprapune liniile grilei (verde = verticale, roșu = orizontale).
      - Desenează simbolul detectat în centrul fiecărei celule, cu culoarea
        corespunzătoare tipului.
    """
    out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    for x in vertical:
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 200, 0), 2)
    for y in horizontal:
        cv2.line(out, (0, y), (out.shape[1], y), (200, 0, 0), 2)

    col_edges = [0] + vertical + [BOARD_SIZE]
    row_edges = [0] + horizontal + [BOARD_SIZE]

    for r in range(min(3, len(row_edges) - 1)):
        for c in range(min(3, len(col_edges) - 1)):
            symbol = board[r][c]
            color  = _SYMBOL_COLOR.get(symbol)
            if color is None:
                continue
            # Centrul celulei (r, c) în coordonate pixel
            cx = (col_edges[c] + col_edges[c + 1]) // 2
            cy = (row_edges[r] + row_edges[r + 1]) // 2
            cv2.putText(out, symbol, (cx - 20, cy + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, color, 3)

    return out


def _board_lines(board) -> list[str]:
    """Formatează matricea de joc ca text ASCII pentru afișare în fereastră."""
    sep = "---+---+---"
    rows = []
    for i, row in enumerate(board):
        rows.append(" | ".join(s if s != " " else "." for s in row))
        if i < 2:
            rows.append(sep)
    return rows


def _draw_board_text(canvas: np.ndarray, board, origin: tuple[int, int]) -> None:
    """Desenează reprezentarea text a matricei de joc pe canvas, la poziția dată."""
    x0, y0 = origin
    for i, line in enumerate(_board_lines(board)):
        cv2.putText(canvas, line, (x0, y0 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


# =============================================================================
# Bucla principală
# =============================================================================

def run_realtime(camera_index: int = 0) -> None:
    """
    Deschide camera, rulează pipeline-ul de detecție pe fiecare cadru și
    afișează rezultatele în timp real.

    Parametri de inițializare cameră:
      - CAP_DSHOW: backend DirectShow (Windows) — evită întârzierea de 30-60s
        a backend-ului implicit MSMF (Media Foundation).
      - Rezoluție setată la CAPTURE_W × CAPTURE_H.
      - Primele 5 cadre sunt consumate (drain) deoarece camera trimite cadre
        întunecate/corupte în primele momente de funcționare.

    Controlul FPS:
      cv2.waitKey primește durata în ms calculată din REALTIME_FPS,
      limitând rata de procesare la valoarea configurată.
    """

    # CAP_DSHOW evită întârzierea de inițializare MSMF pe Windows
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    if not cap.isOpened():
        raise RuntimeError(f"Nu se poate deschide camera {camera_index}")

    # Afișăm imediat un ecran de loading pentru feedback vizual instant
    loading = np.zeros((BOARD_SIZE, BOARD_SIZE * 2, 3), np.uint8)
    cv2.putText(loading, "Initializing camera...", (40, BOARD_SIZE // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imshow("Tic-Tac-Toe — real-time", loading)
    cv2.waitKey(1)

    # Consumăm primele cadre întunecate ale camerei
    for _ in range(5):
        cap.read()

    print("Detecție Tic-Tac-Toe în timp real  |  apasă 'q' pentru ieșire")

    last_annotated: np.ndarray | None = None
    last_board = None
    t_prev = time.perf_counter()
    status = "searching..."

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Calculul FPS real al buclei
        t_now  = time.perf_counter()
        fps    = 1.0 / max(t_now - t_prev, 1e-9)
        t_prev = t_now

        # Procesare cadru curent — erorile sunt capturate pentru continuitate
        try:
            board, processed, vertical, horizontal = _process_frame(frame)
            last_annotated = _annotate(processed, board, vertical, horizontal)
            last_board = board
            status = "detected"
        except Exception as exc:
            status = f"no grid ({exc})"

        # --- Panoul stâng: cadrul original redimensionat ---------------------
        left = cv2.resize(frame, (BOARD_SIZE, BOARD_SIZE))

        # --- Panoul drept: imagine procesată cu overlay ----------------------
        if last_annotated is not None:
            right = last_annotated.copy()
        else:
            right = np.zeros((BOARD_SIZE, BOARD_SIZE, 3), np.uint8)
            cv2.putText(right, "waiting...", (30, BOARD_SIZE // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 80), 2)

        if last_board is not None:
            _draw_board_text(right, last_board, (10, BOARD_SIZE - 115))

        # --- Compunere afișaj side-by-side -----------------------------------
        display = np.hstack([left, right])
        cv2.putText(display, f"FPS {fps:.1f}  |  {status}", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Tic-Tac-Toe — real-time", display)

        # Așteptare calibrată la FPS-ul țintă din config; 'q' oprește bucla
        if cv2.waitKey(max(1, int(1000 / REALTIME_FPS))) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime()
