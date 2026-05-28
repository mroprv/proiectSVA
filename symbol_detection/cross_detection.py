# =============================================================================
# cross_detection.py — detecția simbolului X (cruce)
#
# Metoda folosită: detecția a două familii de segmente diagonale.
# Un X este definit de două linii care se intersectează diagonal:
#   - o linie cu pantă pozitivă (~+45°, de la stânga-jos la dreapta-sus)
#   - o linie cu pantă negativă (~-45°, de la stânga-sus la dreapta-jos)
#
# Detecția se face prin Transformata Hough Probabilistică (HoughLinesP),
# care returnează segmente de dreaptă definite prin coordonatele capetelor.
# Fiecare segment este clasificat după unghiul față de axa orizontală.
# =============================================================================

import cv2
import numpy as np


def detect_x(image: np.ndarray) -> bool:
    """
    Detectează dacă celula conține simbolul X.

    Pași:
    1. Normalizare la imagine binară cu prim-plan alb.
    2. Decupare margine (20% pe fiecare latură) pentru a elimina resturile
       liniilor grilei care ar putea fi interpretate greșit ca diagonale.
    3. HoughLinesP cu lungime minimă adaptivă (25% din latura celulei):
       detectează segmentele de dreaptă semnificative din celulă.
    4. Clasificarea segmentelor în două familii de diagonale:
       - „pozitive"  (unghi între +20° și +70°): pantă stânga-jos → dreapta-sus
       - „negative"  (unghi între -20° și -70°): pantă stânga-sus → dreapta-jos
    5. X este confirmat dacă ambele familii sunt prezente simultan.

    Filtrare unghiuri:
       Segmentele prea apropiate de orizontal (< 20°) sau de vertical (> 70°)
       sunt excluse — acestea sunt mai probabil linii de grilă remanente,
       nu diagonale ale unui X.
    """

    # --- Pas 1: Normalizare la binar cu prim-plan alb ------------------------
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    # Dacă fundalul este alb (mai mult de jumătate din pixeli sunt albi),
    # inversăm pentru a asigura convenția prim-plan=alb, fundal=negru.
    if np.count_nonzero(image) > image.size // 2:
        image = cv2.bitwise_not(image)

    # --- Pas 2: Decupare margine pentru excluderea liniilor de grilă ---------
    h, w = image.shape[:2]
    percent = 0.2
    image = image[
        int(percent * h) : int((1 - percent) * h),
        int(percent * w) : int((1 - percent) * w),
    ]

    # --- Pas 3: Detecție segmente prin HoughLinesP ---------------------------
    cell_size = min(image.shape)
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=20,
        minLineLength=int(cell_size * 0.25),   # minim 25% din latura celulei
        maxLineGap=int(cell_size * 0.10),       # gap maxim 10% — completează linii cu întreruperi mici
    )

    if lines is None:
        return False

    # --- Pas 4: Clasificare segmente în familii de diagonale -----------------
    has_pos, has_neg = False, False

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Normalizare la [-90°, +90°]
        if angle > 90:
            angle -= 180
        if angle <= -90:
            angle += 180

        # Acceptăm doar unghiuri în zona diagonală (20°–70°)
        if 20 <= abs(angle) <= 70:
            if angle > 0:
                has_pos = True   # diagonală pozitivă (╱)
            else:
                has_neg = True   # diagonală negativă (╲)

    # --- Pas 5: X confirmat doar dacă ambele diagonale sunt prezente ---------
    return has_pos and has_neg
