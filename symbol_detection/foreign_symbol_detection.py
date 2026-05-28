# =============================================================================
# foreign_symbol_detection.py — detecția simbolurilor necunoscute
#
# Această funcție este apelată după ce s-a confirmat că celula nu conține
# nici X, nici O. Scopul ei este să distingă între:
#   - celule cu un simbol nerecunoscut (literă, săgeată, pată etc.) → returnează True
#   - celule goale sau cu zgomot minor → returnează False
#
# Metoda: analiză de densitate + verificare contur semnificativ.
# =============================================================================

import cv2
import numpy as np

from config import FOREIGN_MARGIN, FOREIGN_MIN_DENSITY, FOREIGN_MIN_CONTOUR_AREA


def _to_binary_foreground(image: np.ndarray) -> np.ndarray:
    """
    Convertește orice imagine (color sau gri) la binar cu prim-plan alb.
    Dacă mai mult de jumătate din pixeli sunt deja albi (imagine inversată),
    aplică bitwise_not pentru a normaliza la convenția prim-plan=alb.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)
    return binary


def detect_foreign(
    image: np.ndarray,
    margin: float = FOREIGN_MARGIN,
    min_density: float = FOREIGN_MIN_DENSITY,
    min_contour_area: int = FOREIGN_MIN_CONTOUR_AREA,
) -> bool:
    """
    Detectează dacă celula conține un simbol necunoscut (nu X, nu O).

    Pași:
    1. Conversie la binar cu prim-plan alb.
    2. Decupare margine (FOREIGN_MARGIN % pe fiecare latură) pentru a exclude
       resturile liniilor grilei — acestea apar aproape de bordura celulei
       și ar genera fals-pozitive pentru celule goale.
    3. Calcul densitate prim-plan: dacă mai puțin de FOREIGN_MIN_DENSITY
       (implicit 2%) din pixelii zonei centrale sunt albi, celula este
       considerată goală și funcția returnează False.
    4. Verificare contur: cel puțin un contur cu arie ≥ FOREIGN_MIN_CONTOUR_AREA
       confirmă prezența unui element structurat (nu zgomot izolat).
       Un contur cu arie mică (< 10 px²) este probabil un artefact de compresie
       sau un pixel izolat rămas după morfologie.

    Returnează True dacă există conținut structurat care nu e X sau O.
    """
    binary = _to_binary_foreground(image)

    # --- Pas 2: Decupare margine pentru excluderea liniilor de grilă remanente
    h, w = binary.shape
    binary = binary[
        int(margin * h) : int((1 - margin) * h),
        int(margin * w) : int((1 - margin) * w),
    ]

    # --- Pas 3: Filtru de densitate — respinge celulele practic goale --------
    if np.count_nonzero(binary) / binary.size < min_density:
        return False

    # --- Pas 4: Confirmare prin contur semnificativ --------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return any(cv2.contourArea(c) >= min_contour_area for c in contours)
