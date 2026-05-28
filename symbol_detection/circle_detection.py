# =============================================================================
# circle_detection.py — detecția simbolului O (cerc)
#
# Metoda folosită: analiza circularității conturului dominant.
# Circularitatea este un descriptor de formă adimensional definit ca:
#
#       C = (4 × π × Arie) / Perimetru²
#
# Pentru un cerc perfect: C = 1.0
# Pentru forme tot mai neregulate: C → 0
#
# Problema cercurilor incomplete (trasate manual, cu întreruperi):
#   Un cerc deschis are perimetrul mai mare decât cel al unui cerc închis
#   cu aceeași arie, deci circularitatea scade sub pragul de detecție.
#   Soluție: operație morfologică de CLOSING înainte de extragerea conturului,
#   care unește întreruperile mici fără a deforma forma generală.
# =============================================================================

import cv2
import numpy as np

from config import O_CLOSING_KERNEL_SIZE, O_CLOSING_ITERATIONS


def detect_circle(
    binary,
    closing_kernel_size: int = O_CLOSING_KERNEL_SIZE,
    closing_iterations: int = O_CLOSING_ITERATIONS,
) -> bool:
    """
    Detectează dacă celula conține simbolul O.

    Pași:
    1. Closing morfologic cu kernel eliptic pentru a închide întreruperile
       mici din conturul cercului.
       - Kernel ELLIPSE (nu dreptunghiular): se potrivește mai bine cu forma
         circulară a simbolului decât un kernel pătrat.
       - Closing = Dilatare urmată de Eroziune:
           * Dilatarea extinde zonele albe, unind segmentele apropiate.
           * Eroziunea readuce forma aproape la dimensiunea inițială, dar cu
             golurile mici acum închise.
    2. Extragerea contururilor externe cu CHAIN_APPROX_SIMPLE (comprimă
       segmentele orizontale/verticale/diagonale la doar capetele lor).
    3. Selectarea conturului cu aria maximă (cel mai mare element din celulă).
    4. Calculul circularității și compararea cu pragul 0.25.
       Pragul 0.25 acceptă și cercuri foarte imperfecte (desenate liber).

    Returnează True dacă circularitatea depășește pragul.
    """
    # --- Pas 1: Closing morfologic pentru completarea cercurilor incomplete --
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # --- Pas 2: Extragere contururi externe ----------------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    # --- Pas 3: Conturul dominant (cel cu aria cea mai mare) -----------------
    c = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(c)
    if area < 50:   # filtru: ignorăm contururi prea mici (zgomot rezidual)
        return False

    perimeter = cv2.arcLength(c, True)
    if perimeter == 0:
        return False

    # --- Pas 4: Calcul circularitate și decizie ------------------------------
    circularity = 4 * np.pi * area / (perimeter ** 2)

    return circularity > 0.25
