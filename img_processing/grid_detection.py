# =============================================================================
# grid_detection.py — detecția liniilor grilei tic-tac-toe
#
# Primește imaginea binară preprocesată și returnează coordonatele celor
# 2 linii verticale și 2 linii orizontale care formează grila.
# =============================================================================

from config import *

import cv2
import numpy as np
import math


def detect_grid_lines(binarizedImage):
    """
    Detectează liniile grilei folosind Transformata Hough Probabilistică.

    Transformata Hough caută segmente de dreaptă în imaginea binară votând
    în spațiul parametric (rho, theta). Varianta probabilistică (HoughLinesP)
    returnează direct segmente definite prin coordonatele capetelor (x1,y1,x2,y2),
    nu drepte infinite, ceea ce o face mai eficientă și mai precisă.

    Parametri HoughLinesP:
      rho=1          — rezoluția distanței în pixeli
      theta=π/180    — rezoluția unghiului în radiani (1 grad)
      threshold=100  — numărul minim de voturi pentru a accepta o linie
      minLineLength=100 — lungimea minimă a unui segment detectat
      maxLineGap=25  — distanța maximă între două segmente colineare pentru
                       a fi unite în același segment

    Clasificare linii:
      |unghi| > 70°  → linie verticală  (x-ul mediu este coordonata grilei)
      |unghi| < 20°  → linie orizontală (y-ul mediu este coordonata grilei)
    """

    lines = cv2.HoughLinesP(
        binarizedImage,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=25,
    )

    if lines is None:
        raise Exception("No grid lines detected")

    vertical = []
    horizontal = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculăm unghiul față de axa orizontală folosind arctan2
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        # Linie aproape verticală: coordonata relevantă este X
        if abs(angle) > 70:
            x_avg = (x1 + x2) // 2
            vertical.append(x_avg)

        # Linie aproape orizontală: coordonata relevantă este Y
        elif abs(angle) < 20:
            y_avg = (y1 + y2) // 2
            horizontal.append(y_avg)

    # Hough poate detecta mai multe segmente pentru aceeași linie fizică;
    # process_grids_array le grupează și returnează o singură valoare per linie.
    return process_grids_array(vertical), process_grids_array(horizontal)


def process_grids_array(arr):
    """
    Filtrează un șir de coordonate detectate, eliminând duplicatele apropiate.

    Sortează valorile și parcurge șirul; două valori consecutive sunt
    considerate aceeași linie dacă diferența lor este mai mică decât
    LINE_THRESHOLD (definit în config.py). Se păstrează prima valoare din
    fiecare grup.

    Exemplu: [245, 248, 251, 495, 498] cu LINE_THRESHOLD=20
             → [245, 495]  (două linii distincte)
    """
    arr.sort()
    arr = [int(x) for x in arr]
    newarr = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] > LINE_THRESHOLD:
            newarr.append(arr[i])

    return newarr
