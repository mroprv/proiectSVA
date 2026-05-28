# =============================================================================
# visualization.py — utilitare de afișare pentru depanare și prezentare
#
# Funcțiile din acest modul sunt folosite exclusiv pentru vizualizarea
# rezultatelor intermediare și finale. Nu modifică datele de procesare.
# =============================================================================

import cv2
import numpy as np


def show(title, img):
    """Afișează o imagine într-o fereastră OpenCV cu titlul dat."""
    cv2.imshow(title, img)


def draw_lines(rgbImage, vertical, horizontal):
    """
    Suprapune liniile detectate ale grilei pe imaginea procesată și o afișează.

    Liniile verticale sunt desenate în verde, cele orizontale în albastru.
    Se lucrează pe o copie a imaginii (convertită la BGR dacă e gri) pentru
    a nu modifica datele originale.
    """
    out = rgbImage.copy()
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    # Linii verticale — de sus în jos, la coordonata X detectată
    for x in vertical:
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 255, 0), 2)

    # Linii orizontale — de la stânga la dreapta, la coordonata Y detectată
    for y in horizontal:
        cv2.line(out, (0, y), (out.shape[1], y), (255, 0, 0), 2)

    show("lines", out)


def draw_separated_images(images):
    """
    Afișează toate cele 9 celule ale tablei pe un singur canvas, aranjate
    în grila 3×3, cu spații (gap) între ele pentru claritate vizuală.

    Algoritmul calculează dinamic înălțimea maximă per rând și lățimea maximă
    per coloană pentru a acomoda celule de dimensiuni diferite (pot apărea
    dacă liniile grilei nu sunt perfect centrate).
    """
    n = len(images)
    gap = 15       # spațiu în pixeli între celule și față de margini
    cols = 3
    rows = int(np.ceil(n / cols))

    # Calculăm dimensiunea maximă per rând și per coloană
    row_heights = [0] * rows
    col_widths  = [0] * cols

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        r = i // cols
        c = i % cols
        row_heights[r] = max(row_heights[r], h)
        col_widths[c]  = max(col_widths[c],  w)

    # Canvas gri închis pe care se plasează celulele
    canvas_h = sum(row_heights) + (rows + 1) * gap
    canvas_w = sum(col_widths)  + (cols + 1) * gap
    canvas = np.full((canvas_h, canvas_w), 40, dtype=np.uint8)

    # Plasare celule pe canvas
    y_offset = gap
    for r in range(rows):
        x_offset = gap
        for c in range(cols):
            idx = r * cols + c
            if idx >= n:
                break
            img = images[idx]
            h, w = img.shape[:2]
            canvas[y_offset : y_offset + h, x_offset : x_offset + w] = img
            x_offset += col_widths[c] + gap
        y_offset += row_heights[r] + gap

    cv2.imshow("all tiles", canvas)
