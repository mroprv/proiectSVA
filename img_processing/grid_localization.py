# =============================================================================
# grid_localization.py — localizarea și alinierea grilei în imagine
#
# Modulul rezolvă două probleme frecvente în imagini reale:
#   1. Grila este înconjurată de alt conținut (masă, mână, fundal aglomerat)
#      → trebuie decupat doar zona grilei.
#   2. Grila este ușor rotită (camera sau hârtia nu sunt perfect aliniate)
#      → trebuie corectată rotația (deskew) înainte de analiză.
#
# Strategia de crop se bazează pe celula centrală a grilei: se detectează
# cele 2 linii interioare verticale și 2 orizontale, se calculează dimensiunea
# celulei de mijloc, iar crop-ul este setat la GRID_CROP_FACTOR × acea dimensiune,
# centrat pe centrul celulei.
# =============================================================================

import cv2
import math
import numpy as np

from config import GRID_LOCALIZE_PADDING, GRID_CROP_FACTOR


def _weighted_median(values: list[float], weights: list[float]) -> float:
    """
    Mediană ponderată: valoarea de la mijlocul distribuției cumulative
    a greutăților (lungimilor segmentelor).

    Față de media aritmetică, mediana este robustă la valori extreme —
    un segment scurt de zgomot nu va deforma estimarea unghiului de înclinare.
    """
    pairs = sorted(zip(values, weights))
    half = sum(weights) / 2
    cumsum = 0.0
    for val, w in pairs:
        cumsum += w
        if cumsum >= half:
            return val
    return pairs[-1][0]


def _compute_skew(lines: np.ndarray) -> float:
    """
    Estimează unghiul de înclinare al imaginii (în grade, pozitiv = rotire orară)
    pe baza segmentelor detectate de HoughLinesP.

    Principiu:
      - Toate unghiurile sunt normalizate la [0°, 180°) pentru a elimina
        ambiguitatea direcției liniei (un segment A→B și B→A reprezintă
        aceeași linie orientată).
      - Linii aproape orizontale (< 45°): deviația față de 0° = unghiul însuși.
      - Linii aproape verticale (45°–135°): deviația față de 90° = 90° − unghi.
      - Ambele formule produc același semn pentru aceeași rotire fizică,
        astfel încât pot fi combinate într-o singură estimare.
      - Estimarea finală este mediana ponderată cu lungimea fiecărui segment
        (segmentele lungi = linii reale de grilă, au prioritate față de
        segmentele scurte = margini de simboluri sau zgomot).
    """
    deviations: list[float] = []
    weights: list[float] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        length = math.hypot(x2 - x1, y2 - y1)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Normalizare la [0°, 180°): o linie și oglinda ei dau același unghi
        if angle <= 0:
            angle += 180

        if angle < 45:           # aproape orizontală
            deviations.append(angle)
            weights.append(length)
        elif 45 < angle < 135:   # aproape verticală
            deviations.append(90.0 - angle)
            weights.append(length)
        # Liniile diagonale (margini de X, O etc.) sunt ignorate

    if not deviations:
        return 0.0
    return _weighted_median(deviations, weights)


def _cluster_positions(positions: list[int], threshold: int = 15) -> list[int]:
    """
    Grupează pozițiile apropiate în clustere și returnează media fiecărui cluster.

    Liniile grilei generează mai multe detecții Hough pentru aceeași linie
    fizică (segmente parțiale, suprapuse). Această funcție le unifica:
    dacă două poziții consecutive sunt mai aproape de `threshold` pixeli,
    aparțin aceluiași cluster.
    """
    if not positions:
        return []
    positions = sorted(positions)
    clusters: list[list[int]] = [[positions[0]]]
    for p in positions[1:]:
        if p - clusters[-1][-1] <= threshold:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [int(np.mean(c)) for c in clusters]


def _inner_grid_lines(
    lines: np.ndarray, img_w: int, img_h: int
) -> tuple[list[int], list[int]]:
    """
    Extrage cele 2 linii verticale interioare (coordonate X) și cele 2 linii
    orizontale interioare (coordonate Y) ale grilei.

    Strategia „cel mai aproape de centru": dintr-un set potențial mai mare
    de clustere (dacă bordurile hârtiei sau alte elemente au fost detectate),
    se aleg cele 2 clustere cu pozițiile cele mai aproape de centrul imaginii
    pe fiecare axă — acestea corespund liniilor interioare ale grilei,
    nu marginilor exterioare.
    """
    v_xs: list[int] = []
    h_ys: list[int] = []

    for x1, y1, x2, y2 in lines[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle <= 0:
            angle += 180
        if 65 < angle < 115:              # segment vertical → reținem X-ul mediu
            v_xs.append((x1 + x2) // 2)
        elif angle < 25 or angle > 155:   # segment orizontal → reținem Y-ul mediu
            h_ys.append((y1 + y2) // 2)

    v_clusters = _cluster_positions(v_xs)
    h_clusters = _cluster_positions(h_ys)

    # Păstrăm cele 2 clustere cele mai centrale pe fiecare axă
    v2 = sorted(sorted(v_clusters, key=lambda x: abs(x - img_w / 2))[:2])
    h2 = sorted(sorted(h_clusters, key=lambda y: abs(y - img_h / 2))[:2])
    return v2, h2


def _hough_lines(binary: np.ndarray, min_line: int) -> np.ndarray | None:
    """
    Aplică HoughLinesP cu parametri adaptivi la dimensiunea imaginii.
    min_line este calculat ca 15% din latura minimă a imaginii, astfel încât
    funcția funcționează indiferent de rezoluția de intrare.
    """
    return cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=min_line,
        maxLineGap=20,
    )


def extract_and_deskew_grid(
    binary: np.ndarray,
    padding: int = GRID_LOCALIZE_PADDING,
    crop_factor: int = GRID_CROP_FACTOR,
) -> np.ndarray:
    """
    Pipeline de localizare și aliniere a grilei:

    1. Detecție linii Hough pe imaginea binară originală.
    2. Calcul unghi de înclinare (skew) prin mediană ponderată.
    3. Rotire corectivă cu warpAffine dacă skew > 0,5°.
       (unghi negativ în getRotationMatrix2D = rotire în sens trigonometric,
        corectând o înclinare în sens orar)
    4. Re-detecție Hough pe imaginea corectată.
    5. Crop primar: localizează cele 2 linii interioare per axă,
       calculează dimensiunea celulei centrale și decupează la
       crop_factor × dimensiunea celulei, centrat pe celula de mijloc.
    6. Crop fallback: dacă liniile interioare nu pot fi izolate,
       se folosește bounding-box-ul tuturor liniilor axiale + padding.

    Returnează imaginea binară decupată și aliniată.
    """
    H, W = binary.shape
    min_line = int(min(H, W) * 0.15)

    lines = _hough_lines(binary, min_line)
    if lines is None:
        return binary

    skew = _compute_skew(lines)

    # Corectare rotație: un skew pozitiv înseamnă rotire orară a imaginii;
    # corectăm cu unghi negativ (rotire trigonometrică) în coordonate imagine.
    if abs(skew) > 0.5:
        M = cv2.getRotationMatrix2D((W / 2, H / 2), -skew, 1.0)
        binary = cv2.warpAffine(binary, M, (W, H), borderValue=0)
        H, W = binary.shape

    # Re-detectăm liniile după corectare pentru a obține coordonate precise
    lines2 = _hough_lines(binary, min_line)
    if lines2 is None:
        return binary

    # --- Crop primar: bazat pe dimensiunea celulei centrale ------------------
    v, h = _inner_grid_lines(lines2, W, H)

    if len(v) == 2 and len(h) == 2:
        # Distanța dintre cele 2 linii interioare = dimensiunea unei celule
        cell_w = v[1] - v[0]
        cell_h = h[1] - h[0]
        # Centrul celulei de mijloc a tablei
        cx = (v[0] + v[1]) // 2
        cy = (h[0] + h[1]) // 2

        # Crop la crop_factor × celulă, centrat pe celula de mijloc
        half_w = (crop_factor * cell_w) // 2
        half_h = (crop_factor * cell_h) // 2

        x0 = max(0, cx - half_w)
        x1_ = min(W, cx + half_w)
        y0 = max(0, cy - half_h)
        y1_ = min(H, cy + half_h)

        return binary[y0:y1_, x0:x1_]

    # --- Crop fallback: bounding-box al tuturor liniilor axiale --------------
    xs, ys = [], []
    for x1, y1, x2, y2 in lines2[:, 0]:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle <= 0:
            angle += 180
        if angle < 25 or angle > 155 or 65 < angle < 115:
            xs.extend([x1, x2])
            ys.extend([y1, y2])

    if not xs:
        return binary

    x0 = max(0, min(xs) - padding)
    x1_ = min(W, max(xs) + padding)
    y0 = max(0, min(ys) - padding)
    y1_ = min(H, max(ys) + padding)

    return binary[y0:y1_, x0:x1_]
