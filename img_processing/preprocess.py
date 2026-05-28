# =============================================================================
# preprocess.py — preprocesarea imaginii de intrare
#
# Scopul acestui modul este de a transforma o imagine color (RGB) într-o
# imagine binară (alb = prim-plan, negru = fundal) cât mai curată, eliminând
# variațiile de iluminare, zgomotul și artefactele care ar îngreuna detecția
# liniilor și a simbolurilor în etapele ulterioare.
# =============================================================================

import cv2
import numpy as np

from config import IS_PHOTO_REAL


def preprocess(rgbImage):
    # --- Pas 1: Conversie la tonuri de gri -----------------------------------
    # Reducem imaginea de la 3 canale (BGR) la 1 canal (intensitate luminoasă),
    # deoarece informația de culoare nu este relevantă pentru detecția formelor.
    gray = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)

    # --- Pas 2: CLAHE (Contrast Limited Adaptive Histogram Equalization) -----
    # Egalizare adaptivă a histogramei pe regiuni (tile-uri) de 8×8 pixeli.
    # Spre deosebire de egalizarea globală, CLAHE îmbunătățește contrastul local
    # fără a amplifica excesiv zgomotul în zonele uniforme.
    # clipLimit=2.0 limitează amplificarea maximă a contrastului per tile.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Pas 3: Estimarea fundalului prin blur puternic ----------------------
    # Un blur Gaussian cu kernel mare (55×55) estompează toate detaliile fine
    # (linii, simboluri) și lasă doar variația lentă a iluminării — adică
    # fundalul „iluminat neuniform".
    bg = cv2.GaussianBlur(gray, (55, 55), 0)

    # --- Pas 4: Normalizare prin împărțire (illumination normalization) ------
    # Împărțim pixel cu pixel imaginea la estimarea fundalului și scalăm la
    # [0, 255]. Rezultatul este o imagine în care variațiile de iluminare sunt
    # compensate: o zonă mai întunecată va fi adusă la același nivel de
    # intensitate ca o zonă mai luminoasă.
    norm = cv2.divide(gray, bg, scale=255)

    # --- Pas 5: Blur ușor pentru reducerea zgomotului de înaltă frecvență ----
    # Kernel 5×5 — pregătește imaginea pentru pragul Otsu fără a estompa
    # contururile principale.
    blur = cv2.GaussianBlur(norm, (5, 5), 0)

    # --- Pas 6: Prag binar Otsu (THRESH_BINARY_INV) --------------------------
    # Otsu calculează automat pragul optim de separare prim-plan/fundal pe baza
    # bimodalității histogramei. THRESH_BINARY_INV inversează rezultatul astfel
    # încât liniile și simbolurile (care sunt mai închise) devin ALBE (255) iar
    # fundalul devine NEGRU (0) — convenție necesară pentru cv2.findContours
    # și HoughLinesP.
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Pas 7: Operații morfologice de curățare -----------------------------
    kernel = np.ones((3, 3), np.uint8)

    # MORPH_CLOSE (dilatare + eroziune): unește întreruperile mici din linii și
    # contururi, „lipind" segmentele care ar fi trebuit să fie continue.
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

    # MORPH_OPEN (eroziune + dilatare): elimină zgomotul punctiform și obiectele
    # mici izolate care nu aparțin grilei sau simbolurilor.
    # iterations=2 pentru fotografii reale (mai mult zgomot), 1 pentru imagini
    # digitale curate.
    thresh = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=1 + IS_PHOTO_REAL
    )

    # --- Pas 8: Eliminare pixeli izolați (doar pentru fotografii reale) ------
    # Un pixel izolat (fără niciun vecin din cei 8 vecini direcți) este zgomot
    # și nu poate face parte dintr-un contur util.
    # Filtrul de vecinătate numără câți vecini albi are fiecare pixel alb;
    # cei cu 0 vecini sunt resetați la negru.
    if IS_PHOTO_REAL:
        kernel = np.array(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8
        )
        neighbor_count = cv2.filter2D(thresh, -1, kernel)
        thresh[(thresh == 1) & (neighbor_count == 0)] = 0

    return thresh
