# =============================================================================
# classify.py — clasificarea unei celule individuale
#
# Aplică detectorii în ordine de prioritate și returnează primul rezultat
# pozitiv. Ordinea este importantă: O este verificat înaintea lui X deoarece
# un cerc incomplet ar putea conține și segmente diagonale scurte care să
# declanșeze fals detecția X.
# =============================================================================

from symbol_detection.circle_detection import detect_circle
from symbol_detection.cross_detection import detect_x
from symbol_detection.foreign_symbol_detection import detect_foreign
from config import O_CLOSING_KERNEL_SIZE, O_CLOSING_ITERATIONS


def classify_cell(img) -> str:
    """
    Clasifică conținutul unei celule a tablei de joc.

    Ordinea de testare și logica de decizie:
      1. detect_circle  → "O" : circularitate > 0.25 după closing morfologic
      2. detect_x       → "X" : două familii de diagonale detectate prin Hough
      3. detect_foreign → "?" : conținut structurat care nu e nici O nici X
      4. fallback       → " " : celulă goală (niciun contur semnificativ)

    Parametrii pentru closing-ul morfologic al cercului sunt citiți din
    config.py (O_CLOSING_KERNEL_SIZE, O_CLOSING_ITERATIONS) pentru a permite
    ajustarea fără a modifica logica de detecție.
    """

    if detect_circle(img, O_CLOSING_KERNEL_SIZE, O_CLOSING_ITERATIONS):
        return "O"

    if detect_x(img):
        return "X"

    if detect_foreign(img):
        return "?"

    return " "
