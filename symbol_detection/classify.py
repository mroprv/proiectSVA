from symbol_detection.circle_detection import detect_circle
from symbol_detection.cross_detection import detect_x
from symbol_detection.foreign_symbol_detection import detect_foreign


def classify_cell(img) -> str:

    if detect_circle(img):return "O"

    if detect_x(img):return "X"

    if detect_foreign(img):return "?"
    
    return " "
