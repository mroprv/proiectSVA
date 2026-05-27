from symbol_detection.circle_detection import is_circle
from symbol_detection.cross_detection import is_cross
from symbol_detection.foreign_symbol_detection import is_foreign_symbol
def classify_cell(img)->str:
    if is_circle(img): return "O"
    if is_cross(img): return "X"
    if is_foreign_symbol(img): return "?"
    return " "