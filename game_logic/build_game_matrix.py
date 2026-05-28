# =============================================================================
# build_game_matrix.py — construirea matricei de joc 3×3
#
# Primește lista celor 9 imagini de celule (în ordinea stânga-dreapta,
# sus-jos) și returnează o matrice 3×3 cu simbolurile detectate.
# =============================================================================

from symbol_detection.classify import classify_cell


def build_matrix(imgArr):
    """
    Construiește matricea de joc clasificând fiecare celulă individual.

    Mapare index → poziție pe tablă:
      0→(0,0)  1→(0,1)  2→(0,2)
      3→(1,0)  4→(1,1)  5→(1,2)
      6→(2,0)  7→(2,1)  8→(2,2)

    Fiecare element al matricei poate fi:
      "X" — simbol X detectat
      "O" — simbol O detectat
      "?" — simbol necunoscut detectat
      " " — celulă goală
    """
    board = [[None for _ in range(3)] for _ in range(3)]

    for i, img in enumerate(imgArr):
        row = i // 3   # rândul: 0, 0, 0, 1, 1, 1, 2, 2, 2
        col = i % 3    # coloana: 0, 1, 2, 0, 1, 2, 0, 1, 2
        board[row][col] = classify_cell(img)

    return board
