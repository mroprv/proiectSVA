from symbol_detection.classify import classify_cell


def build_matrix(imgArr):
    board = [[None for _ in range(3)] for _ in range(3)]

    for i, img in enumerate(imgArr):
        row = i // 3
        col = i % 3

        board[row][col] = classify_cell(img)

    return board
