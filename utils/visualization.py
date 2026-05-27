import cv2
import numpy as np


def show(title, img):
    cv2.imshow(title, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_lines(
    rgbImage, vertical, horizontal
):  # draws the detected 2 horizontal,vertical lines onto the image and displays it

    out = rgbImage.copy()
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for x in vertical:
        cv2.line(out, (x, 0), (x, out.shape[0]), (0, 255, 0), 2)

    for y in horizontal:
        cv2.line(out, (0, y), (out.shape[1], y), (255, 0, 0), 2)

    show("lines", out)


import numpy as np
import cv2


def draw_separated_images(images):
    n = len(images)
    gap = 15
    cols = 3
    rows = int(np.ceil(n / cols))

    # compute max height per row and max width per column
    row_heights = [0] * rows
    col_widths = [0] * cols

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        r = i // cols
        c = i % cols
        row_heights[r] = max(row_heights[r], h)
        col_widths[c] = max(col_widths[c], w)

    canvas_h = sum(row_heights) + (rows + 1) * gap
    canvas_w = sum(col_widths) + (cols + 1) * gap

    canvas = np.full((canvas_h, canvas_w), 40, dtype=np.uint8)

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
