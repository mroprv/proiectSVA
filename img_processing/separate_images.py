def separate_img(
    img, vertical, horizontal
):  # returns an array of 9 binarized images for each slot on the grid
    # TOP ROW
    img0 = img[0 : horizontal[0], 0 : vertical[0]]
    img1 = img[0 : horizontal[0], vertical[0] : vertical[1]]
    img2 = img[0 : horizontal[0], vertical[1] : img.shape[1]]

    # MIDDLE ROW
    img3 = img[horizontal[0] : horizontal[1], 0 : vertical[0]]
    img4 = img[horizontal[0] : horizontal[1], vertical[0] : vertical[1]]
    img5 = img[horizontal[0] : horizontal[1], vertical[1] : img.shape[1]]

    # BOTTOM ROW
    img6 = img[horizontal[1] : img.shape[0], 0 : vertical[0]]
    img7 = img[horizontal[1] : img.shape[0], vertical[0] : vertical[1]]
    img8 = img[horizontal[1] : img.shape[0], vertical[1] : img.shape[1]]

    return img0, img1, img2, img3, img4, img5, img6, img7, img8
