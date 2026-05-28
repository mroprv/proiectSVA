# =============================================================================
# separate_images.py — împărțirea tablei în 9 celule individuale
#
# Folosind coordonatele celor 2 linii verticale și 2 orizontale detectate,
# imaginea tablei este decupată în 9 sub-imagini binare, câte una pentru
# fiecare celulă a grilei tic-tac-toe.
#
# Indexare celule:
#   img0 | img1 | img2    (rândul de sus)
#   -----+------+-----
#   img3 | img4 | img5    (rândul de mijloc)
#   -----+------+-----
#   img6 | img7 | img8    (rândul de jos)
#
# Coordonate de decupare (notație numpy [rând_start:rând_stop, col_start:col_stop]):
#   - vertical[0], vertical[1]   = pozițiile X ale celor 2 linii verticale
#   - horizontal[0], horizontal[1] = pozițiile Y ale celor 2 linii orizontale
# =============================================================================


def separate_img(img, vertical, horizontal):
    """
    Decupează imaginea tablei în 9 sub-imagini, una per celulă.
    Returnează un tuplu de 9 imagini binare în ordinea stânga-dreapta,
    sus-jos (citire normală).
    """

    # Rândul de sus: de la marginea de sus (y=0) până la prima linie orizontală
    img0 = img[0 : horizontal[0], 0 : vertical[0]]            # coloana stângă
    img1 = img[0 : horizontal[0], vertical[0] : vertical[1]]  # coloana mijloc
    img2 = img[0 : horizontal[0], vertical[1] : img.shape[1]] # coloana dreaptă

    # Rândul de mijloc: între cele 2 linii orizontale
    img3 = img[horizontal[0] : horizontal[1], 0 : vertical[0]]
    img4 = img[horizontal[0] : horizontal[1], vertical[0] : vertical[1]]
    img5 = img[horizontal[0] : horizontal[1], vertical[1] : img.shape[1]]

    # Rândul de jos: de la a doua linie orizontală până la marginea de jos
    img6 = img[horizontal[1] : img.shape[0], 0 : vertical[0]]
    img7 = img[horizontal[1] : img.shape[0], vertical[0] : vertical[1]]
    img8 = img[horizontal[1] : img.shape[0], vertical[1] : img.shape[1]]

    return img0, img1, img2, img3, img4, img5, img6, img7, img8
