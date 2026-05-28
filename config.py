# =============================================================================
# config.py — parametri globali ai proiectului
# Toate modulele importă din acest fișier, astfel încât orice ajustare se face
# dintr-un singur loc fără a modifica logica de procesare.
# =============================================================================

# Calea către imaginea de intrare (folosită de main.py)
IMAGE_PATH = "testing_photos/raw.jpg"  # change photo name to select photo to use

# Dimensiunea (în pixeli) la care se redimensionează tabla înainte de analiză.
# O valoare pătratică simplifică calculele ulterioare de index.
BOARD_SIZE = 500

# Distanța minimă (în pixeli) între două linii detectate pentru a fi considerate
# linii distincte ale grilei (filtru anti-duplicat după Hough).
LINE_THRESHOLD = 20

# 0 = imagine digitală curată (desen, screenshot)
# 1 = fotografie reală (zgomot de cameră, variații de iluminare)
# Activează un MORPH_OPEN suplimentar și eliminarea pixelilor izolați în preprocess.
IS_PHOTO_REAL = 0

# --- Detecție simbol O -------------------------------------------------------
# Dimensiunea kernel-ului eliptic folosit la operația morfologică de closing
# aplicată înainte de analiza contururilor, pentru a închide întreruperile mici
# din cercurile incomplete.
O_CLOSING_KERNEL_SIZE = 5

# Numărul de iterații ale operației de closing (mai multe iterații = goluri mai
# mari pot fi închise, dar și detalii fine pot dispărea).
O_CLOSING_ITERATIONS = 2

# --- Detecție simbol străin --------------------------------------------------
# Fracțiunea din marginea celulei care se elimină înainte de analiză, pentru a
# exclude resturile de linii ale grilei care ar putea fi confundate cu conținut.
FOREIGN_MARGIN = 0.40

# Procentul minim de pixeli de prim-plan (albi) din zona centrală a celulei
# pentru ca aceasta să fie considerată ne-goală.
FOREIGN_MIN_DENSITY = 0.02

# Aria minimă (px²) a unui contur detectat pentru a fi considerat un semn real
# și nu zgomot rezidual.
FOREIGN_MIN_CONTOUR_AREA = 10

# --- Localizare grilă -------------------------------------------------------
# Numărul de pixeli adăugați ca bordură în jurul grilei detectate, utilizat în
# varianta de fallback (bounding-box) când celula centrală nu poate fi izolată.
GRID_LOCALIZE_PADDING = 15

# Factorul de decupare bazat pe celula centrală:
# latura crop-ului = GRID_CROP_FACTOR × dimensiunea celulei centrale.
# 3 = crop strâns pe grilă, 4 = grilă + mică bordură de context.
GRID_CROP_FACTOR = 4

# --- Modul real-time --------------------------------------------------------
# Numărul țintă de cadre pe secundă pentru bucla de captură video.
# Controlează durata de așteptare dintre cadre (cv2.waitKey).
REALTIME_FPS = 15
