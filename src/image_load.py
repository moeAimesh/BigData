import cv2


def fast_load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise Exception(f"Bild konnte nicht geladen werden: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
