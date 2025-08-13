import numpy as np
import cv2

def to_uint8(img):
    if img is None:
        raise ValueError("img is None")
    # Grau -> 3 Kanäle
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # RGBA -> RGB/BGR (Alpha weg)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    # Auf uint8 bringen
    if img.dtype == np.uint8:
        return img
    im = img.astype(np.float32)
    # Falls [0,1] float: hochskalieren
    if im.max() <= 1.0 + 1e-6:
        im *= 255.0
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im

def calc_histogram(img, bins=32):
    """
    Erwartet BGR, gibt Feature-Vektor mit 3*bins zurück (B, G, R).
    Normierung: L1 (Summe = 1).
    """
    img_u8 = to_uint8(img)  # schnell & sicher

    hists = []
    # Für Speed: kein Farbwechsel nötig; bleib bei BGR
    for ch in (0, 1, 2):  # B, G, R
        h = cv2.calcHist([img_u8], [ch], None, [bins], [0, 256])  # 0..255 -> [0,256]
        h = h.ravel().astype(np.float32)
        hists.append(h)

    hist = np.concatenate(hists)                # Länge: 3*bins
    s = hist.sum()
    if s > 0:
        hist /= s                               # L1-Normierung
    return hist
