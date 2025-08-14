import json
import numpy as np
import sqlite3
from typing import List, Dict, Any
import cv2
# from features.color_vec import calc_histogram   # TODO: später erneut testen

# === Konstanten ===
DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]

# Histogramm-Setup
BINS = 32    # wie in der DB genutzt
CH = 3
DIM = CH * BINS   # 96

def to_uint8(img):
    """Bringt ein Bild sicher auf uint8 und 3 Kanäle (BGR)."""
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
    img_u8 = to_uint8(img)

    hists = []
    for ch in (0, 1, 2):  # B, G, R
        h = cv2.calcHist([img_u8], [ch], None, [bins], [0, 256])
        h = h.ravel().astype(np.float32)
        hists.append(h)

    hist = np.concatenate(hists)   # Länge: 3*bins
    s = hist.sum()
    if s > 0:
        hist /= s                  # L1-Normierung
    return hist

# Beispielbild (BGR) laden
img = cv2.imread(
    r"D:\data\image_data\pixabay_dataset_v1\images_01\analog-camera-kodak-lens-2256976.jpg",
    cv2.IMREAD_COLOR
)
q_hist = calc_histogram(img, bins=BINS)  # L1-normalisiert

def parse_hist_text(s: str) -> np.ndarray:
    """CSV-Text -> Vektor (DIM) mit L1-Normierung."""
    v = np.fromstring(s, dtype=np.float32, sep=",")
    if v.size != DIM:
        raise ValueError(f"Expected {DIM}, got {v.size}")
    ss = v.sum()
    return v / ss if ss > 0 else v
