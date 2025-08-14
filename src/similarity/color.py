import json
import numpy as np
import sqlite3
from typing import List, Dict, Any
import cv2
# from features.color_vec import calc_histogram   # später erneut testen

# === Konstanten ===
DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]

# Histogramm-Parameter
PARTS = 7
BINS = 32
CH = 3
DIM = CH * BINS   # 96

def to_uint8(img):
    """Konvertiert Bild zu uint8 BGR mit 3 Kanälen."""
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:  # Graustufen -> 3 Kanäle
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> BGR
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img
    im = img.astype(np.float32)
    if im.max() <= 1.0 + 1e-6:  # [0,1] → [0,255]
        im *= 255.0
    im = np.clip(im, 0, 255).astype(np.uint8)
    return im

def calc_histogram(img, bins=32):
    """
    Erwartet BGR, gibt Feature-Vektor mit 3*bins zurück.
    L1-normalisiert.
    """
    img_u8 = to_uint8(img)
    hists = []
    for ch in (0, 1, 2):  # B, G, R
        h = cv2.calcHist([img_u8], [ch], None, [bins], [0, 256])
        hists.append(h.ravel().astype(np.float32))
    hist = np.concatenate(hists)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist
