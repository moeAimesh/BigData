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

img = cv2.imread(
    r"D:\data\image_data\pixabay_dataset_v1\images_01\analog-camera-kodak-lens-2256976.jpg",
    cv2.IMREAD_COLOR
)
q_hist = calc_histogram(img, bins=32)

def parse_hist_text(s: str) -> np.ndarray:
    """CSV-Text -> np.ndarray (DIM,) L1-normalisiert."""
    v = np.fromstring(s, dtype=np.float32, sep=",")
    if v.size != DIM:
        raise ValueError(f"Expected {DIM}, got {v.size}")
    ss = v.sum()
    return v / ss if ss > 0 else v

def chi2_distance(X: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 0.5 * (((X - q) ** 2) / (X + q + eps)).sum(axis=1)

def hist_intersection_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sim = np.minimum(X, q).sum(axis=1)
    return 1.0 - sim

def hellinger_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sqX, sqq = np.sqrt(X, dtype=np.float32), np.sqrt(q, dtype=np.float32)
    return np.linalg.norm(sqX - sqq, axis=1) / np.sqrt(2.0)

def emd1d_per_channel_distance(X: np.ndarray, q: np.ndarray, bins=BINS) -> np.ndarray:
    Xr, qr = X.reshape(-1, 3, bins), q.reshape(3, bins)
    cX, cq = np.cumsum(Xr, axis=2), np.cumsum(qr, axis=1)
    emd = np.abs(cX - cq).sum(axis=2).mean(axis=1) / (bins - 1)
    return emd.astype(np.float32)

def to_similarity(metric: str, d: np.ndarray) -> np.ndarray:
    m = metric.lower()
    if m == "intersect":
        return 1.0 - d
    if m in ("hellinger", "emd"):
        return 1.0 - d
    if m in ("chi2", "chisquare", "chi-square"):
        p5, p95 = np.percentile(d, 5), np.percentile(d, 95)
        denom = max(p95 - p5, 1e-9)
        return 1.0 - np.clip((d - p5) / denom, 0.0, 1.0)
    raise ValueError(f"unknown metric: {metric}")
