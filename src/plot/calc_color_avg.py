# === Imports ===
import sqlite3
import os
import numpy as np

# === Config ===
DB_PATH   = r"C:\BIG_DATA\data\database.db"
TABLES    = [f"image_features_part_{i}" for i in range(1, 8)]
HIST_COL  = "color_hist"   # ggf. anpassen
OUT_NPZ   = r"C:\BIG_DATA\data\avg_rgb_points_from_hist.npz"

BINS = 32       # pro Kanal
DIM  = 3 * BINS # 96 Werte insgesamt
MODE = "avg"    # 'avg' oder 'dominant'

# Bin-Kanten/Zentren (wie in calc_histogram genutzt)
edges   = np.linspace(0, 256, BINS+1, dtype=np.float32)
centers = (edges[:-1] + edges[1:]) * 0.5

def parse_hist_text(s: str) -> np.ndarray:
    """CSV-Text -> L1-normalisierter Histogrammvektor (96 Werte)."""
    v = np.fromstring(s, sep=',', dtype=np.float32)
    if v.size != DIM:
        return None
    ssum = v.sum()
    if ssum > 0:
        v /= ssum
    return v

def hist_to_rgb(h: np.ndarray, mode: str = "avg") -> np.ndarray:
    """
    h: (96,) = [B(32), G(32), R(32)] L1-normalisiert
    -> RGB-Wert (0..255) als np.ndarray
    """
    B = h[0:BINS]
    G = h[BINS:2*BINS]
    R = h[2*BINS:3*BINS]

    if mode == "avg":
        b = float((B * centers).sum())
        g = float((G * centers).sum())
        r = float((R * centers).sum())
    elif mode == "dominant":
        b = float(centers[int(np.argmax(B))])
        g = float(centers[int(np.argmax(G))])
        r = float(centers[int(np.argmax(R))])
    else:
        raise ValueError("MODE must be 'avg' or 'dominant'")

    return np.array([r, g, b], dtype=np.float32).clip(0, 255)
