import sqlite3, os, numpy as np

DB_PATH   = r"C:\BIG_DATA\data\database.db"
TABLES    = [f"image_features_part_{i}" for i in range(1, 8)]
HIST_COL  = "color_hist"
OUT_NPZ   = r"C:\BIG_DATA\data\avg_rgb_points_from_hist_bold_mode_maxed.npz"

BINS = 32
DIM  = 3*BINS

# Farbverstärkung
SAT_BOOST = 1.7
VAL_BOOST = 1.15
GAMMA     = 1.3

# Dominanz-Modus
MODE  = "soft"
ALPHA = 3.0

# Positions-Stretch
STRETCH = 1.2

# Bin-Zentren
edges   = np.linspace(0, 256, BINS+1, dtype=np.float32)
centers = (edges[:-1] + edges[1:]) * 0.5

def parse_hist_text(s: str) -> np.ndarray:
    v = np.fromstring(s, sep=',', dtype=np.float32)
    if v.size != DIM:
        return None
    ssum = v.sum()
    if ssum > 0: v /= ssum
    return v

def channel_from_hist(hc: np.ndarray, mode: str) -> float:
    if mode == "mean":
        return float((hc * centers).sum())
    if mode == "mode":
        return float(centers[int(np.argmax(hc))])
    if mode == "soft":
        w = np.power(hc + 1e-12, ALPHA)
        w /= w.sum()
        return float((w * centers).sum())
    raise ValueError

def hist_to_rgb(h: np.ndarray, mode: str = MODE) -> np.ndarray:
    B = h[0:BINS]; G = h[BINS:2*BINS]; R = h[2*BINS:3*BINS]
    b = channel_from_hist(B, mode)
    g = channel_from_hist(G, mode)
    r = channel_from_hist(R, mode)
    return np.array([r, g, b], dtype=np.float32).clip(0, 255)
