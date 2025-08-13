import json
import numpy as np
import sqlite3
from typing import List, Dict, Any
import cv2
#from features.color_vec import calc_histogram   ----> später erneut testen 



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



DB_PATH = r"C:\BIG_DATA\data\database.db"
img = cv2.imread(r"D:\\data\\image_data\\pixabay_dataset_v1\\images_01\\analog-camera-kodak-lens-2256976.jpg", cv2.IMREAD_COLOR)  # BGR
q_hist = calc_histogram(img, bins=32)  # deine Funktion, L1-normalisiert
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]

PARTS = 7
BINS = 32         # wie für datenbank benutzt 
CH = 3
DIM = CH * BINS   # 96




# --- Parser: CSV-Text -> L1-normalisierter Vektor (96,) ---
def parse_hist_text(s: str) -> np.ndarray:
    v = np.fromstring(s, dtype=np.float32, sep=",")
    if v.size != DIM:
        raise ValueError(f"Expected {DIM}, got {v.size}")
    ss = v.sum()
    return v / ss if ss > 0 else v

# --- Distanzen ---
def chi2_distance(X: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 0.5 * (((X - q) ** 2) / (X + q + eps)).sum(axis=1)

def hist_intersection_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sim = np.minimum(X, q).sum(axis=1)
    return 1.0 - sim  # Distanz

def hellinger_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sqX, sqq = np.sqrt(X, dtype=np.float32), np.sqrt(q, dtype=np.float32)
    return np.linalg.norm(sqX - sqq, axis=1) / np.sqrt(2.0)

def emd1d_per_channel_distance(X: np.ndarray, q: np.ndarray, bins=BINS) -> np.ndarray:
    Xr, qr = X.reshape(-1, 3, bins), q.reshape(3, bins)
    cX, cq = np.cumsum(Xr, axis=2), np.cumsum(qr, axis=1)
    emd = np.abs(cX - cq).sum(axis=2).mean(axis=1) / (bins - 1)  # ∈[0,1]
    return emd.astype(np.float32)

# --- Distanz -> Similarity [0,1] ---
def to_similarity(metric: str, d: np.ndarray) -> np.ndarray:
    m = metric.lower()
    if m == "intersect":
        return 1.0 - d            # Intersection ist schon [0,1]
    if m in ("hellinger", "emd"):
        return 1.0 - d            # beide ∈[0,1]
    if m in ("chi2", "chisquare", "chi-square"):
        # robuste Variante via Perzentile (gegen Ausreißer)
        p5, p95 = np.percentile(d, 5), np.percentile(d, 95)
        denom = max(p95 - p5, 1e-9)
        return 1.0 - np.clip((d - p5) / denom, 0.0, 1.0)
        # einfache Alternative:
        # return 1.0 / (1.0 + d)
    raise ValueError(f"unknown metric: {metric}")

# --- Hauptsuche mit Voting über mehrere Metriken ---
def search_color_voting(
    db_path: str,
    q_hist: np.ndarray,
    hist_col: str = "color_hist",
    metrics = ("chi2", "hellinger", "intersect", "emd"),
    weights: Dict[str, float] = None,
    topk: int = 20
) -> List[Dict[str, Any]]:
    if weights is None:
        weights = {"chi2": 1.0, "hellinger": 1.0, "intersect": 0.8, "emd": 1.2}

    # 1) Daten laden
    ids, paths, feats, tabs = [], [], [], []
    con = sqlite3.connect(db_path); cur = con.cursor()
    for t in TABLES:
        cur.execute(f"PRAGMA table_info({t})")
        cols = {r[1] for r in cur.fetchall()}
        if "id" not in cols or hist_col not in cols: 
            continue
        has_path = ("filepath" in cols) or ("path" in cols)
        sel = f"id,{hist_col}" + (",filepath" if "filepath" in cols else "") + (",path" if "path" in cols else "")
        for row in cur.execute(f"SELECT {sel} FROM {t} WHERE {hist_col} IS NOT NULL"):
            try:
                h = parse_hist_text(row[1])
            except Exception:
                continue
            ids.append(int(row[0])); feats.append(h); tabs.append(t)
            paths.append(row[-1] if has_path else None)
    con.close()
    if not feats:
        return []

    X = np.vstack(feats).astype(np.float32)
    # Sicherheit: L1
    X = (X.T / (X.sum(axis=1) + 1e-12)).T
    q = q_hist.astype(np.float32); q = q / (q.sum() + 1e-12)

    # 2) Distanzen je Metrik
    D: Dict[str, np.ndarray] = {}
    for m in metrics:
        ml = m.lower()
        if ml in ("chi2", "chisquare", "chi-square"):
            D[m] = chi2_distance(X, q)
        elif ml in ("hellinger", "bhattacharyya", "bhatta"):
            D[m] = hellinger_distance(X, q)
        elif ml in ("intersect", "intersection"):
            D[m] = hist_intersection_distance(X, q)
        elif ml in ("emd", "wasserstein"):
            D[m] = emd1d_per_channel_distance(X, q)
        else:
            raise ValueError(f"unknown metric {m}")

    # 3) Similarities + Voting
    S = [weights[m]*to_similarity(m, D[m]) for m in metrics]
    wsum = sum(weights[m] for m in metrics)
    fused = np.sum(S, axis=0) / (wsum + 1e-12)  # ∈[0,1]

    # 4) Top-k
    k = min(topk, fused.size)
    idx = np.argpartition(-fused, k-1)[:k]
    idx = idx[np.argsort(-fused[idx])]

    out = []
    for i in idx:
        res = {
            "table": tabs[i], "id": int(ids[i]),
            "path": None if paths[i] is None else str(paths[i]),
            "fused_similarity": float(fused[i]),
            "per_metric": {m: float(to_similarity(m, D[m][i:i+1])[0]) for m in metrics}
        }
        out.append(res)
    return out




res = search_color_voting(
    db_path=DB_PATH,
    q_hist=q_hist,
    hist_col="color_hist",                   # ggf. anpassen
    metrics=("chi2","hellinger","intersect","emd"),
    weights={"chi2":1.0,"hellinger":1.0,"intersect":0.8,"emd":1.2},
    topk=5
)
for r in res[:5]:
    print(r["fused_similarity"], r["per_metric"], r["path"])