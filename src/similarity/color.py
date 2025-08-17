# color_search_variantA_args.py
# -*- coding: utf-8 -*-

import argparse
import sqlite3
from typing import List, Dict, Any, Sequence, Optional

import cv2
import numpy as np

# ======================== FIXED CONFIG ========================
HIST_COL      = "color_hist"
TABLE_PREFIX  = "image_features_part_"
BINS          = 32
DIM           = 3 * BINS
DO_SWAP_RB    = True  # Query [B|G|R] -> [R|G|B] zum DB-Match

DEFAULT_METRICS = ("chi2", "hellinger", "intersect", "emd")

PRESET_WEIGHTS = {
    "balanced":   {"chi2": 1.8, "hellinger": 1.0, "intersect": 0.2, "emd": 1.3},
    "two_color":  {"chi2": 2.2, "hellinger": 0.8, "intersect": 0.0, "emd": 1.5},
    "flat_logo":  {"chi2": 1.0, "hellinger": 0.8, "intersect": 1.2, "emd": 0.5},
}
USE_PRESET: Optional[str] = None  # None => Auto-Weights

# ======================== AUTO-WEIGHTS ========================
def auto_weights(q: np.ndarray, bins: int = 32) -> Dict[str, float]:
    ch = q.reshape(3, bins).sum(axis=1)  # (B,G,R)
    p  = q[q > 0]
    H  = -(p * np.log(p)).sum() / np.log(q.size) if p.size else 0.0

    w = {"chi2": 1.8, "hellinger": 1.0, "intersect": 0.2, "emd": 1.3}
    if H < 0.55:  # einfarbig/geringe Entropie
        w.update({"intersect": 1.0, "emd": 0.7})
    if (ch > ch.sum() * 0.40).sum() == 2:  # zwei dominante Kanäle
        w.update({"chi2": 2.2, "emd": 1.6, "intersect": 0.0})
    return w

# ======================== UTILS ========================
def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img
    im = img.astype(np.float32)
    if im.max() <= 1.0 + 1e-6:
        im *= 255.0
    return np.clip(im, 0, 255).astype(np.uint8)

def calc_histogram(img_bgr: np.ndarray, bins: int = BINS) -> np.ndarray:
    img_u8 = to_uint8(img_bgr)
    hists = []
    for ch in (0, 1, 2):  # B,G,R
        h = cv2.calcHist([img_u8], [ch], None, [bins], [0, 256]).ravel().astype(np.float32)
        hists.append(h)
    hist = np.concatenate(hists)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

def swap_rb_hist(v: np.ndarray, bins: int = BINS) -> np.ndarray:
    b, g, r = v[:bins], v[bins:2 * bins], v[2 * bins:]
    out = np.concatenate([r, g, b])
    out /= (out.sum() + 1e-12)
    return out

def parse_hist_text(s: str) -> np.ndarray:
    v = np.fromstring(s, dtype=np.float32, sep=",")
    if v.size != DIM:
        raise ValueError(f"Expected {DIM}, got {v.size}")
    ss = float(v.sum())
    return v / ss if ss > 0 else v

def list_feature_tables(con: sqlite3.Connection, prefix: str = TABLE_PREFIX) -> List[str]:
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?", (f"{prefix}%",))
    return [r[0] for r in cur.fetchall()]

# ======================== DISTANCES & FUSION ========================
def chi2_distance(X: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 0.5 * (((X - q) ** 2) / (X + q + eps)).sum(axis=1)

def hist_intersection_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sim = np.minimum(X, q).sum(axis=1)
    return 1.0 - sim

def hellinger_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sqX = np.sqrt(X)
    sqq = np.sqrt(q)
    return np.linalg.norm(sqX - sqq, axis=1) / np.sqrt(2.0)

def emd1d_per_channel_distance(X: np.ndarray, q: np.ndarray, bins: int = BINS) -> np.ndarray:
    Xr = X.reshape(-1, 3, bins)
    qr = q.reshape(3, bins)
    cX = np.cumsum(Xr, axis=2)
    cq = np.cumsum(qr, axis=1)
    emd = np.abs(cX - cq).sum(axis=2).mean(axis=1) / (bins - 1)
    return emd.astype(np.float32)

def to_similarity(metric: str, d: np.ndarray) -> np.ndarray:
    m = metric.lower()
    if m in ("intersect", "intersection", "hellinger", "emd", "wasserstein"):
        return 1.0 - d
    if m in ("chi2", "chisquare", "chi-square"):
        p5, p95 = np.percentile(d, 5), np.percentile(d, 95)
        denom = max(p95 - p5, 1e-9)
        return 1.0 - np.clip((d - p5) / denom, 0.0, 1.0)
    raise ValueError(f"unknown metric: {metric}")

# ======================== UNIVERSAL CONSTRAINTS ========================
def apply_channel_constraints_universal(
    fused: np.ndarray,
    X: np.ndarray,
    q: np.ndarray,
    bins: int,
    tol_frac: float = 0.010,
    alpha_missing: float = 0.90,
    alpha_bins: float = 0.50,
    alpha_prop: float = 0.35
) -> np.ndarray:
    X3 = X.reshape(-1, 3, bins)
    q3 = q.reshape(3, bins)

    q_ch = q3.sum(axis=1)
    X_ch = X3.sum(axis=2)

    tol_ch = tol_frac
    missing_mask = (q_ch < tol_ch).astype(np.float32)
    extra_mass = np.maximum(X_ch - (q_ch + tol_ch), 0.0) * missing_mask
    penalty_missing = alpha_missing * extra_mass.sum(axis=1)

    tol_bin = tol_ch / max(bins, 1)
    forbid_bins = (q3 < tol_bin).astype(np.float32)
    penalty_bins = alpha_bins * (X3 * forbid_bins).sum(axis=(1, 2))

    active_idx = np.where(q_ch >= tol_ch)[0]
    if active_idx.size >= 2:
        cq = q_ch[active_idx] / (q_ch[active_idx].sum() + 1e-12)
        cx = X_ch[:, active_idx]
        cx = (cx.T / (cx.sum(axis=1) + 1e-12)).T
        prop_l1 = np.abs(cx - cq).sum(axis=1)
        penalty_prop = alpha_prop * 0.5 * prop_l1
    else:
        penalty_prop = np.zeros((X.shape[0],), dtype=np.float32)

    penalty = penalty_missing + penalty_bins + penalty_prop
    return np.clip(fused - penalty, 0.0, 1.0)

# ======================== CORE SEARCH ========================
def search_color_voting(
    db_path: str,
    q_hist: np.ndarray,
    hist_col: str = HIST_COL,
    metrics: Sequence[str] = DEFAULT_METRICS,
    weight_map: Optional[Dict[str, float]] = None,
    topk: int = 10,
) -> List[Dict[str, Any]]:
    if weight_map is None:
        weight_map = PRESET_WEIGHTS["balanced"]
    if not isinstance(weight_map, dict):
        raise TypeError(f"'weight_map' muss dict sein, bekommen: {type(weight_map).__name__}")

    ids: List[int] = []
    paths: List[str] = []
    feats: List[np.ndarray] = []
    tabs: List[str] = []

    con = sqlite3.connect(db_path)
    cur = con.cursor()
    tables = list_feature_tables(con, TABLE_PREFIX)

    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = {r[1] for r in cur.fetchall()}
        if "id" not in cols or hist_col not in cols:
            continue

        path_col = None
        for candidate in ("filepath", "path", "image_path"):
            if candidate in cols:
                path_col = candidate
                break

        sel_cols = f"id,{hist_col}" + (f",{path_col}" if path_col else "")
        cur.execute(f"SELECT {sel_cols} FROM {t} WHERE {hist_col} IS NOT NULL")

        for row in cur.fetchall():
            _id = int(row[0])
            hist_csv = row[1]
            try:
                h = parse_hist_text(hist_csv)
            except Exception:
                continue
            ids.append(_id)
            feats.append(h)
            tabs.append(t)
            paths.append(row[2] if path_col else None)

    con.close()

    if not feats:
        return []

    X = np.vstack(feats).astype(np.float32)
    X = (X.T / (X.sum(axis=1) + 1e-12)).T

    q = q_hist.astype(np.float32)
    q = q / (q.sum() + 1e-12)

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

    S = [weight_map.get(m, 1.0) * to_similarity(m, D[m]) for m in metrics]
    wsum = sum(weight_map.get(m, 1.0) for m in metrics)
    fused = np.sum(S, axis=0) / (wsum + 1e-12)

    fused = apply_channel_constraints_universal(fused, X, q, bins=BINS)

    k = min(topk, fused.size)
    idx = np.argpartition(-fused, k - 1)[:k]
    idx = idx[np.argsort(-fused[idx])]

    out: List[Dict[str, Any]] = []
    for i in idx:
        per_metric = {m: float(to_similarity(m, D[m][i:i+1])[0]) for m in metrics}
        out.append({
            "table": tabs[i],
            "id": int(ids[i]),
            "path": None if paths[i] is None else str(paths[i]),
            "fused_similarity": float(fused[i]),
            "per_metric": per_metric,
        })
    return out

# ======================== CLI ========================
def parse_args():
    p = argparse.ArgumentParser(description="Color-Histogramm Suche mit universellen Kanal-Constraints.")
    p.add_argument("--db", dest="db_path", required=True, help="Pfad zur SQLite-DB")
    p.add_argument("--QUERY_IMG", dest="query_img", required=True, help="Pfad zum Query-Bild")
    p.add_argument("--top_k", dest="top_k", type=int, default=10, help="Anzahl der Top-Ergebnisse")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # 1) Bild laden
    img = cv2.imread(args.query_img, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Konnte Bild nicht laden: {args.query_img}")

    # 2) Query-Hist [B|G|R]
    q_bgr = calc_histogram(img, bins=BINS)

    # 3) Optionaler Swap B<->R zum DB-Match (ergibt [R|G|B])
    q_db = swap_rb_hist(q_bgr, BINS) if DO_SWAP_RB else q_bgr

    # 4) Gewichte
    if USE_PRESET in PRESET_WEIGHTS:
        WEIGHTS = PRESET_WEIGHTS[USE_PRESET]
    else:
        WEIGHTS = auto_weights(q_db, BINS)
    print("Weights:", WEIGHTS)

    # 5) Suche
    results = search_color_voting(
        db_path=args.db_path,
        q_hist=q_db,
        hist_col=HIST_COL,
        metrics=DEFAULT_METRICS,
        weight_map=WEIGHTS,
        topk=args.top_k,
    )

    # 6) Ausgabe
    if not results:
        print("Keine Ergebnisse gefunden (prüfe Tabellen/Spalten).")
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i:02d}] sim={r['fused_similarity']:.4f}  "
                  f"path={r['path']}  (table={r['table']}, id={r['id']})")
            print(f"     per_metric: {r['per_metric']}")



#python src/similarity/color.py --db "C:\BIG_DATA\data\database.db" --QUERY_IMG "Z:\CODING\UNI\BIG_DATA\data\TEST_IMAGES\kontrastierendes-outdoor-texturdesign.jpg" --top_k 5
