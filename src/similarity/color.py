# color_search_variantA_nargs.py
# -*- coding: utf-8 -*-
"""
Variante A (ohne Args):
- Query-Histogramm wird als [B|G|R] berechnet (OpenCV/BGR).
- Danach R↔B getauscht -> [R|G|B], um eine DB zu matchen, die effektiv RGB gespeichert hat.
- Universelle Kanal-Constraints verhindern "zusätzliche" Farben (egal welcher Kanal ~0 ist).
- Keine Änderungen an der DB nötig.

CONFIG unten anpassen und Skript direkt starten.
"""

import sqlite3
from typing import List, Dict, Any, Sequence, Optional

import cv2
import numpy as np

# ======================== CONFIG ========================
DB_PATH    = r"C:\BIG_DATA\data\database.db"
IMAGE_PATH = r"Z:\CODING\UNI\BIG_DATA\data\TEST_IMAGES\OIP (1).webp"
HIST_COL   = "color_hist"
TOPK       = 10

# Falls du zum Gegencheck OHNE Swap testen willst -> False
DO_SWAP_RB = True

TABLE_PREFIX = "image_features_part_"   # Tabellenpräfix
BINS = 32
DIM  = 3 * BINS

# Welche Metriken werden genutzt (Reihenfolge bleibt beim Reporting)
# Tipp: Für strenge 2-Farben-Queries Intersection rausnehmen.
DEFAULT_METRICS = ("chi2", "hellinger", "intersect", "emd")

# Presets für feste Gewichte (falls du Auto-Weights nicht willst)
PRESET_WEIGHTS = {
    "balanced":   {"chi2": 1.8, "hellinger": 1.0, "intersect": 0.2, "emd": 1.3},
    "two_color":  {"chi2": 2.2, "hellinger": 0.8, "intersect": 0.0, "emd": 1.5},
    "flat_logo":  {"chi2": 1.0, "hellinger": 0.8, "intersect": 1.2, "emd": 0.5},
}
USE_PRESET: Optional[str] = None   # z.B. "balanced" setzen, um Auto-Weights zu überschreiben

# ======================== AUTO-WEIGHTS ========================
def auto_weights(q: np.ndarray, bins: int = 32) -> Dict[str, float]:
    """
    Leichtes Auto-Tuning anhand der Query-Verteilung.
    """
    ch = q.reshape(3, bins).sum(axis=1)  # (B,G,R)
    p  = q[q > 0]
    H  = -(p * np.log(p)).sum() / np.log(q.size)  # normierte Entropie ∈[0,1]

    # Default: Balanced
    w = {"chi2": 1.8, "hellinger": 1.0, "intersect": 0.2, "emd": 1.3}

    # Einfarbig (niedrige Entropie) -> Intersection etwas stärker, EMD etwas runter
    if H < 0.55:
        w.update({"intersect": 1.0, "emd": 0.7})

    # Zwei dominante Kanäle -> χ² + EMD hoch, Intersection aus
    if (ch > ch.sum() * 0.40).sum() == 2:
        w.update({"chi2": 2.2, "emd": 1.6, "intersect": 0.0})

    return w

# ======================== UTILS ========================
def to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:  # Grau -> BGR
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:  # RGBA -> BGR
        img = img[..., :3]
    if img.dtype == np.uint8:
        return img
    im = img.astype(np.float32)
    if im.max() <= 1.0 + 1e-6:
        im *= 255.0
    return np.clip(im, 0, 255).astype(np.uint8)

def calc_histogram(img_bgr: np.ndarray, bins: int = BINS) -> np.ndarray:
    """
    Erwartet BGR (OpenCV).
    Liefert L1-normalisiertes Histogramm [B | G | R] (Länge 3*bins).
    """
    img_u8 = to_uint8(img_bgr)
    hists = []
    for ch in (0, 1, 2):  # B, G, R
        h = cv2.calcHist([img_u8], [ch], None, [bins], [0, 256]).ravel().astype(np.float32)
        hists.append(h)
    hist = np.concatenate(hists)
    s = hist.sum()
    if s > 0:
        hist /= s
    return hist

def swap_rb_hist(v: np.ndarray, bins: int = BINS) -> np.ndarray:
    """
    [B|G|R] -> [R|G|B]
    """
    b, g, r = v[:bins], v[bins:2 * bins], v[2 * bins:]
    out = np.concatenate([r, g, b])
    out /= (out.sum() + 1e-12)
    return out

def parse_hist_text(s: str) -> np.ndarray:
    """
    CSV-Text (float32) -> L1-normalisiertes Array (DIM,).
    """
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
    return 1.0 - sim  # ∈[0,1]

def hellinger_distance(X: np.ndarray, q: np.ndarray) -> np.ndarray:
    sqX = np.sqrt(X)
    sqq = np.sqrt(q)
    return np.linalg.norm(sqX - sqq, axis=1) / np.sqrt(2.0)  # ∈[0,1]

def emd1d_per_channel_distance(X: np.ndarray, q: np.ndarray, bins: int = BINS) -> np.ndarray:
    """
    1D-EMD je Kanal über kumulative Summen, dann Mittel über 3 Kanäle.
    Normalisierung durch (bins-1), damit ∈[0,1].
    """
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
    tol_frac: float = 0.010,   # ~1% Toleranz pro Kanal
    alpha_missing: float = 0.90,  # Strafe für "extra" Masse in quasi-leeren Query-Kanälen
    alpha_bins: float = 0.50,     # Strafe pro Bin in verbotenen Bereichen
    alpha_prop: float = 0.35      # Strafe für falsches Verhältnis der aktiven Kanäle
) -> np.ndarray:
    """
    Universelle Kanal-Constraints:
      - funktioniert für beliebige Query-Kombis (1-, 2- oder 3-Kanal),
      - kein Sonderfall "grün"; alle Kanäle werden symmetrisch behandelt.
    """
    X3 = X.reshape(-1, 3, bins)    # (N,3,B)
    q3 = q.reshape(3, bins)        # (3,B)

    # Kanal-Summen
    q_ch = q3.sum(axis=1)          # (3,)
    X_ch = X3.sum(axis=2)          # (N,3)

    # (A) Query-Kanal ~0? -> "extra" Masse bestrafen
    tol_ch = tol_frac
    missing_mask = (q_ch < tol_ch).astype(np.float32)                         # (3,)
    extra_mass = np.maximum(X_ch - (q_ch + tol_ch), 0.0) * missing_mask       # (N,3)
    penalty_missing = alpha_missing * extra_mass.sum(axis=1)                  # (N,)

    # (B) Bin-Level: Masse in Bins, die Query fast 0 hat, runterziehen
    tol_bin = tol_ch / max(bins, 1)
    forbid_bins = (q3 < tol_bin).astype(np.float32)                           # (3,B)
    penalty_bins = alpha_bins * (X3 * forbid_bins).sum(axis=(1, 2))           # (N,)

    # (C) Verhältnis der "aktiven" Kanäle (die nicht quasi 0 sind)
    active_idx = np.where(q_ch >= tol_ch)[0]                                  # z.B. [0,2] bei B+R
    if active_idx.size >= 2:
        cq = q_ch[active_idx] / (q_ch[active_idx].sum() + 1e-12)              # (K,)
        cx = X_ch[:, active_idx]
        cx = (cx.T / (cx.sum(axis=1) + 1e-12)).T                              # (N,K)
        # L1-Abstand der Kanalverhältnisse (sanft, skaliert in [0,2])
        prop_l1 = np.abs(cx - cq).sum(axis=1)                                 # (N,)
        penalty_prop = alpha_prop * 0.5 * prop_l1                             # ~[0,1]
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
    topk: int = TOPK,
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
    X = (X.T / (X.sum(axis=1) + 1e-12)).T  # L1-Sicherheitsnormierung

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

    # >>> Universelle Kanal-Constraints <<<
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

# ======================== RUN ========================
if __name__ == "__main__":
    # 1) Bild laden (BGR)
    img = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Konnte Bild nicht laden: {IMAGE_PATH}")

    # 2) Query-Hist [B|G|R]
    q_bgr = calc_histogram(img, bins=BINS)

    # 3) Variante A: Query -> [R|G|B] (R↔B tauschen), um DB zu matchen
    q_db = swap_rb_hist(q_bgr, BINS) if DO_SWAP_RB else q_bgr

    # 4) Gewichte bestimmen
    if USE_PRESET in PRESET_WEIGHTS:
        WEIGHTS = PRESET_WEIGHTS[USE_PRESET]
    else:
        WEIGHTS = auto_weights(q_db, BINS)  # Auto-Tuning anhand der Query
    print("Weights:", WEIGHTS)

    # 5) Suche
    results = search_color_voting(
        db_path=DB_PATH,
        q_hist=q_db,
        hist_col=HIST_COL,
        metrics=DEFAULT_METRICS,
        weight_map=WEIGHTS,   # <-- dict, nicht Funktion
        topk=TOPK,
    )

    # 6) Ausgabe
    if not results:
        print("Keine Ergebnisse gefunden (prüfe Tabellen/Spalten).")
    else:
        for i, r in enumerate(results, 1):
            print(f"[{i:02d}] sim={r['fused_similarity']:.4f}  "
                  f"path={r['path']}  (table={r['table']}, id={r['id']})")
            print(f"     per_metric: {r['per_metric']}")
