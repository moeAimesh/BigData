# src/engine/core.py
# -*- coding: utf-8 -*-
"""
Dispatcher:
  mix        → similarity.mix.search_image
  embedding  → similarity.embedding.search_image
  color      → interne, schnelle Color-Suche mit Lazy-Cache (TEXT→Memmap)
  hash       → similarity.hash (a/d/pHash + Voting)

Color-Cache:
  Beim ersten Aufruf pro Tabelle wird die TEXT-Spalte (z.B. 'color_hist')
  einmalig in Memmap-Dateien umgewandelt:
     <dbdir>/<dbstem>_color_cache/<table>.f32.bin   (float32, N x (3*bins))
     <dbdir>/<dbstem>_color_cache/<table>.ids.npy
     <dbdir>/<dbstem>_color_cache/<table>.paths.npy
     <dbdir>/<dbstem>_color_cache/<table>.meta.json
     <dbdir>/<dbstem>_color_cache/<table>.ok
  Danach wird nur noch aus dem Cache gelesen (kein np.fromstring mehr).
"""

from __future__ import annotations
import sys, json, re, sqlite3, os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# sys.path → "src"
# -----------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# -----------------------------------------------------------------------------
# Externe Modi (unverändert)
# -----------------------------------------------------------------------------
from similarity.mix import search_image as mix_search
from similarity.embedding import search_image as emb_search
from similarity.hash import compute_query_hashes, search_by_hash_voting_multitables
from features.color_vec import calc_histogram  # Query-Hist aus Bild

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class Config:
    mode: str = "mix"                       # "mix" | "embedding" | "color" | "hash"
    top_k: int = 5
    table_prefix: str = "image_features_part_"

    # Color (TEXT -> Lazy-Cache)
    color_bins: int = 32                    # 3*bins = 96
    hist_col_text: str = "color_hist"       # TEXT-Quelle
    # Optional: Gewichte (None => Defaults)
    color_weights: Dict[str, float] | None = None

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def _read_image_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Bild nicht lesbar: {path}")
    return img

def _wrap_final(items: List[Dict[str, Any]], query_path: Path, mode: str, k: int) -> Dict[str, Any]:
    return {
        "query": {"image_path": str(query_path)},
        "results": {"final_top_k": items[:k]},
        "params": {"mode": mode, "k": k},
    }

def warmup(cfg: Config) -> None:
    _ = (mix_search, emb_search, compute_query_hashes, search_by_hash_voting_multitables, calc_histogram)

# -----------------------------------------------------------------------------
# COLOR: Lazy-Cache (nur in core.py)
# -----------------------------------------------------------------------------
EPS = 1e-9
_DEF_W = {"hellinger": 0.35, "chi2": 0.35, "intersect": 0.20, "emd": 0.10}

def _l1n(x: np.ndarray) -> np.ndarray:
    return x / (float(x.sum()) + EPS)

def _hellinger_sim(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    sqM = np.sqrt(M, dtype=np.float32, where=(M > 0))
    sqq = np.sqrt(q, dtype=np.float32, where=(q > 0))
    return sqM @ sqq

def _chi2_sim(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    d = 0.5 * ((M - q) ** 2 / (M + q + EPS)).sum(axis=1)
    return 1.0 / (1.0 + d)

def _intersect_sim(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.minimum(M, q).sum(axis=1)

def _emd1d_sim(M: np.ndarray, q: np.ndarray, bins: int) -> np.ndarray:
    M3 = M.reshape(-1, 3, bins)
    q3 = q.reshape(1, 3, bins)
    cM = M3.cumsum(axis=2)
    cQ = q3.cumsum(axis=2)
    dist = np.abs(cM - cQ).sum(axis=(1, 2))
    return 1.0 / (1.0 + dist)

def _clean_text_to_array(s: str) -> np.ndarray:
    s = s.strip().lstrip("[").rstrip("]")
    s = s.replace(";", ",")
    s = re.sub(r",\s*,", ",", s)
    s = re.sub(r"(nan|NaN|inf|-inf)", "0", s)
    s = s.rstrip(",")
    try:
        return np.fromstring(s, sep=",", dtype=np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def _downsample_1d(x: np.ndarray, from_bins: int, to_bins: int) -> np.ndarray:
    g = from_bins // to_bins
    return x.reshape(to_bins, g).sum(axis=1)

def _to_3x_bins(arr: np.ndarray, bins: int) -> np.ndarray:
    """Akzeptiert 3*{bins,64,256} oder 1*{bins,64,256} und bringt auf 3*bins; 1-Kanal ⇒ Grau repliziert."""
    n = arr.size
    if n == 0:
        return arr
    if n % 3 == 0:
        per = n // 3
        B, G, R = arr[:per], arr[per:2*per], arr[2*per:]
        if per == bins:
            pass
        elif per in (64, 256) and per % bins == 0:
            B = _downsample_1d(B, per, bins)
            G = _downsample_1d(G, per, bins)
            R = _downsample_1d(R, per, bins)
        else:
            return np.array([], np.float32)
        return np.concatenate([B, G, R]).astype(np.float32, copy=False)
    # 1-kanalig → Grau replizieren
    if n in (bins, 64, 256):
        X = arr
        if n != bins:
            if n % bins != 0:
                return np.array([], np.float32)
            X = _downsample_1d(X, n, bins)
        return np.concatenate([X, X, X]).astype(np.float32, copy=False)
    return np.array([], np.float32)

def _cache_dir_for_db(db_path: Path) -> Path:
    return db_path.parent / f"{db_path.stem}_color_cache"

def _table_cache_paths(cache_dir: Path, table: str) -> Dict[str, Path]:
    base = cache_dir / table
    return {
        "mm":   base.with_suffix(".f32.bin"),
        "ids":  base.with_suffix(".ids.npy"),
        "paths":base.with_suffix(".paths.npy"),
        "meta": base.with_suffix(".meta.json"),
        "ok":   base.with_suffix(".ok"),
    }

def _load_cached_table(cache_dir: Path, table: str) -> Tuple[np.memmap, np.ndarray, np.ndarray] | Tuple[None, None, None]:
    P = _table_cache_paths(cache_dir, table)
    if not (P["mm"].exists() and P["ids"].exists() and P["paths"].exists() and P["meta"].exists() and P["ok"].exists()):
        return None, None, None
    meta = json.loads(P["meta"].read_text(encoding="utf-8"))
    N, D = int(meta["N"]), int(meta["D"])
    mm   = np.memmap(P["mm"], mode="r", dtype=np.float32, shape=(N, D))
    ids  = np.load(P["ids"])
    paths= np.load(P["paths"], allow_pickle=True)
    return mm, ids, paths

def _build_cache_for_table(db_path: Path, table: str, hist_col: str, bins: int, cache_dir: Path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    P = _table_cache_paths(cache_dir, table)

    # Rebuild-Guard: wenn teilweise vorhanden, auf .ok warten
    if any(p.exists() for p in (P["mm"], P["ids"], P["paths"], P["meta"])) and not P["ok"].exists():
        import time
        for _ in range(600):
            if P["ok"].exists():
                break
            time.sleep(0.1)
    if P["ok"].exists():
        return _load_cached_table(cache_dir, table)

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(f'SELECT COUNT(*) FROM "{table}" WHERE "{hist_col}" IS NOT NULL AND TRIM("{hist_col}")<>""')
        N = int(cur.fetchone()[0])
        if N == 0:
            P["ok"].write_text("empty", encoding="utf-8")
            return None, None, None

        D  = 3 * bins
        mm = np.memmap(P["mm"], mode="w+", dtype=np.float32, shape=(N, D))
        ids = np.empty((N,), dtype=np.int64)
        paths = np.empty((N,), dtype=object)

        cur.execute(f'SELECT id, "{hist_col}", path FROM "{table}" WHERE "{hist_col}" IS NOT NULL AND TRIM("{hist_col}")<>""')
        i = 0
        BATCH = 20000
        rows = cur.fetchmany(BATCH)
        while rows:
            for rid, txt, p in rows:
                s = txt if isinstance(txt, str) else txt.decode("utf-8", "ignore")
                arr = _to_3x_bins(_clean_text_to_array(s), bins=bins)
                if arr.size != D:
                    continue
                v = _l1n(arr)
                mm[i, :] = v
                ids[i]   = int(rid)
                paths[i] = p
                i += 1
            rows = cur.fetchmany(BATCH)

        # ggf. kürzen, wenn Einträge geskippt wurden
        mm.flush(); del mm
        if i < N:
            data = np.memmap(P["mm"], mode="r", dtype=np.float32, shape=(N, D))[:i].copy()
            data.tofile(P["mm"])
            meta = {"N": i, "D": D}
            np.save(P["ids"],  ids[:i])
            np.save(P["paths"],paths[:i])
        else:
            meta = {"N": N, "D": D}
            np.save(P["ids"],  ids)
            np.save(P["paths"],paths)

        Path(P["meta"]).write_text(json.dumps(meta), encoding="utf-8")
        Path(P["ok"]).write_text("ok", encoding="utf-8")

    return _load_cached_table(cache_dir, table)

def _color_tables(db_path: Path, table_prefix: str) -> List[str]:
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name ASC",
                    (f"{table_prefix}%",))
        return [r[0] for r in cur.fetchall()]

def _search_color_cached_core(
    db_path: Path,
    q_hist: np.ndarray,
    hist_col_text: str,
    bins: int,
    weight_map: Dict[str, float] | None,
    topk: int,
    table_prefix: str,
) -> List[Dict[str, Any]]:
    import heapq

    dbp = Path(db_path)
    cache_dir = _cache_dir_for_db(dbp)

    # L1 + Vorberechnungen
    q = _l1n(q_hist.astype(np.float32, copy=False))
    W = (weight_map or _DEF_W)

    # wie viele Kandidaten pro Tabelle vorselektieren?
    topk_per_table = max(topk * 10, 200)  # z.B. 200 bei K=20

    # Für Hellinger sqrt(q) einmalig berechnen
    sqq = np.sqrt(q, dtype=np.float32, where=(q > 0))

    # Min-Heap für globale Top-K (hält K größte Elemente)
    # Elemente: (score, table, idx_in_table, id, path, per_metric_tuple)
    heap: List[Tuple[float, str, int, int, str, Tuple[float,float,float,float]]] = []

    for t in _color_tables(dbp, table_prefix):
        M, ids, paths = _load_cached_table(cache_dir, t)
        if M is None:
            M, ids, paths = _build_cache_for_table(dbp, t, hist_col_text, bins, cache_dir)
        if M is None or M.shape[0] == 0:
            continue

        # --- Vektorisiert Scores berechnen ---
        # Hellinger
        s_h = (np.sqrt(M, dtype=np.float32, where=(M > 0)) @ sqq)  # (N,)

        # Chi2
        d = 0.5 * ((M - q) ** 2 / (M + q + EPS)).sum(axis=1)
        s_c2 = 1.0 / (1.0 + d)

        # Intersection
        s_i = np.minimum(M, q).sum(axis=1)

        # EMD (abschaltbar, wenn W["emd"]==0)
        if W.get("emd", 0.0) > 0.0:
            M3 = M.reshape(-1, 3, bins)
            q3 = q.reshape(1, 3, bins)
            # cumsum ist teuer → machen wir nur, wenn emd>0
            dist = np.abs(M3.cumsum(axis=2) - q3.cumsum(axis=2)).sum(axis=(1, 2))
            s_e = 1.0 / (1.0 + dist)
        else:
            # Null-Array spart Rechenzeit
            s_e = np.zeros(M.shape[0], dtype=np.float32)

        fused = (W.get("hellinger", 0.35) * s_h
                 + W.get("chi2", 0.35)      * s_c2
                 + W.get("intersect", 0.20) * s_i
                 + W.get("emd", 0.10)       * s_e)

        # --- Nur die besten K′ dieser Tabelle in Betracht ziehen ---
        Kp = min(topk_per_table, fused.shape[0])
        idx_local = np.argpartition(-fused, Kp - 1)[:Kp]  # unsortiert, O(N)
        # Für diese K′ per-metric Werte auslesen
        fh, fc2, fi, fe = s_h[idx_local], s_c2[idx_local], s_i[idx_local], s_e[idx_local]
        ff = fused[idx_local]
        ids_loc = ids[idx_local]
        paths_loc = paths[idx_local]

        # In globalen Heap mergen (K klein halten)
        for i_local in range(Kp):
            item = (float(ff[i_local]), t, int(idx_local[i_local]),
                    int(ids_loc[i_local]), str(paths_loc[i_local]),
                    (float(fh[i_local]), float(fc2[i_local]), float(fi[i_local]), float(fe[i_local])))
            if len(heap) < topk:
                heapq.heappush(heap, item)             # kleiner Heap wächst bis K
            else:
                # pushpop: effizienter als push+pop
                if item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

    # Heap hat K beste (min-heap) → in absteigender Reihenfolge ausgeben
    heap.sort(key=lambda x: -x[0])

    out: List[Dict[str, Any]] = []
    for score, table, _idx, rid, path, (sh, sc2, si, se) in heap:
        out.append({
            "table": table,
            "id": rid,
            "path": path,
            "fused_similarity": score,
            "per_metric": {
                "hellinger": sh,
                "chi2":      sc2,
                "intersect": si,
                "emd":       se,
            },
        })
    return out

# -----------------------------------------------------------------------------
# Haupt-API
# -----------------------------------------------------------------------------
def search_once(image_path: Path, db_path: Path, cfg: Config) -> Dict[str, Any]:
    mode = (cfg.mode or "mix").lower()

    if mode == "mix":
        return mix_search(image_path, db_path)

    if mode == "embedding":
        return emb_search(image_path, db_path)

    if mode == "color":
        img_bgr = _read_image_bgr(image_path)
        q_hist  = calc_histogram(img_bgr, bins=cfg.color_bins).astype(np.float32)
        q_hist  = q_hist / (float(q_hist.sum()) + 1e-9)

        results = _search_color_cached_core(
            db_path=db_path,
            q_hist=q_hist,
            hist_col_text=cfg.hist_col_text,  # TEXT-Quelle (wird ge-cached)
            bins=cfg.color_bins,
            weight_map=cfg.color_weights,
            topk=cfg.top_k,
            table_prefix=cfg.table_prefix,
        )

        final = [{
            "table": r["table"],
            "id": int(r["id"]),
            "image_path": r.get("path") or "",
            "score": float(r["fused_similarity"]),
            "per_metric": r.get("per_metric", {}),
            "label": "color",
        } for r in results]
        return _wrap_final(final, image_path, mode, cfg.top_k)

    if mode == "hash":
        q_ah, q_dh, q_ph = compute_query_hashes(str(image_path))
        out = search_by_hash_voting_multitables(
            db_path=str(db_path),
            q_ahash=q_ah, q_dhash=q_dh, q_phash=q_ph,
            weights="auto",
            topk_per_hash=300,
            final_k=cfg.top_k,
            return_weights=False,
        )
        final = [{
            "table": r["table"],
            "id": int(r["id"]),
            "image_path": r.get("path") or "",
            "score": float(r["score"]),
            "hamming": r.get("hamming", {}),
            "label": "hash",
        } for r in out]
        return _wrap_final(final, image_path, mode, cfg.top_k)

    return mix_search(image_path, db_path)
