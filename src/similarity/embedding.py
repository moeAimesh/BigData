#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cascade_search_3stage_json_umap_separate_models_FIXED.py

3-stufige Suche (32 -> 64 -> 512) mit JSON-Output und optionalem UMAP-Plot.
Wesentliche Fixes ggü. deiner Version:
- Query-Embedding exakt wie im Ingest: (H,W,3) uint8 -> resize -> float32 [0..1] -> an extract_embeddings([...]).
  (extract_embeddings wendet intern erneut /255 & ImageNet-Standardisierung an – das entspricht deiner DB-Pipeline.)
- 32D = erste 32 Komponenten von 64D (q64_raw[:32]) – so wie deine pca_32 entstanden ist.
- HNSW-Index wird auf "Staleness" geprüft und bei Missmatch automatisch neu gebaut.
- Saubere DB-Handles, kleine Robustheits-Checks.

Nur --db und --image als Args. Alles andere oben konfigurieren.
"""

import os, sys, argparse, sqlite3, pickle, json, time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from joblib import load
import hnswlib
import matplotlib.pyplot as plt

# ===================== EXTERNAL (Feature-Extractor) =====================

# in src/similarity/embedding.py
try:
    # 1) wenn embedding_vec.py im gleichen Ordner liegt: src/similarity/embedding_vec.py
    from .embedding_vec import extract_embeddings
except ImportError:
    try:
        # 2) wenn es eine Ebene höher liegt: src/embedding_vec.py
        from ..embedding_vec import extract_embeddings
    except Exception as e:
        raise ImportError(
            "Konnte 'extract_embeddings' nicht importieren. "
            "Lege embedding_vec.py entweder in 'src/similarity/' (bevorzugt) "
            "oder in 'src/' ab und nutze die relativen Importe wie oben."
        ) from e

# ===================== KONFIG =====================
# Modelle
MODELS_DIR        = Path(r"C:\BIG_DATA\models")
IPCA_PATH         = MODELS_DIR / "ipca.joblib"   # IncrementalPCA – exakt dasselbe Modell wie im DB-Build für 64D!

# Separate UMAP-Modelle je Dimension (optional)
UMAP32_PATH       = MODELS_DIR / "umap32_reducer.joblib"
UMAP64_PATH       = MODELS_DIR / "umap_reducer.joblib"
UMAP512_PATH      = MODELS_DIR / "umap512_reducer.joblib"

# Für UMAP.transform() normierten Query-Vektor verwenden?
UMAP_TRANSFORM_USE_L2 = {32: False, 64: False, 512: False}  # metric='euclidean' -> False, metric='cosine' -> True

# HNSW (32D)
INDEX_DIR   = Path(r"C:\BIG_DATA\ann_index_32d")
INDEX_BIN   = INDEX_DIR / "hnsw_32d.bin"
INDEX_META  = INDEX_DIR / "labels_meta.pkl"
HNSW_M              = 16
HNSW_EF_CONSTRUCT   = 200
HNSW_EF_SEARCH      = 200

# Tabellen- und Spaltennamen
TABLE_PREFIX    = "image_features_part_"
COL_ID          = "id"
COL_IMAGE_PATH  = "path"
COL_EMB32_PATH  = "pca_32"         # 32D .npy (erste 32 Komponenten deines 64D PCA)
COL_EMB64_PATH  = "pca_embedding"  # 64D (oder >=64) .npy
COL_EMB512_PATH = "embedding_path" # 512D .npy

# UMAP-Koordinaten-Spalten je Dimension
UMAP_COLS = {
    32: ("umap32_x",  "umap32_y"),
    64: ("umap_x",    "umap_y"),
    512:("umap512_x", "umap512_y"),
}

# Such-Umfang
TOP_K      = 5
X_FACTOR   = 160   # X ≈ 40*k
Y_FACTOR   = 32    # Y ≈ 8*k
M_FACTOR   = 8    # M ≈ 2*k

# Plot-Schalter
PLOT_ENABLED   = False
PLOT_OUTPUT    = Path("umap_topX_Y_M.png")

# Query-Preprocess (exakt wie im Ingest)
TARGET_SIZE = (224, 224)  # (W,H)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False
import cv2  # wir nutzen cv2.imdecode für robustes Lesen (Unicode-Pfade etc.)

# OPTIONAL: Self-Check (hilfreich beim Debuggen)
DEBUG_SELF_CHECK = False  # auf True setzen, wenn du für *dieses* Query-Bild den 512D Self-Match prüfen willst


# ===================== UTILS =====================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.ndim == 1:
        n = np.linalg.norm(mat) + eps
        return (mat / n).astype(np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return (mat / n).astype(np.float32)

def _fast_load_bytes_bgr(p: Path) -> np.ndarray:
    """Robustes Lesen über Bytes + imdecode (BGR). Keine Farbkonvertierung – wir replizieren deine Ingest-Annahme."""
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError(f"cv2.imdecode konnte Bild nicht lesen: {p}")
    return img  # BGR

def preprocess_for_model(img_uint8: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    """
    EXAKT wie in deiner Ingest-Pipeline:
      - nur Resize auf (224,224)
      - float32 in [0..1]
      - KEINE Mean/Std hier, KEIN Channel-Swap.
    """
    if PIL_AVAILABLE:
        im = Image.fromarray(img_uint8)  # interpretiert Array wie es kommt; in deiner Pipeline war das konsistent
        im = im.resize(target_size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    else:
        arr = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return arr  # (224,224,3) float32 [0..1]

def to_embed_dtype(arr: np.ndarray) -> np.ndarray:
    # In deiner Ingest war EMBED_INPUT_DTYPE="float32" – einfach durchreichen.
    return arr.astype(np.float32, copy=False)


# ===================== PCA =====================
class PCATransformer:
    def __init__(self, ipca_path: Path):
        if not ipca_path.exists():
            raise FileNotFoundError(f"IPCA-Modell nicht gefunden: {ipca_path}")
        self.ipca = load(ipca_path)  # IncrementalPCA
    def transform64(self, vec512_raw: np.ndarray) -> np.ndarray:
        # Erwartet den ROHEN 512D-Vektor (nicht normiert), exakt wie im DB-Build
        return self.ipca.transform(vec512_raw[None, :]).astype(np.float32)[0]


# ===================== HNSW (32D) =====================
class HNSW32:
    def __init__(self, dim=32):
        self.index = hnswlib.Index(space='ip', dim=dim)  # IP + L2-Norm -> Cosine
        # labels_meta[i] = (table, id, image_path, emb32_path, emb64_path, emb512_path)
        self.labels_meta: List[Tuple[str, int, str, str, str, str]] = []
    def save(self):
        ensure_dir(INDEX_DIR)
        self.index.save_index(str(INDEX_BIN))
        with open(INDEX_META, 'wb') as f:
            pickle.dump(self.labels_meta, f)
    def load(self):
        self.index.load_index(str(INDEX_BIN))
        with open(INDEX_META, 'rb') as f:
            self.labels_meta = pickle.load(f)
        self.index.set_ef(HNSW_EF_SEARCH)
    def init_new(self, max_elements: int):
        self.index.init_index(max_elements=max_elements, ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M)
        self.index.set_ef(HNSW_EF_SEARCH)
    def add_batch(self, vecs32: np.ndarray, metas: List[Tuple[str,int,str,str,str,str]]):
        start = len(self.labels_meta)
        labels = np.arange(start, start + vecs32.shape[0])
        self.index.add_items(vecs32, labels)
        self.labels_meta.extend(metas)
    def knn(self, q32: np.ndarray, k: int):
        labels, sims = self.index.knn_query(q32[None, :], k=k)
        return labels[0], sims[0].astype(np.float32)


# ===================== DB =====================
def list_feature_tables(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name ASC",
                (f"{TABLE_PREFIX}%",))
    return [r[0] for r in cur.fetchall()]

def count_rows(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{table}"')
    return int(cur.fetchone()[0])

def table_has_columns(conn: sqlite3.Connection, table: str, cols: Tuple[str, str]) -> bool:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    names = {r[1] for r in cur.fetchall()}
    return cols[0] in names and cols[1] in names

def fetch_rows_iter(conn: sqlite3.Connection, table: str, batch: int = 20000):
    # Rückgabe: (id, image_path, emb32_path, emb64_path, emb512_path)
    cur = conn.cursor()
    cur.execute(
        f'SELECT {COL_ID}, {COL_IMAGE_PATH}, {COL_EMB32_PATH}, {COL_EMB64_PATH}, {COL_EMB512_PATH} FROM "{table}"'
    )
    while True:
        rows = cur.fetchmany(batch)
        if not rows: break
        for r in rows:
            yield int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])

def fetch_umap_coords(conn: sqlite3.Connection, table: str, ids: List[int], cols: Tuple[str, str]) -> Dict[int, Tuple[float, float]]:
    if not ids: return {}
    q_marks = ",".join("?" for _ in ids)
    sql = f'SELECT {COL_ID}, "{cols[0]}", "{cols[1]}" FROM "{table}" WHERE {COL_ID} IN ({q_marks})'
    cur = conn.cursor()
    cur.execute(sql, ids)
    out = {}
    for rid, x, y in cur.fetchall():
        if x is not None and y is not None:
            out[int(rid)] = (float(x), float(y))
    return out


# ===================== Index-Build (32D, Konsistenz-Check) =====================
def build_index_if_needed(db_path: Path) -> HNSW32:
    idx = HNSW32(dim=32)
    need_rebuild = True
    if INDEX_BIN.exists() and INDEX_META.exists():
        try:
            idx.load()
            with sqlite3.connect(str(db_path)) as conn:
                tables = list_feature_tables(conn)
                db_total = sum(count_rows(conn, t) for t in tables)
            index_count = idx.index.get_current_count()
            meta_len    = len(idx.labels_meta)
            if db_total == index_count == meta_len:
                need_rebuild = False
            else:
                print(f"[WARN] Index stale? db_total={db_total:,}, index={index_count:,}, meta={meta_len:,} → rebuild")
        except Exception:
            need_rebuild = True

    if not need_rebuild:
        idx.index.set_ef(HNSW_EF_SEARCH)
        return idx

    # ===== rebuild from DB =====
    print("[Info] Baue 32D-HNSW aus DB …")
    with sqlite3.connect(str(db_path)) as conn:
        tables = list_feature_tables(conn)
        if not tables:
            raise RuntimeError(f"Keine Tabellen mit Präfix '{TABLE_PREFIX}' gefunden.")
        total = sum(count_rows(conn, t) for t in tables)
        print(f"[Info] Elemente gesamt: {total:,}")
        idx.init_new(max_elements=total)

        for t in tables:
            vecs, metas = [], []
            for rid, img_path, emb32_path, emb64_path, emb512_path in fetch_rows_iter(conn, t):
                try:
                    v32 = np.load(emb32_path, mmap_mode='r')
                    v32 = v32[0] if (v32.ndim == 2 and v32.shape[0] == 1) else v32
                    if v32.shape[0] < 32: 
                        continue
                    # deine 32D sind die ersten 32 aus dem 64D – in Dateien schon 32D abgelegt
                    vecs.append(v32[:32].astype(np.float32))
                    metas.append((t, rid, img_path, emb32_path, emb64_path, emb512_path))
                except Exception:
                    continue
                if len(vecs) >= 20000:
                    idx.add_batch(l2_normalize_rows(np.vstack(vecs)), metas)
                    vecs, metas = [], []
            if vecs:
                idx.add_batch(l2_normalize_rows(np.vstack(vecs)), metas)
            print(f"[Info] Tabelle {t}: hinzugefügt.")
    idx.save()
    print(f"[Info] Index gespeichert: {INDEX_BIN}")
    return idx


# ===================== Re-Ranking =====================
def rerank_mid_64(q64: np.ndarray, cand_meta, topY: int):
    mats, valid_idx = [], []
    for i, (_, _, _, _, emb64_path, _) in enumerate(cand_meta):
        try:
            v64 = np.load(emb64_path, mmap_mode='r')
            v64 = v64[0] if (v64.ndim == 2 and v64.shape[0] == 1) else v64
            if v64.size < 64:
                continue
            v64 = v64.astype(np.float32).ravel()[:64]
            mats.append(v64)
            valid_idx.append(i)
        except Exception:
            continue
    if not mats:
        return []
    M = l2_normalize_rows(np.vstack(mats))
    q = q64.astype(np.float32).ravel()
    sims = M @ q
    order = np.argsort(-sims)[:topY]
    return [(valid_idx[int(i)], float(sims[int(i)])) for i in order]

def rerank_fine_512(q512: np.ndarray, cand_meta: List[Tuple[int, Tuple[str,int,str,str,str,str]]], topM: int) -> List[Tuple[int, float]]:
    mats, valid_idx = [], []
    for cand_idx, meta in cand_meta:
        _, _, _, _, _, emb512_path = meta
        try:
            v512 = np.load(emb512_path, mmap_mode='r')
            v512 = v512[0] if (v512.ndim == 2 and v512.shape[0] == 1) else v512
            mats.append(v512.astype(np.float32).ravel())
            valid_idx.append(cand_idx)
        except Exception:
            continue
    if not mats: return []
    M = l2_normalize_rows(np.vstack(mats))
    sims = M @ q512.astype(np.float32).ravel()
    order = np.argsort(-sims)[:topM]
    return [(valid_idx[int(i)], float(sims[int(i)])) for i in order]


# ===================== UMAP =====================
def load_umap_models_separate() -> Dict[int, Any]:
    models: Dict[int, Any] = {}
    try:
        if UMAP32_PATH and UMAP32_PATH.exists(): models[32] = load(UMAP32_PATH)
    except Exception: pass
    try:
        if UMAP64_PATH and UMAP64_PATH.exists(): models[64] = load(UMAP64_PATH)
    except Exception: pass
    try:
        if UMAP512_PATH and UMAP512_PATH.exists(): models[512] = load(UMAP512_PATH)
    except Exception: pass
    return models

def plot_umap_from_db(db_path: Path,
                      coarse_idxs: List[int],
                      mid_pairs: List[Tuple[int, float]],
                      fine_pairs: List[Tuple[int, float]],
                      cand_meta: List[Tuple[str,int,str,str,str,str]],
                      q_umap_vecs: Dict[int, np.ndarray],
                      out_path: Path):
    with sqlite3.connect(str(db_path)) as conn:
        # IDs pro Tabelle & Dim sammeln
        by_table_dim: Dict[Tuple[str,int], List[int]] = {}
        for cand_idx in coarse_idxs:
            table, rid, *_ = cand_meta[cand_idx]
            by_table_dim.setdefault((table, 32), []).append(rid)
        for cand_idx, _ in mid_pairs:
            table, rid, *_ = cand_meta[cand_idx]
            by_table_dim.setdefault((table, 64), []).append(rid)
        for cand_idx, _ in fine_pairs:
            table, rid, *_ = cand_meta[cand_idx]
            by_table_dim.setdefault((table, 512), []).append(rid)

        # DB-Koordinaten holen
        coords: Dict[int, Dict[int, Tuple[float,float]]] = {32:{}, 64:{}, 512:{}}
        for (table, dim), ids in by_table_dim.items():
            cols = UMAP_COLS[dim]
            if table_has_columns(conn, table, cols):
                coords[dim].update(fetch_umap_coords(conn, table, ids, cols))

    # Query-Koords
    q_coords: Dict[int, Tuple[float,float]] = {}
    try:
        models = load_umap_models_separate()
        for dim, qv in q_umap_vecs.items():
            if dim in models:
                XYq = models[dim].transform(qv[None, :])
                q_coords[dim] = (float(XYq[0,0]), float(XYq[0,1]))
    except Exception:
        pass

    # Plot
    plt.figure(figsize=(9, 7))
    colors = {32: "red", 64: "blue", 512: "green"}

    def _scatter(dim: int, label: str, src_idxs: List[int]):
        xs, ys = [], []
        for cand_idx in src_idxs:
            _, rid, *_ = cand_meta[cand_idx]
            if rid in coords[dim]:
                x,y = coords[dim][rid]
                xs.append(x); ys.append(y)
        if xs:
            plt.scatter(xs, ys, s=16, alpha=0.7, c=colors[dim], label=label)

    _scatter(32, "Top-X (32D)", coarse_idxs)
    _scatter(64, "Top-Y (64D)", [i for i,_ in mid_pairs])
    _scatter(512,"Top-M (512D)", [i for i,_ in fine_pairs])

    for dim in (32, 64, 512):
        if dim in q_coords:
            qx, qy = q_coords[dim]
            plt.scatter([qx],[qy], s=110, marker="*", edgecolor="black", linewidths=0.8,
                        c=colors[dim], label=f"Query ({dim}D)")

    plt.title("UMAP: Top-X/Y/M aus DB + Query")
    plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    plt.legend(loc="best")
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[UMAP] Plot gespeichert: {out_path}")


# ===================== SEARCH CORE =====================
def search_image(image_path: Path, db_path: Path) -> Dict[str, Any]:
    # Modelle / Index
    pcat = PCATransformer(IPCA_PATH)
    index32 = build_index_if_needed(db_path)

    # --- Query-Embedding EXAKT wie im Ingest bauen ---
    img_arr   = _fast_load_bytes_bgr(image_path)            # (H,W,3) BGR uint8 – wie in deiner Pipeline angenommen
    img_small = preprocess_for_model(img_arr)               # (224,224,3) float32 [0..1]
    embed_in  = to_embed_dtype(img_small)                   # float32
    embs      = extract_embeddings([embed_in])              # -> (1,512); intern erneut /255 & ImageNet-Norm
    q512_raw  = np.asarray(embs[0], dtype=np.float32).ravel()
    assert q512_raw.size == 512, f"Unerwartete Query-Embedding-Länge: {q512_raw.shape}"

    # (Optional) Self-Check gegen DB (nur wenn das Bild in der DB ist und DEBUG_SELF_CHECK=True)
    if DEBUG_SELF_CHECK:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.cursor()
            cur.execute(f'SELECT {COL_EMB512_PATH} FROM "{TABLE_PREFIX}1" WHERE {COL_IMAGE_PATH}=? LIMIT 1', (str(image_path),))
            row = cur.fetchone()
        if row and row[0] and os.path.exists(row[0]):
            v_db = np.load(row[0]).astype(np.float32).ravel()
            def l2n(x): x = x.astype(np.float32); return x/(np.linalg.norm(x)+1e-12)
            cos512 = float(l2n(q512_raw) @ l2n(v_db))
            print(f"[DBG] cos512 self = {cos512:.6f}")

    # PCA → 64D, 32D (ROH)
    q64_raw = pcat.transform64(q512_raw)  # (64,)
    q32_raw = q64_raw[:32]                # (32,)  ← entspricht deiner pca_32 Definition

    # L2-Normierungen für Cosine/IP
    q512 = l2_normalize_rows(q512_raw)
    q64  = l2_normalize_rows(q64_raw)
    q32  = l2_normalize_rows(q32_raw)

    # ----- Coarse (32D HNSW) -----
    topX = max(TOP_K * X_FACTOR, TOP_K)
    labels, coarse_sims = index32.knn(q32, topX)
    cand_meta = [index32.labels_meta[int(lbl)] for lbl in labels]  # (table,id,img,emb32,emb64,emb512)
    coarse_map = {i: float(coarse_sims[i]) for i in range(len(labels))}

    # ----- Mid (64D Re-Ranking) -----
    topY = max(TOP_K * Y_FACTOR, TOP_K)
    mid = rerank_mid_64(q64, cand_meta, topY)                         # [(cand_idx, mid_sim)]
    if not mid: 
        return {"query":{"image_path":str(image_path)}, "results":{"final_top_k":[]}, "params":{"k":TOP_K}}

    # ----- Fine (512D Re-Ranking) -----
    Y_list = [(cand_idx, cand_meta[cand_idx]) for (cand_idx, _) in mid]
    topM = max(TOP_K * M_FACTOR, TOP_K)
    fine = rerank_fine_512(q512, Y_list, topM)                        # [(cand_idx, fine_sim)]
    if not fine: 
        return {"query":{"image_path":str(image_path)}, "results":{"final_top_k":[]}, "params":{"k":TOP_K}}

    # ----- JSON-Listen -----
    coarse_top_x, mid_top_y, fine_top_m = [], [], []

    for rank, cand_idx in enumerate(range(min(len(cand_meta), topX)), start=1):
        if cand_idx not in coarse_map: 
            continue
        table, rid, img, p32, p64, p512 = cand_meta[cand_idx]
        coarse_top_x.append({
            "label": "32D", "score": coarse_map[cand_idx], "rank": rank,
            "table": table, "id": rid, "image_path": img,
            "path_32": p32, "path_64": p64, "path_512": p512,
        })

    for rank, (cand_idx, mid_sim) in enumerate(mid, start=1):
        table, rid, img, p32, p64, p512 = cand_meta[cand_idx]
        mid_top_y.append({
            "label": "64D", "score": mid_sim, "rank": rank,
            "table": table, "id": rid, "image_path": img,
            "path_32": p32, "path_64": p64, "path_512": p512,
        })

    for rank, (cand_idx, fine_sim) in enumerate(fine, start=1):
        table, rid, img, p32, p64, p512 = cand_meta[cand_idx]
        fine_top_m.append({
            "label": "512D", "score": fine_sim, "rank": rank,
            "table": table, "id": rid, "image_path": img,
            "path_32": p32, "path_64": p64, "path_512": p512,
        })

    final_top_k = sorted(fine_top_m, key=lambda d: -float(d["score"]))[:TOP_K]

    # ----- (Optional) UMAP-Plot + Query-UMAP-Koordinaten -----
    if PLOT_ENABLED:
        q_umap_vecs = {
            32: (q32 if UMAP_TRANSFORM_USE_L2.get(32, False) else q32_raw),
            64: (q64 if UMAP_TRANSFORM_USE_L2.get(64, False) else q64_raw),
            512:(q512 if UMAP_TRANSFORM_USE_L2.get(512, False) else q512_raw),
        }
        coarse_idxs = list(range(min(len(cand_meta), topX)))
        plot_umap_from_db(
            db_path=db_path,
            coarse_idxs=coarse_idxs,
            mid_pairs=mid,
            fine_pairs=fine,
            cand_meta=cand_meta,
            q_umap_vecs=q_umap_vecs,
            out_path=PLOT_OUTPUT
        )

    # UMAP-Koords aus DB anreichern
    def attach_umap_to_items(items: List[dict], dim: int):
        if not items: return
        with sqlite3.connect(str(db_path)) as conn:
            by_table: Dict[str, List[int]] = {}
            for it in items:
                by_table.setdefault(it["table"], []).append(it["id"])
            coords = {}
            for table, ids in by_table.items():
                cols = UMAP_COLS[dim]
                if table_has_columns(conn, table, cols):
                    coords.update(fetch_umap_coords(conn, table, ids, cols))
            for it in items:
                rid = it["id"]
                if rid in coords:
                    it["umap"] = {"x": coords[rid][0], "y": coords[rid][1]}

    attach_umap_to_items(coarse_top_x, 32)
    attach_umap_to_items(mid_top_y, 64)
    attach_umap_to_items(fine_top_m, 512)
    attach_umap_to_items(final_top_k, 512)

    return {
        "query": {"image_path": str(image_path)},
        "results": {
            "coarse_top_x": coarse_top_x,
            "mid_top_y": mid_top_y,
            "fine_top_m": fine_top_m,
            "final_top_k": final_top_k
        },
        "params": {"k": TOP_K, "X_factor": X_FACTOR, "Y_factor": Y_FACTOR, "M_factor": M_FACTOR}
    }


# ===================== CLI =====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=str, help="Pfad zur SQLite-DB")
    ap.add_argument("--image", required=True, type=str, help="Pfad zum Query-Bild")
    args = ap.parse_args()

    out = search_image(Path(args.image), Path(args.db))

    # Nur die finalen Top-5 drucken (unabhängig von TOP_K)
    PRINT_TOP_N = 5
    to_print = {
        "query": out.get("query", {}),
        "results": {
            "final_top_k": out["results"]["final_top_k"][:PRINT_TOP_N]
        },
        "params": out.get("params", {})
    }
    print(json.dumps(to_print, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    sys.exit(main())
