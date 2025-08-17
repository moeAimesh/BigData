# cascade_search_3stage_json_umap_separate_models_FIXED_DBHIST_RGB.py
# 3-stufige Suche (32 -> 64 -> 512) + finales Fusions-Re-Ranking (cos512, Farbe aus DB, optional SSIM/LPIPS)
# - Query-Histogramm: nutzt deine calc_histogram()-Funktion (BGR, 3×bins)
# - DB-Histogramme: TEXT (csv) mit 96 Werten (3×32 Bins), L1-normalisiert
# - HNSW 32D + 64D/512D Re-Ranking
# - Optional: UMAP-Plot (ausgelassen, falls nicht benötigt)

import os, sys, argparse, sqlite3, pickle, json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from joblib import load
import hnswlib
import cv2
import matplotlib.pyplot as plt

# ===================== EXTERNAL (Feature-Extractor) =====================
try:
    # bevorzugt: src/similarity/embedding_vec.py
    from .embedding_vec import extract_embeddings
except ImportError:
    try:
        # fallback: src/embedding_vec.py
        from ..embedding_vec import extract_embeddings
    except Exception as e:
        raise ImportError(
            "Konnte 'extract_embeddings' nicht importieren. "
            "Lege embedding_vec.py in 'src/similarity/' oder 'src/' ab."
        ) from e

# ===================== DEIN COLOR-HIST IMPORT =====================
# Ersetzt die Query-Hist-Berechnung. calc_histogram erwartet BGR uint8/float und liefert 3*bins L1-normalisiert.
try:
    # Passe das ggf. an deinen Pfad an, z.B. from utils.color_hist import calc_histogram
    from .color_vec import calc_histogram
except Exception:
    # Fallback (falls der Import mal nicht klappt): kompatible Minimal-Implementierung
    def calc_histogram(img, bins=32):
        if img is None:
            raise ValueError("img is None")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[..., :3]
        if img.dtype != np.uint8:
            im = img.astype(np.float32)
            if im.max() <= 1.0 + 1e-6:
                im *= 255.0
            img = np.clip(im, 0, 255).astype(np.uint8)
        hists = []
        for ch in (0, 1, 2):  # B,G,R
            h = cv2.calcHist([img], [ch], None, [bins], [0, 256]).ravel().astype(np.float32)
            hists.append(h)
        hist = np.concatenate(hists)
        s = hist.sum()
        if s > 0:
            hist /= s
        return hist

# ===================== KONFIG =====================
MODELS_DIR        = Path(r"C:\BIG_DATA\models")
IPCA_PATH         = MODELS_DIR / "ipca.joblib"   # IncrementalPCA (64D)

# UMAP (optional – kann auf Wunsch genutzt/abgeschaltet werden)
UMAP32_PATH       = MODELS_DIR / "umap32_reducer.joblib"
UMAP64_PATH       = MODELS_DIR / "umap_reducer.joblib"
UMAP512_PATH      = MODELS_DIR / "umap512_reducer.joblib"
UMAP_TRANSFORM_USE_L2 = {32: False, 64: False, 512: False}
PLOT_ENABLED   = False
PLOT_OUTPUT    = Path("umap_topX_Y_M.png")

# HNSW (32D)
INDEX_DIR   = Path(r"C:\BIG_DATA\ann_index_32d")
INDEX_BIN   = INDEX_DIR / "hnsw_32d.bin"
INDEX_META  = INDEX_DIR / "labels_meta.pkl"
HNSW_M              = 16
HNSW_EF_CONSTRUCT   = 200
HNSW_EF_SEARCH      = 200

# DB Tabellen / Spalten
TABLE_PREFIX    = "image_features_part_"
COL_ID          = "id"
COL_IMAGE_PATH  = "path"
COL_EMB32_PATH  = "pca_32"         # 32D .npy
COL_EMB64_PATH  = "pca_embedding"  # 64D .npy
COL_EMB512_PATH = "embedding_path" # 512D .npy

# UMAP-Spalten (falls vorhanden)
UMAP_COLS = {32: ("umap32_x","umap32_y"), 64: ("umap_x","umap_y"), 512: ("umap512_x","umap512_y")}

# Farb-Histogramm-Spalte (TEXT, csv, 96 Werte = 3×32 Bins aus B,G,R)
HIST_COL         = "color_hist"  # <--- ggf. exakt auf deinen Spaltennamen anpassen
COLOR_BINS       = 32
COLOR_DIM_EXPECT = 3 * COLOR_BINS

# Such-Umfang
TOP_K      = 20
X_FACTOR   = 160
Y_FACTOR   = 32
M_FACTOR   = 8

# Query-Preprocess (für Embeddings)
TARGET_SIZE = (224, 224)  # (W,H)

# ===================== FUSION KONFIG =====================
WEIGHTS_JSON = r"C:\BIG_DATA\eval\weight_grid_db.json"
FUSION_WEIGHTS_MANUAL = {  # final = w_cos*cos512 - w_lpips*LPIPS + w_ssim*SSIM + w_color*COLOR
    "cos":   0.65,
    "lpips": 0.25,
    "ssim":  0.10,
    "color": 0.10,
}
FUSION_ENABLED = True

# ===================== UTILS =====================
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if mat.ndim == 1:
        n = np.linalg.norm(mat) + eps
        return (mat / n).astype(np.float32)
    n = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return (mat / n).astype(np.float32)

def _fast_load_bytes_bgr(p: Path) -> np.ndarray:
    data = np.fromfile(str(p), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError(f"cv2.imdecode konnte Bild nicht lesen: {p}")
    return img

def preprocess_for_model(img_uint8: np.ndarray, target_size=TARGET_SIZE) -> np.ndarray:
    im = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return im  # (H,W,3) float32 [0..1]

def to_embed_dtype(arr: np.ndarray) -> np.ndarray:
    return arr.astype(np.float32, copy=False)

# ===================== PCA =====================
class PCATransformer:
    def __init__(self, ipca_path: Path):
        if not ipca_path.exists():
            raise FileNotFoundError(f"IPCA-Modell nicht gefunden: {ipca_path}")
        self.ipca = load(ipca_path)
    def transform64(self, vec512_raw: np.ndarray) -> np.ndarray:
        return self.ipca.transform(vec512_raw[None, :]).astype(np.float32)[0]

# ===================== HNSW (32D) =====================
class HNSW32:
    def __init__(self, dim=32):
        self.index = hnswlib.Index(space='ip', dim=dim)  # IP + L2-Norm ~ Cosine
        # labels_meta[i] = (table, id, image_path, emb32_path, emb64_path, emb512_path)
        self.labels_meta: List[Tuple[str,int,str,str,str,str]] = []
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

def fetch_rows_iter(conn: sqlite3.Connection, table: str, batch: int = 20000):
    cur = conn.cursor()
    cur.execute(
        f'SELECT {COL_ID}, {COL_IMAGE_PATH}, {COL_EMB32_PATH}, {COL_EMB64_PATH}, {COL_EMB512_PATH} FROM "{table}"'
    )
    while True:
        rows = cur.fetchmany(batch)
        if not rows: break
        for r in rows:
            yield int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])

def table_has_columns(conn: sqlite3.Connection, table: str, cols: Tuple[str, str]) -> bool:
    cur = conn.cursor()
    cur.execute(f'PRAGMA table_info("{table}")')
    names = {r[1] for r in cur.fetchall()}
    return cols[0] in names and cols[1] in names

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

# ---- Farb-Histogramm (TEXT) laden ----
def _decode_hist_text(txt) -> Optional[np.ndarray]:
    """Parst '7e-06,0.000133,...,0.0' → float32[96] (L1-normalisiert)."""
    if txt is None:
        return None
    if isinstance(txt, (bytes, bytearray, memoryview)):
        txt = txt.decode("utf-8", "ignore")
    if not isinstance(txt, str):
        txt = str(txt)
    txt = txt.strip().lstrip("[").rstrip("]")
    arr = np.fromstring(txt, sep=",", dtype=np.float32)
    if arr.size != COLOR_DIM_EXPECT:
        return None
    s = float(arr.sum())
    return (arr / (s + 1e-9)).astype(np.float32)

def fetch_color_hist_bulk_text(conn: sqlite3.Connection, table: str, ids: List[int], col: str = HIST_COL) -> Dict[int, np.ndarray]:
    if not ids:
        return {}
    qmarks = ",".join("?" for _ in ids)
    sql = f'SELECT {COL_ID}, "{col}" FROM "{table}" WHERE {COL_ID} IN ({qmarks})'
    cur = conn.cursor()
    cur.execute(sql, ids)
    out: Dict[int, np.ndarray] = {}
    for rid, txt in cur.fetchall():
        h = _decode_hist_text(txt)
        if h is not None:
            out[int(rid)] = h
    return out

def hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())

# ===================== Index-Build (32D) =====================
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

    print("[Info] Baue 32D-HNSW aus DB …")
    with sqlite3.connect(str(db_path)) as conn:
        tables = list_feature_tables(conn)
        if not tables:
            raise RuntimeError(f"Keine Tabellen mit Präfix '{TABLE_PREFIX}' gefunden.")
        total = sum(count_rows(conn, t) for t in tables)
        print(f"[Info] Elemente gesamt: {total:,}")
        idx.init_new(max_elements=total)

        seen_total, last_report = 0, 0
        REPORT_STEP = 100_000
        BATCH_ADD   = 20_000

        for t in tables:
            vecs, metas = [], []
            for rid, img_path, emb32_path, emb64_path, emb512_path in fetch_rows_iter(conn, t):
                try:
                    v32 = np.load(emb32_path, mmap_mode='r')
                    v32 = v32[0] if (v32.ndim == 2 and v32.shape[0] == 1) else v32
                    if v32.shape[0] < 32:
                        continue
                    vecs.append(v32[:32].astype(np.float32))
                    metas.append((t, rid, img_path, emb32_path, emb64_path, emb512_path))
                except Exception:
                    continue
                if len(vecs) >= BATCH_ADD:
                    V = l2_normalize_rows(np.vstack(vecs))
                    idx.add_batch(V, metas)
                    seen_total += len(metas)
                    vecs, metas = [], []
                    if seen_total - last_report >= REPORT_STEP:
                        print(f"[Index] added {seen_total:,}/{total:,} …")
                        last_report = seen_total

            if vecs:
                V = l2_normalize_rows(np.vstack(vecs))
                idx.add_batch(V, metas)
                seen_total += len(metas)
                if seen_total - last_report >= REPORT_STEP or seen_total == total:
                    print(f"[Index] added {seen_total:,}/{total:,} …")
                    last_report = seen_total

    idx.save()
    print(f"[Info] Index gespeichert: {INDEX_BIN}")
    return idx

# ===================== Re-Ranking =====================
def rerank_mid_64(q64: np.ndarray, cand_meta, topY: int):
    mats, valid_idx = [], []
    for i, (_, _, _, emb32_path, emb64_path, _) in enumerate(cand_meta):
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
    q = l2_normalize_rows(q64)
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
    sims = M @ l2_normalize_rows(q512)
    order = np.argsort(-sims)[:topM]
    return [(valid_idx[int(i)], float(sims[int(i)])) for i in order]

# ===================== LPIPS/SSIM optional =====================
try:
    import torch, lpips
    _HAS_LPIPS = True
except Exception:
    _HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as ssim
    _HAS_SSIM = True
except Exception:
    _HAS_SSIM = False

from functools import lru_cache

@lru_cache(maxsize=4096)
def _rgb01_hsv(path: str):
    data = np.fromfile(path, dtype=np.uint8)
    bgr  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None: raise ValueError(f"Bild nicht lesbar: {path}")
    bgr  = cv2.resize(bgr, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    rgb  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb01= (rgb.astype(np.float32)/255.0)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return rgb01, hsv

_LPIPS_MODEL = None

def _lpips_dist(a_rgb01, b_rgb01) -> Optional[float]:
    if not (_HAS_LPIPS): return None
    global _LPIPS_MODEL
    try:
        if _LPIPS_MODEL is None:
            dev = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
            _LPIPS_MODEL = lpips.LPIPS(net='alex').eval().to(dev)
        dev = next(_LPIPS_MODEL.parameters()).device
        A = torch.from_numpy(a_rgb01).permute(2,0,1)[None].to(dev)*2-1
        B = torch.from_numpy(b_rgb01).permute(2,0,1)[None].to(dev)*2-1
        with torch.no_grad():
            return float(_LPIPS_MODEL(A,B).item())
    except Exception:
        return None

def _ssim_sim(a_rgb01, b_rgb01) -> Optional[float]:
    if not _HAS_SSIM: return None
    try:
        return float(np.clip(ssim(a_rgb01, b_rgb01, data_range=1.0, channel_axis=2),0,1))
    except Exception:
        return None

def _load_weights_from_json(path: str) -> Optional[Dict[str,float]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        best = data.get("best",{}).get("weights")
        if isinstance(best, dict):
            return {
                "cos":   float(best.get("cos",   FUSION_WEIGHTS_MANUAL["cos"])),
                "lpips": float(best.get("lpips", FUSION_WEIGHTS_MANUAL["lpips"])),
                "ssim":  float(best.get("ssim",  FUSION_WEIGHTS_MANUAL["ssim"])),
                "color": float(best.get("color", FUSION_WEIGHTS_MANUAL["color"])),
            }
    except Exception:
        pass
    return None

FUSION_WEIGHTS = _load_weights_from_json(WEIGHTS_JSON) or FUSION_WEIGHTS_MANUAL
USE_LPIPS = _HAS_LPIPS and FUSION_WEIGHTS.get("lpips", 0.0) > 0
USE_SSIM  = _HAS_SSIM  and FUSION_WEIGHTS.get("ssim",  0.0) > 0

def _fusion_score(cos512, lpips_val, ssim_val, color_val, w: Dict[str,float]) -> float:
    lp = lpips_val if lpips_val is not None else 0.0  # kleiner besser
    ss = ssim_val  if ssim_val  is not None else 0.0
    co = color_val if color_val is not None else 0.0
    return w["cos"]*cos512 - w["lpips"]*lp + w["ssim"]*ss + w["color"]*co

# ===================== SEARCH CORE =====================
def search_image(image_path: Path, db_path: Path) -> Dict[str, Any]:
    # Modelle / Index
    pcat = PCATransformer(IPCA_PATH)
    index32 = build_index_if_needed(db_path)

    # Query-Embedding bauen
    img_bgr   = _fast_load_bytes_bgr(image_path)
    img_small = preprocess_for_model(img_bgr)
    embed_in  = to_embed_dtype(img_small)
    embs      = extract_embeddings([embed_in])              # -> (1,512)
    q512_raw  = np.asarray(embs[0], dtype=np.float32).ravel()
    assert q512_raw.size == 512, f"Unerwartete Query-Embedding-Länge: {q512_raw.shape}"

    # PCA → 64D, 32D
    q64_raw = pcat.transform64(q512_raw)
    q32_raw = q64_raw[:32]

    # L2-Normierungen
    q512 = l2_normalize_rows(q512_raw)
    q64  = l2_normalize_rows(q64_raw)
    q32  = l2_normalize_rows(q32_raw)

    # ---- Coarse (32D) ----
    topX = max(TOP_K * X_FACTOR, TOP_K)
    labels, coarse_sims = index32.knn(q32, topX)
    cand_meta = [index32.labels_meta[int(lbl)] for lbl in labels]
    coarse_map = {i: float(coarse_sims[i]) for i in range(len(labels))}

    # ---- Mid (64D Re-Ranking) ----
    topY = max(TOP_K * Y_FACTOR, TOP_K)
    mid = rerank_mid_64(q64, cand_meta, topY)
    if not mid:
        return {"query":{"image_path":str(image_path)}, "results":{"final_top_k":[]}, "params":{"k":TOP_K}}

    # ---- Fine (512D Re-Ranking) ----
    Y_list = [(cand_idx, cand_meta[cand_idx]) for (cand_idx, _) in mid]
    topM = max(TOP_K * M_FACTOR, TOP_K)
    fine = rerank_fine_512(q512, Y_list, topM)
    if not fine:
        return {"query":{"image_path":str(image_path)}, "results":{"final_top_k":[]}, "params":{"k":TOP_K}}

    # JSON-Listen
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

    # ===== FUSION: cos512 + Farbe aus DB (+ optional SSIM/LPIPS progressiv) =====
    if FUSION_ENABLED:
        # Query-Farbhistogramm mit DEINER Funktion (BGR, 3×bins)
        try:
            q_hist = calc_histogram(img_bgr, bins=COLOR_BINS).astype(np.float32)
            # Sicherheitshalber L1:
            q_hist = q_hist / (float(q_hist.sum()) + 1e-9)
        except Exception:
            q_hist = None

        # Für SSIM/LPIPS brauchen wir die Query auch als rgb01:
        try:
            q_rgb01, _ = _rgb01_hsv(str(image_path))
        except Exception:
            q_rgb01 = None

        # Bulk: alle Kandidaten-Hists aus DB (TEXT) pro Tabelle lesen
        table_to_ids: Dict[str, List[int]] = {}
        for it in fine_top_m:
            table_to_ids.setdefault(it["table"], []).append(it["id"])
        id2hist_per_table: Dict[str, Dict[int, np.ndarray]] = {}
        with sqlite3.connect(str(db_path)) as conn:
            for table, ids in table_to_ids.items():
                id2hist_per_table[table] = fetch_color_hist_bulk_text(conn, table, ids, HIST_COL)

        # Progressive Kandidatenmengen
        T_color = min(len(fine_top_m), 4*TOP_K)
        T_ssim  = min(T_color,       2*TOP_K)
        T_lpips = min(T_ssim,          TOP_K)

        # 1) Farbe (aus DB) – kein Bild-Decoding für Kandidaten
        for it in fine_top_m[:T_color]:
            col_val = 0.0
            if q_hist is not None:
                h = id2hist_per_table.get(it["table"], {}).get(it["id"])
                if h is not None:
                    col_val = hist_intersection(q_hist, h)
            it["sim_color"] = float(col_val)

        # 2) SSIM (teurer) nur für Top nach (cos + color), falls gewichtet
        cands_ssim = sorted(
            fine_top_m[:T_color],
            key=lambda d: (float(d.get("score",0.0)) + float(d.get("sim_color",0.0))),
            reverse=True
        )[:T_ssim]

        if USE_SSIM and q_rgb01 is not None:
            for it in cands_ssim:
                try:
                    c_rgb01, _ = _rgb01_hsv(it["image_path"])
                    it["sim_ssim"] = _ssim_sim(q_rgb01, c_rgb01)
                except Exception:
                    pass

        # 3) LPIPS (am teuersten) nur für Top nach SSIM
        cands_lpips = sorted(
            cands_ssim,
            key=lambda d: float(d.get("sim_ssim", 0.0) or 0.0),
            reverse=True
        )[:T_lpips]

        if USE_LPIPS and q_rgb01 is not None:
            for it in cands_lpips:
                try:
                    c_rgb01, _ = _rgb01_hsv(it["image_path"])
                    it["dist_lpips"] = _lpips_dist(q_rgb01, c_rgb01)
                except Exception:
                    pass

        # finaler Fusionsscore
        fused = []
        for it in fine_top_m:
            cos512   = float(it["score"])
            col_val  = float(it.get("sim_color", 0.0))
            ss_val   = it.get("sim_ssim", None)
            lp_val   = it.get("dist_lpips", None)
            it["score_cos512"] = cos512
            it["score_fused"]  = _fusion_score(cos512, lp_val, ss_val, col_val, FUSION_WEIGHTS)
            fused.append(it)

        final_top_k = sorted(fused, key=lambda d: -float(d["score_fused"]))[:TOP_K]
    else:
        final_top_k = sorted(fine_top_m, key=lambda d: -float(d["score"]))[:TOP_K]

    # Ergebnis
    return {
        "query": {"image_path": str(image_path)},
        "results": {
            "coarse_top_x": [],  # auf Wunsch füllen; aktuell ohne UMAP/Post-Load
            "mid_top_y":   [],   # dito (wir konzentrieren uns auf final_top_k)
            "fine_top_m":  [],   # dito
            "final_top_k": final_top_k
        },
        "params": {
            "k": TOP_K, "X_factor": X_FACTOR, "Y_factor": Y_FACTOR, "M_factor": M_FACTOR,
            "fusion": {"enabled": FUSION_ENABLED, "weights": FUSION_WEIGHTS,
                       "use_lpips": USE_LPIPS, "use_ssim": USE_SSIM,
                       "color_bins": COLOR_BINS, "hist_col": HIST_COL}
        }
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
        "results": {"final_top_k": out["results"]["final_top_k"][:PRINT_TOP_N]},
        "params": out.get("params", {})
    }
    print(json.dumps(to_print, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    sys.exit(main())
