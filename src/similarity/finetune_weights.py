"""
finetune_weights.py

Gewichtetuning für das finale Re-Ranking (cos512, LPIPS, SSIM, HSV-Color) mittels Precision@k.

Modi:
  (A) --queries <JSONL> : Supervised Eval ({"query":"...", "positives":[...]})
  (B) --queries_dir <DIR>: Weak-Label Auto-Mining (2-von-3-Regel)
  (C) --queries_from_db N: N zufällige DB-Bilder als Queries (Weak-Label Auto-Mining)

NEU:
- --index_dir: vorhandenen 32D-HNSW-Index (hnsw_32d.bin + labels_meta.pkl) laden/erstellen
- Fortschritts-Logs beim Index-Build
- Rekursive Query-Suche in --queries_dir
- Self-Match-Filter (Query ≠ Kandidat)
"""

import os, sys, json, argparse, sqlite3, pickle, random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import cv2, hnswlib
from joblib import load

# ===== optionale Metriken =====
try:
    import torch, lpips

    _HAS_LPIPS_BASE = True
except Exception:
    _HAS_LPIPS_BASE = False

try:
    from skimage.metrics import structural_similarity as ssim

    _HAS_SSIM_BASE = True
except Exception:
    _HAS_SSIM_BASE = False

# ===== Embedding =====
try:
    from embedding_vec import extract_embeddings
except ImportError:
    raise SystemExit(
        "Fehlt: embedding_vec.extract_embeddings (wie in Pipeline)."
    )

# ===== DB-Schema =====
TABLE_PREFIX = "image_features_part_"
COL_ID = "id"
COL_IMAGE_PATH = "path"
COL_EMB32_PATH = "pca_32"
COL_EMB64_PATH = "pca_embedding"
COL_EMB512_PATH = "embedding_path"
TARGET_SIZE = (224, 224)

# ===== HNSW-Param =====
HNSW_M = 16
HNSW_EF_CONSTRUCT = 200
HNSW_EF_SEARCH = 400

# ===== Mining-Schwellen (Weak-Labels) =====
COS_MIN = 0.70
COLOR_MIN = 0.55
SSIM_MIN = 0.45
LPIPS_MAX = 0.35
VOTES_MIN = 2

# ===== Kandidatenumfänge =====
DEFAULT_X = 50
DEFAULT_Y = 10
DEFAULT_M = 4


# ===== Utils =====
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def l2n(v: np.ndarray, eps=1e-12):
    n = np.linalg.norm(v) + eps
    return (v / n).astype(np.float32)


def _normp(p: str) -> str:
    return os.path.normcase(os.path.normpath(p))


def _read_jsonl(path: Path, limit: Optional[int]) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def _load_uint8_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Kann Bild nicht lesen: {path}")
    return img


def _preprocess_for_model(img_bgr: np.ndarray) -> np.ndarray:
    return (
        cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )
        / 255.0
    )


def _query512(img_path: Path) -> np.ndarray:
    bgr = _load_uint8_bgr(img_path)
    x = _preprocess_for_model(bgr)
    e = extract_embeddings([x])[0]
    return np.asarray(e, dtype=np.float32).ravel()


def _load_rgb01_and_hsv_u8(p: Path, size=(224, 224)):
    data = np.fromfile(str(p), dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Bild nicht lesbar: {p}")
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb01 = rgb.astype(np.float32) / 255.0
    hsv_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return rgb01, hsv_u8


def _hsv_hist_intersection(
    hsv_a: np.ndarray, hsv_b: np.ndarray, h_bins=32, s_bins=32, v_bins=32
) -> float:
    h1 = cv2.calcHist([hsv_a], [0], None, [h_bins], [0, 180]).ravel()
    h2 = cv2.calcHist([hsv_b], [0], None, [h_bins], [0, 180]).ravel()
    s1 = cv2.calcHist([hsv_a], [1], None, [s_bins], [0, 256]).ravel()
    s2 = cv2.calcHist([hsv_b], [1], None, [s_bins], [0, 256]).ravel()
    v1 = cv2.calcHist([hsv_a], [2], None, [v_bins], [0, 256]).ravel()
    v2 = cv2.calcHist([hsv_b], [2], None, [v_bins], [0, 256]).ravel()

    def _l1n(x):
        s = x.sum()
        return x / (s + 1e-9)

    h1, h2, s1, s2, v1, v2 = map(_l1n, (h1, h2, s1, s2, v1, v2))
    inter = lambda a, b: float(np.minimum(a, b).sum())
    return (inter(h1, h2) + inter(s1, s2) + inter(v1, v2)) / 3.0


# ===== PCA & Index =====
class PCATransformer:
    def __init__(self, ipca_path: Path):
        if not Path(ipca_path).exists():
            raise SystemExit(f"IPCA-Modell nicht gefunden: {ipca_path}")
        self.ipca = load(ipca_path)

    def transform64(self, vec512: np.ndarray) -> np.ndarray:
        return self.ipca.transform(vec512[None, :]).astype(np.float32)[0]


class HNSW32:
    def __init__(self, dim=32):
        self.index = hnswlib.Index(space="ip", dim=dim)
        self.meta: List[Tuple[str, int, str, str, str, str]] = []

    def init(self, max_elements: int):
        self.index.init_index(
            max_elements=max_elements, ef_construction=HNSW_EF_CONSTRUCT, M=HNSW_M
        )
        self.index.set_ef(HNSW_EF_SEARCH)

    def add(self, vecs32: np.ndarray, meta):
        start = len(self.meta)
        labels = np.arange(start, start + vecs32.shape[0])
        self.index.add_items(vecs32, labels)
        self.meta.extend(meta)

    def knn(self, q32: np.ndarray, k: int):
        self.index.set_ef(max(HNSW_EF_SEARCH, k))
        l, s = self.index.knn_query(q32[None, :], k=k)
        return l[0], s[0].astype(np.float32)


# ---- DB-Helpers
def list_tables(conn) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ? ORDER BY name ASC",
        (f"{TABLE_PREFIX}%",),
    )
    return [r[0] for r in cur.fetchall()]


def count_rows(conn, t) -> int:
    cur = conn.cursor()
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    return int(cur.fetchone()[0])


def fetch_rows(conn, t, batch=20000):
    cur = conn.cursor()
    cur.execute(
        f'SELECT {COL_ID},{COL_IMAGE_PATH},{COL_EMB32_PATH},{COL_EMB64_PATH},{COL_EMB512_PATH} FROM "{t}"'
    )
    while True:
        rows = cur.fetchmany(batch)
        if not rows:
            break
        for r in rows:
            yield int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4])


# ---- Build (Progress) + Save
def build_index32(db_path: Path, save_to: Optional[Path] = None) -> HNSW32:
    idx = HNSW32(dim=32)
    with sqlite3.connect(str(db_path)) as conn:
        tabs = list_tables(conn)
        if not tabs:
            raise SystemExit(f"Keine Tabellen mit Präfix '{TABLE_PREFIX}' gefunden.")
        total = sum(count_rows(conn, t) for t in tabs)
        print(f"[Index] Building 32D HNSW from DB … total elements: {total:,}")
        idx.init(total)

        seen_total = 0
        last_report = 0
        REPORT_STEP = 100_000
        BATCH_ADD = 20_000

        for t in tabs:
            vecs = []
            metas = []
            for rid, img, p32, p64, p512 in fetch_rows(conn, t):
                try:
                    v32 = np.load(p32, mmap_mode="r")
                    v32 = v32[0] if (v32.ndim == 2 and v32.shape[0] == 1) else v32
                    if v32.shape[0] < 32:
                        continue
                    vecs.append(v32[:32].astype(np.float32))
                    metas.append((t, rid, img, p32, p64, p512))
                except Exception:
                    continue

                if len(vecs) >= BATCH_ADD:
                    V = np.vstack([l2n(x) for x in vecs])
                    idx.add(V, metas)
                    seen_total += len(metas)
                    vecs, metas = [], []
                    if seen_total - last_report >= REPORT_STEP:
                        print(f"[Index] added {seen_total:,}/{total:,} …")
                        last_report = seen_total

            if vecs:
                V = np.vstack([l2n(x) for x in vecs])
                idx.add(V, metas)
                seen_total += len(metas)
                if seen_total - last_report >= REPORT_STEP or seen_total == total:
                    print(f"[Index] added {seen_total:,}/{total:,} …")
                    last_report = seen_total

    print("[Index] Build complete.")
    if save_to:
        save_to.mkdir(parents=True, exist_ok=True)
        binp = save_to / "hnsw_32d.bin"
        metp = save_to / "labels_meta.pkl"
        idx.index.save_index(str(binp))
        with open(metp, "wb") as f:
            pickle.dump(idx.meta, f)
        print(f"[Index] Saved to: {binp} and {metp}")
    return idx


# ---- Load-or-Build
def build_or_load_index32(db_path: Path, index_dir: Optional[Path]) -> HNSW32:
    if index_dir:
        binp = index_dir / "hnsw_32d.bin"
        metp = index_dir / "labels_meta.pkl"
        if binp.exists() and metp.exists():
            idx = HNSW32(dim=32)
            idx.index.load_index(str(binp))
            idx.index.set_ef(HNSW_EF_SEARCH)
            with open(metp, "rb") as f:
                idx.meta = pickle.load(f)
            print(f"[Index] Loaded existing index from: {binp}")
            return idx
        else:
            print(f"[Index] No existing index in {index_dir} → building new one …")
            return build_index32(db_path, save_to=index_dir)
    return build_index32(db_path, save_to=None)


# ===== 32→64→512 Kandidaten =====
def rerank64(q64: np.ndarray, cand_meta, topY: int):
    mats = []
    idxs = []
    for i, (_, _, _, _, p64, _) in enumerate(cand_meta):
        try:
            v = np.load(p64, mmap_mode="r")
            v = v[0] if (v.ndim == 2 and v.shape[0] == 1) else v
            v = v.astype(np.float32).ravel()[:64]
            mats.append(v)
            idxs.append(i)
        except Exception:
            continue
    if not mats:
        return []
    M = np.vstack([l2n(v) for v in mats])
    q = l2n(q64)
    sims = M @ q
    order = np.argsort(-sims)[:topY]
    return [(idxs[int(i)], float(sims[int(i)])) for i in order]


def rerank512(q512: np.ndarray, cand_meta_pairs, topM: int):
    mats = []
    idxs = []
    for cand_idx, meta in cand_meta_pairs:
        _, _, _, _, _, p512 = meta
        try:
            v = np.load(p512, mmap_mode="r")
            v = v[0] if (v.ndim == 2 and v.shape[0] == 1) else v
            mats.append(v.astype(np.float32).ravel())
            idxs.append(cand_idx)
        except Exception:
            continue
    if not mats:
        return []
    M = np.vstack([l2n(v) for v in mats])
    q = l2n(q512)
    sims = M @ q
    order = np.argsort(-sims)[:topM]
    return [(idxs[int(i)], float(sims[int(i)])) for i in order]


def get_fine_list_for_query(
    qpath: Path, pca: PCATransformer, idx32: HNSW32, X: int, Y: int, M: int, k: int
) -> List[Dict[str, Any]]:
    q512 = _query512(qpath)
    q64 = pca.transform64(q512)
    q32 = q64[:32]
    q512n, q64n, q32n = l2n(q512), l2n(q64), l2n(q32)

    labels, _ = idx32.knn(q32n, max(k * X, k))
    cand_meta = [idx32.meta[int(l)] for l in labels]

    mid = rerank64(q64n, cand_meta, max(k * Y, k))
    if not mid:
        return []
    Y_pairs = [(i, cand_meta[i]) for (i, _) in mid]

    fine = rerank512(q512n, Y_pairs, max(k * M, k))
    if not fine:
        return []

    qnorm = _normp(str(qpath))
    out = []
    for cand_idx, cos in fine:
        table, rid, img, p32, p64, p512 = cand_meta[cand_idx]
        # Self-Match ausschließen
        if _normp(img) == qnorm:
            continue
        out.append({"image_path": img, "path_512": p512, "score": float(cos)})
    return out


# ===== Zusatzmetriken & Fusion =====
_LPIPS_MODEL = None


def _lpips_dist(q_rgb01: np.ndarray, c_rgb01: np.ndarray) -> Optional[float]:
    if not _HAS_LPIPS:
        return None
    global _LPIPS_MODEL
    try:
        if _LPIPS_MODEL is None:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            _LPIPS_MODEL = lpips.LPIPS(net="alex").eval().to(dev)
        dev = next(_LPIPS_MODEL.parameters()).device
        a = torch.from_numpy(q_rgb01).permute(2, 0, 1)[None].to(dev=dev) * 2 - 1
        b = torch.from_numpy(c_rgb01).permute(2, 0, 1)[None].to(dev=dev) * 2 - 1
        with torch.no_grad():
            d = _LPIPS_MODEL(a, b).item()
        return float(d)
    except Exception:
        return None


def _ssim_sim(q_rgb01: np.ndarray, c_rgb01: np.ndarray) -> Optional[float]:
    if not _HAS_SSIM:
        return None
    try:
        val = ssim(q_rgb01, c_rgb01, data_range=1.0, channel_axis=2)
        return float(np.clip(val, 0, 1))
    except Exception:
        return None


def fusion_score(
    cos512: float,
    lpips_val: Optional[float],
    ssim_val: Optional[float],
    color_val: Optional[float],
    w_cos: float,
    w_lpips: float,
    w_ssim: float,
    w_color: float,
) -> float:
    lp = lpips_val if lpips_val is not None else 0.0
    ss = ssim_val if ssim_val is not None else 0.0
    co = color_val if color_val is not None else 0.0
    return w_cos * cos512 - w_lpips * lp + w_ssim * ss + w_color * co


def precision_at_k(pred_paths: List[str], pos_set: set, k: int) -> float:
    hits = sum(1 for p in pred_paths[:k] if _normp(p) in pos_set)
    return hits / float(k)


# ===== Weak-Label Mining (2-von-3) =====
def mine_positives_for_query(
    qpath: Path,
    pca: PCATransformer,
    idx32: HNSW32,
    X: int,
    Y: int,
    M: int,
    k: int,
    use_rnn: bool = False,
    rnn_top: int = 20,
) -> List[str]:
    fine_list = get_fine_list_for_query(qpath, pca, idx32, X, Y, M, k)
    if not fine_list:
        return []

    try:
        q_rgb01, q_hsv = _load_rgb01_and_hsv_u8(qpath)
    except Exception:
        q_rgb01 = q_hsv = None

    positives = []
    for it in fine_list:
        img = it["image_path"]
        cos = float(it["score"])
        votes = 0

        if cos >= COS_MIN:
            votes += 1

        if q_rgb01 is not None:
            try:
                c_rgb01, c_hsv = _load_rgb01_and_hsv_u8(Path(img))
                col = _hsv_hist_intersection(q_hsv, c_hsv)
                if col >= COLOR_MIN:
                    votes += 1
                ok_perc = False
                if _HAS_SSIM:
                    ok_perc = (_ssim_sim(q_rgb01, c_rgb01) or 0.0) >= SSIM_MIN
                if not ok_perc and _HAS_LPIPS:
                    d = _lpips_dist(q_rgb01, c_rgb01)
                    ok_perc = d is not None and d <= LPIPS_MAX
                if ok_perc:
                    votes += 1
            except Exception:
                pass

        if votes >= VOTES_MIN:
            positives.append(img)

    if use_rnn and positives:
        keep = []
        for p in positives:
            try:
                rev = get_fine_list_for_query(
                    Path(p), pca, idx32, X=5, Y=3, M=2, k=rnn_top
                )
                rev_paths = [x["image_path"] for x in rev]
                if str(qpath) in rev_paths:
                    keep.append(p)
            except Exception:
                keep.append(p)
        positives = keep

    return positives[:k]


# ===== Grid =====
def grid(
    weights_grid: Dict[str, List[float]],
) -> List[Tuple[float, float, float, float]]:
    wc = weights_grid.get("cos", [0.6, 0.7, 0.8])
    wl = weights_grid.get("lpips", [0.1, 0.2, 0.3])
    ws = weights_grid.get("ssim", [0.0, 0.1, 0.2])
    wh = weights_grid.get("color", [0.0, 0.1, 0.2])
    combos = []
    for a in wc:
        for b in wl:
            for c in ws:
                for d in wh:
                    bb = b if _HAS_LPIPS_BASE else 0.0
                    cc = c if _HAS_SSIM_BASE else 0.0
                    combos.append((a, bb, cc, d))
    return combos


# ===== Queries aus DB ziehen =====
def sample_queries_from_db(db_path: Path, n: int, ext_filter=True) -> List[Path]:
    paths: List[Path] = []
    with sqlite3.connect(str(db_path)) as conn:
        tabs = list_tables(conn)
        if not tabs:
            return []
        # fair über Tabellen verteilen
        per_tab = max(1, n // len(tabs))
        remain = n
        for t in tabs:
            want = min(per_tab, remain)
            if want <= 0:
                break
            # Für kleine n ist RANDOM() ok (einmalige Eval)
            cur = conn.cursor()
            cur.execute(
                f'SELECT {COL_IMAGE_PATH} FROM "{t}" ORDER BY RANDOM() LIMIT ?', (want,)
            )
            for (p,) in cur.fetchall():
                paths.append(Path(p))
            remain -= want
        # evtl. auffüllen, falls durch Rundung weniger wurden
        if len(paths) < n:
            extra = n - len(paths)
            t0 = tabs[0]
            cur = conn.cursor()
            cur.execute(
                f'SELECT {COL_IMAGE_PATH} FROM "{t0}" ORDER BY RANDOM() LIMIT ?',
                (extra,),
            )
            for (p,) in cur.fetchall():
                paths.append(Path(p))
    # optional: nur existierende Bilddateien
    if ext_filter:
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        paths = [p for p in paths if p.suffix.lower() in allowed and p.is_file()]
    return paths[:n]


# ===== Haupt-Eval =====
def run_eval(
    db_path: Path,
    ipca_path: Path,
    idx_dir: Optional[Path],
    queries_jsonl: Optional[Path],
    queries_dir: Optional[Path],
    queries_from_db: Optional[int],
    out_json: Path,
    out_mined_jsonl: Optional[Path],
    k: int,
    maxq: int,
    X: int,
    Y: int,
    M: int,
    weights_grid: Dict[str, List[float]],
    disable_lpips: bool,
    disable_ssim: bool,
    use_rnn: bool,
) -> Dict[str, Any]:

    global _HAS_LPIPS, _HAS_SSIM
    _HAS_LPIPS = _HAS_LPIPS_BASE and (not disable_lpips)
    _HAS_SSIM = _HAS_SSIM_BASE and (not disable_ssim)

    pca = PCATransformer(ipca_path)
    idx32 = build_or_load_index32(db_path, idx_dir)

    # ---- Queries vorbereiten
    eval_rows: List[Dict[str, Any]] = []

    if queries_jsonl:
        rows = _read_jsonl(queries_jsonl, maxq)
        if not rows:
            raise SystemExit("Keine Zeilen in --queries gefunden.")
        eval_rows = rows

    elif queries_from_db:
        q_files = sample_queries_from_db(db_path, n=min(maxq, queries_from_db))
        if not q_files:
            raise SystemExit("Konnte keine Query-Bilder aus der DB ziehen.")
        mined = []
        for q in q_files:
            pos = mine_positives_for_query(
                q, pca, idx32, X, Y, M, k, use_rnn=use_rnn, rnn_top=20
            )
            if pos:
                mined.append({"query": str(q), "positives": pos})
                print(f"[MINE/DB] {q} -> {len(pos)} positives")
            else:
                print(f"[MINE/DB] {q} -> 0 positives")
        if not mined:
            raise SystemExit(
                "Auto-Mining (DB) fand keine Positiven – Schwellen senken oder X/Y/M erhöhen."
            )
        eval_rows = mined
        if out_mined_jsonl:
            with open(out_mined_jsonl, "w", encoding="utf-8") as f:
                for r in mined:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[MINE/DB] Weak-Labels gespeichert: {out_mined_jsonl}")

    else:
        # queries_dir (rekursiv)
        allowed = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        q_files = [
            p
            for p in Path(queries_dir).rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        ]
        random.shuffle(q_files)
        q_files = q_files[:maxq]
        if not q_files:
            raise SystemExit(
                "Keine Query-Bilder in --queries_dir gefunden (auch rekursiv nicht)."
            )

        mined = []
        for q in q_files:
            pos = mine_positives_for_query(
                q, pca, idx32, X, Y, M, k, use_rnn=use_rnn, rnn_top=20
            )
            if pos:
                mined.append({"query": str(q), "positives": pos})
                print(f"[MINE] {q} -> {len(pos)} positives")
            else:
                print(f"[MINE] {q} -> 0 positives")
        if not mined:
            raise SystemExit(
                "Auto-Mining fand keine Positiven – Schwellen senken oder X/Y/M erhöhen."
            )
        eval_rows = mined
        if out_mined_jsonl:
            with open(out_mined_jsonl, "w", encoding="utf-8") as f:
                for r in mined:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[MINE] Weak-Labels gespeichert: {out_mined_jsonl}")

    # ---- Fine-Kandidaten cachen
    per_query_data = []
    for row in eval_rows:
        qpath = Path(row["query"])
        pos_set = set(_normp(p) for p in row.get("positives", []))
        fine_list = get_fine_list_for_query(qpath, pca, idx32, X, Y, M, k)
        per_query_data.append((str(qpath), pos_set, fine_list))

    # ---- Bildfeature-Cache
    img_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def get_img_feats(p: str):
        if p not in img_cache:
            rgb01, hsv = _load_rgb01_and_hsv_u8(Path(p))
            img_cache[p] = (rgb01, hsv)
        return img_cache[p]

    # ---- Grid-Search
    combos = grid(weights_grid)
    results = []
    for w_cos, w_lp, w_ss, w_col in combos:
        pks = []
        for qpath, pos, fine_list in per_query_data:
            if not fine_list:
                pks.append(0.0)
                continue
            try:
                q_rgb01, q_hsv = get_img_feats(qpath)
            except Exception:
                sorted_paths = [
                    it["image_path"]
                    for it in sorted(fine_list, key=lambda d: -d["score"])
                ]
                pks.append(precision_at_k(sorted_paths, pos, k))
                continue

            scored = []
            for it in fine_list:
                cand = it["image_path"]
                cos512 = it["score"]
                lp_val = ss_val = None
                col_val = 0.0
                try:
                    c_rgb01, c_hsv = get_img_feats(cand)
                    if _HAS_LPIPS:
                        lp_val = _lpips_dist(q_rgb01, c_rgb01)
                    if _HAS_SSIM:
                        ss_val = _ssim_sim(q_rgb01, c_rgb01)
                    col_val = _hsv_hist_intersection(q_hsv, c_hsv)
                except Exception:
                    pass
                final = fusion_score(
                    cos512, lp_val, ss_val, col_val, w_cos, w_lp, w_ss, w_col
                )
                scored.append((final, cand))
            scored.sort(key=lambda x: -x[0])
            pred_paths = [c for _, c in scored[:k]]
            pks.append(precision_at_k(pred_paths, pos, k))
        mean_pk = float(np.mean(pks)) if pks else 0.0
        results.append(
            {
                "weights": {"cos": w_cos, "lpips": w_lp, "ssim": w_ss, "color": w_col},
                "P@k": mean_pk,
            }
        )

    results.sort(key=lambda r: -r["P@k"])
    out = {
        "k": k,
        "mode": (
            "supervised"
            if queries_jsonl
            else ("db" if queries_from_db else "weak_labels")
        ),
        "num_queries": len(per_query_data),
        "results": results,
        "best": results[0] if results else None,
        "params": {"X": X, "Y": Y, "M": M},
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return out


# ===== CLI =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, type=str)
    ap.add_argument("--ipca", required=True, type=str)

    # EINE von drei:
    ap.add_argument(
        "--queries", type=str, default=None, help="JSONL mit {query, positives}"
    )
    ap.add_argument(
        "--queries_dir",
        type=str,
        default=None,
        help="Ordner mit Query-Bildern (rekursiv)",
    )
    ap.add_argument(
        "--queries_from_db",
        type=int,
        default=None,
        help="N zufällige Bilder direkt aus der DB als Queries",
    )

    # Index:
    ap.add_argument(
        "--index_dir",
        type=str,
        default=None,
        help="Ordner mit hnsw_32d.bin + labels_meta.pkl",
    )

    # Output:
    ap.add_argument("--out", type=str, default="eval_weight_grid.json")
    ap.add_argument("--out_mined", type=str, default=None)

    # k & Limits:
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--maxq", type=int, default=20)

    # Kandidatenumfänge:
    ap.add_argument("--x", type=int, default=DEFAULT_X)
    ap.add_argument("--y", type=int, default=DEFAULT_Y)
    ap.add_argument("--m", type=int, default=DEFAULT_M)

    # Grid:
    ap.add_argument("--cos", type=str, default="0.5,0.6,0.7,0.8")
    ap.add_argument("--lpips", type=str, default="0.1,0.2,0.3")
    ap.add_argument("--ssim", type=str, default="0.0,0.1,0.2")
    ap.add_argument("--color", type=str, default="0.0,0.1,0.2")

    # Heavy metrics toggles:
    ap.add_argument("--no_lpips", action="store_true", help="LPIPS deaktivieren")
    ap.add_argument("--no_ssim", action="store_true", help="SSIM deaktivieren")

    # Optional RNN fürs Mining:
    ap.add_argument("--use_rnn", action="store_true")

    args = ap.parse_args()

    # Modus-Prüfung
    modes = sum(bool(x) for x in [args.queries, args.queries_dir, args.queries_from_db])
    if modes != 1:
        raise SystemExit(
            "Gib GENAU EINES an: --queries ODER --queries_dir ODER --queries_from_db N"
        )

    wgrid = {
        "cos": [float(x) for x in args.cos.split(",") if x],
        "lpips": [float(x) for x in args.lpips.split(",") if x],
        "ssim": [float(x) for x in args.ssim.split(",") if x],
        "color": [float(x) for x in args.color.split(",") if x],
    }

    out = run_eval(
        db_path=Path(args.db),
        ipca_path=Path(args.ipca),
        idx_dir=Path(args.index_dir) if args.index_dir else None,
        queries_jsonl=Path(args.queries) if args.queries else None,
        queries_dir=Path(args.queries_dir) if args.queries_dir else None,
        queries_from_db=args.queries_from_db,
        out_json=Path(args.out),
        out_mined_jsonl=(
            Path(args.out_mined)
            if args.out_mined
            else (
                Path(args.out).with_suffix(".mined.jsonl")
                if args.queries_dir or args.queries_from_db
                else None
            )
        ),
        k=args.k,
        maxq=args.maxq,
        X=args.x,
        Y=args.y,
        M=args.m,
        weights_grid=wgrid,
        disable_lpips=args.no_lpips,
        disable_ssim=args.no_ssim,
        use_rnn=args.use_rnn,
    )

    print("\nBest Weights & Precision@k")
    print(json.dumps(out.get("best"), ensure_ascii=False, indent=2))
    print("\nTop 10 (by P@k):")
    for r in out["results"][:10]:
        print(r)
    print(f"\nGespeichert: {args.out}")
    if args.queries_dir or args.queries_from_db:
        print(f"Weak-Labels: {Path(args.out).with_suffix('.mined.jsonl')}")


if __name__ == "__main__":
    sys.exit(main())
