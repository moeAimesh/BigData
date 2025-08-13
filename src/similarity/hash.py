import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import imagehash
# ---------- Helpers ----------
def _parse_hash_value(v) -> Optional[np.uint64]:
    """Akzeptiert int/str/bytes/None. Erkennt HEX (0x.. / a-f), BIN (0/1) oder Dezimal."""
    if v is None:
        return None
    if isinstance(v, (np.integer, int)):
        return np.uint64(int(v))
    if isinstance(v, (bytes, bytearray)):
        return np.uint64(int.from_bytes(v, "big", signed=False))
    if isinstance(v, str):
        s = v.strip().lower()
        if s.startswith("0x") or any(c in s for c in "abcdef"):
            return np.uint64(int(s, 16))
        if set(s) <= {"0","1"} and len(s) >= 32:
            return np.uint64(int(s, 2))
        return np.uint64(int(s))
    return np.uint64(int(v))

def _bit_norm(bits_guess_from_q: int, db_max: int) -> int:
    """Bestimmt eine sinnvolle Normlänge in Bits (64/128/256)."""
    max_bits = max(bits_guess_from_q, int(db_max).bit_length())
    norm_bits = int(np.ceil(max_bits / 8.0) * 8)
    if norm_bits <= 64:  return 64
    if norm_bits <= 128: return 128
    return 256

def _hamming_uint64_vec(db_hashes: np.ndarray, q: np.uint64) -> np.ndarray:
    x = np.bitwise_xor(db_hashes, q)
    b = x.view(np.uint8).reshape(-1, 8)  # 64 Bit -> 8 Bytes
    return np.unpackbits(b, axis=1).sum(axis=1)

def _topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=int)
    k = min(k, arr.shape[0])
    idx = np.argpartition(arr, k-1)[:k]
    return idx[np.argsort(arr[idx])]

def _column_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())

# ---------- Hauptfunktion ----------
def search_by_hash_voting_multitables(
    db_path: str,
    q_ahash: int, q_dhash: int, q_phash: int,
    table_prefix: str = "image_features_part_",
    parts: int = 7,
    weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    topk_per_hash: int = 200,
    final_k: int = 20
) -> List[Dict[str, Any]]:
    """
    Durchsucht image_features_part_1..N und kombiniert aHash/dHash/pHash per Voting.
    Erwartete Spalten: id, image_hash, dhash, phash, [optional: filepath]
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = [f"{table_prefix}{i}" for i in range(1, parts+1)]
    filepath_exists = {t: _column_exists(cur, t, "path") for t in tables}

    all_A, all_D, all_P = [], [], []
    meta_ids, meta_paths, meta_tables = [], [], []

    for t in tables:
        cols = "id, image_hash, dhash, phash"
        if filepath_exists[t]:
            cols += ", path"
        cur.execute(f"SELECT {cols} FROM {t}")
        rows = cur.fetchall()
        if not rows:
            continue

        A, D, P, IDs, PATHS = [], [], [], [], []
        for r in rows:
            iid = r[0]
            ah = _parse_hash_value(r[1])
            dh = _parse_hash_value(r[2])
            ph = _parse_hash_value(r[3])
            if ah is None or dh is None or ph is None:
                continue
            A.append(ah); D.append(dh); P.append(ph); IDs.append(iid)
            PATHS.append(r[4] if filepath_exists[t] else None)

        if not A:
            continue

        all_A.append(np.array(A, dtype=np.uint64))
        all_D.append(np.array(D, dtype=np.uint64))
        all_P.append(np.array(P, dtype=np.uint64))
        meta_ids.append(np.array(IDs, dtype=np.int64))
        meta_paths.append(np.array(PATHS, dtype=object))
        meta_tables.extend([t]*len(IDs))

    con.close()

    if not all_A:
        return []

    A = np.concatenate(all_A)
    D = np.concatenate(all_D)
    P = np.concatenate(all_P)
    IDs = np.concatenate(meta_ids)
    PATHS = np.concatenate(meta_paths)
    TABLES = np.array(meta_tables, dtype=object)

    normA = _bit_norm(int(q_ahash).bit_length(), int(A.max()))
    normD = _bit_norm(int(q_dhash).bit_length(), int(D.max()))
    normP = _bit_norm(int(q_phash).bit_length(), int(P.max()))

    da = _hamming_uint64_vec(A, np.uint64(q_ahash))
    dd = _hamming_uint64_vec(D, np.uint64(q_dhash))
    dp = _hamming_uint64_vec(P, np.uint64(q_phash))

    ia = _topk_indices(da, topk_per_hash)
    id_ = _topk_indices(dd, topk_per_hash)
    ip = _topk_indices(dp, topk_per_hash)
    cand = np.unique(np.concatenate([ia, id_, ip]))
    if cand.size == 0:
        return []

    sa = 1.0 - (da[cand] / float(normA))
    sd = 1.0 - (dd[cand] / float(normD))
    sp = 1.0 - (dp[cand] / float(normP))
    w_a, w_d, w_p = weights
    score = w_a*sa + w_d*sd + w_p*sp

    order = np.argsort(-score)[:final_k]

    out: List[Dict[str, Any]] = []
    for pos in order:
        i = int(cand[pos])
        sc = float(score[pos])  # Score passend zum sortierten Index
        out.append({
            "table": str(TABLES[i]),
            "id": int(IDs[i]),
            "path": (None if PATHS[i] is None else str(PATHS[i])),
            "hamming": {"ahash": int(da[i]), "dhash": int(dd[i]), "phash": int(dp[i])},
            "score": sc
        })
    return out

def compute_query_hashes(path, hash_size=8):
    img = Image.open(path).convert("RGB")
    ah = imagehash.average_hash(img, hash_size=hash_size)  # 8x8 -> 64 Bit
    dh = imagehash.dhash(img,       hash_size=hash_size)
    ph = imagehash.phash(img,       hash_size=hash_size)
    # sichere Integer-Repräsentation (funktioniert immer)
    q_ahash = int(str(ah), 16)
    q_dhash = int(str(dh), 16)
    q_phash = int(str(ph), 16)
    return q_ahash, q_dhash, q_phash

# ------- Beispielaufruf -------
q_ahash, q_dhash, q_phash = compute_query_hashes(r"D:\\data\\image_data\\Fruits_Vegetables\\train\\lettuce\\Image_38.jpg")
results = search_by_hash_voting_multitables(
    db_path=r"C:\BIG_DATA\data\database.db",
    q_ahash=q_ahash, q_dhash=q_dhash, q_phash=q_phash,
    weights=(1.0, 0.9, 1.2),  # z.B. pHash etwas höher
    topk_per_hash=300, final_k=5
)
for r in results:
     print(r)
