import argparse
import json
import sqlite3
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
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
        if set(s) <= {"0", "1"} and len(s) >= 32:
            return np.uint64(int(s, 2))
        return np.uint64(int(s))
    return np.uint64(int(v))

def _bit_norm(bits_guess_from_q: int, db_max: int) -> int:
    """Bestimmt eine sinnvolle Normlänge in Bits (64/128/256). Hinweis: Hamming unten ist 64-bit-optimiert."""
    max_bits = max(bits_guess_from_q, int(db_max).bit_length())
    norm_bits = int(np.ceil(max_bits / 8.0) * 8)
    if norm_bits <= 64:  return 64
    if norm_bits <= 128: return 128
    return 256

def _hamming_uint64_vec(db_hashes: np.ndarray, q: np.uint64) -> np.ndarray:
    """Hamming-Distanz für 64-bit-Hashes (imagehash mit hash_size=8 -> 64 Bit)."""
    x = np.bitwise_xor(db_hashes, q)
    b = x.view(np.uint8).reshape(-1, 8)  # 64 Bit -> 8 Bytes
    return np.unpackbits(b, axis=1).sum(axis=1)

def _topk_indices(arr: np.ndarray, k: int) -> np.ndarray:
    if arr.size == 0:
        return np.array([], dtype=int)
    k = min(k, arr.shape[0])
    idx = np.argpartition(arr, k - 1)[:k]
    return idx[np.argsort(arr[idx])]

def _column_exists(cur: sqlite3.Cursor, table: str, col: str) -> bool:
    cur.execute(f"PRAGMA table_info({table})")
    return any(row[1] == col for row in cur.fetchall())

# ---------- Query-aware Gewichtung ----------
def _softmax(x, temp=6.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    ex = np.exp(x * temp)
    return ex / (ex.sum() + 1e-12)

def _margin_confidence(dists: np.ndarray, norm_bits: int, top_idx: np.ndarray) -> float:
    """Trennschärfe: Mittel der Top-5-Ähnlichkeiten minus Median der Top-k (>=0)."""
    if top_idx.size == 0:
        return 0.0
    s = 1.0 - (dists[top_idx] / float(norm_bits))  # in [0,1]
    q = int(min(5, s.size))
    top_mean = np.partition(s, -q)[-q:].mean()
    med = float(np.median(s))
    return max(0.0, top_mean - med)

def _auto_hash_weights(
    da: np.ndarray, dd: np.ndarray, dp: np.ndarray,
    normA: int, normD: int, normP: int,
    ia: np.ndarray, id_: np.ndarray, ip: np.ndarray,
    temp: float = 6.0
) -> np.ndarray:
    # 1) Trennschärfe je Hash
    ca = _margin_confidence(da, normA, ia)
    cd = _margin_confidence(dd, normD, id_)
    cp = _margin_confidence(dp, normP, ip)

    # 2) Überlappung der Top-Listen als Verstärker
    Sa, Sd, Sp = set(ia.tolist()), set(id_.tolist()), set(ip.tolist())
    k = max(1, len(ia))
    ov_a = (len(Sa & Sd) + len(Sa & Sp)) / (2.0 * k)
    ov_d = (len(Sd & Sa) + len(Sd & Sp)) / (2.0 * k)
    ov_p = (len(Sp & Sa) + len(Sp & Sd)) / (2.0 * k)

    raw = np.array([ca * (1.0 + ov_a), cd * (1.0 + ov_d), cp * (1.0 + ov_p)], dtype=np.float64)
    if raw.max() <= 1e-12:
        return np.array([1/3, 1/3, 1/3], dtype=np.float64)
    return _softmax(raw, temp=temp)

# ---------- Hauptfunktion ----------
def search_by_hash_voting_multitables(
    db_path: str,
    q_ahash: int, q_dhash: int, q_phash: int,
    table_prefix: str = "image_features_part_",
    parts: int = 7,
    weights: Union[Tuple[float, float, float], str] = (1.0, 1.0, 1.0),  # auch "auto" möglich
    topk_per_hash: int = 200,
    final_k: int = 20,
    return_weights: bool = False
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Tuple[float, float, float]]]:
    """
    Durchsucht image_features_part_1..N und kombiniert aHash/dHash/pHash per Voting.
    Erwartete Spalten: id, image_hash, dhash, phash, [optional: path]
    """
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    tables = [f"{table_prefix}{i}" for i in range(1, parts + 1)]
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
        meta_tables.extend([t] * len(IDs))

    con.close()

    if not all_A:
        return ([], (1/3, 1/3, 1/3)) if return_weights else []

    A = np.concatenate(all_A)
    D = np.concatenate(all_D)
    P = np.concatenate(all_P)
    IDs = np.concatenate(meta_ids)
    PATHS = np.concatenate(meta_paths)
    TABLES = np.array(meta_tables, dtype=object)

    # Normierung (64/128/256)
    normA = _bit_norm(int(q_ahash).bit_length(), int(A.max()))
    normD = _bit_norm(int(q_dhash).bit_length(), int(D.max()))
    normP = _bit_norm(int(q_phash).bit_length(), int(P.max()))

    # Hamming-Distanzen
    da = _hamming_uint64_vec(A, np.uint64(q_ahash))
    dd = _hamming_uint64_vec(D, np.uint64(q_dhash))
    dp = _hamming_uint64_vec(P, np.uint64(q_phash))

    # Top-k je Hash
    ia = _topk_indices(da, topk_per_hash)
    id_ = _topk_indices(dd, topk_per_hash)
    ip = _topk_indices(dp, topk_per_hash)

    # Kandidatenmenge
    cand = np.unique(np.concatenate([ia, id_, ip]))
    if cand.size == 0:
        return ([], (1/3, 1/3, 1/3)) if return_weights else []

    # Gewichte wählen
    if isinstance(weights, str) and weights.lower() == "auto":
        w = _auto_hash_weights(da, dd, dp, normA, normD, normP, ia, id_, ip, temp=6.0)
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / (w.sum() + 1e-12)

    # Similarities (größer = besser)
    sa = 1.0 - (da[cand] / float(normA))
    sd = 1.0 - (dd[cand] / float(normD))
    sp = 1.0 - (dp[cand] / float(normP))

    score = w[0] * sa + w[1] * sd + w[2] * sp
    order = np.argsort(-score)[:final_k]

    out: List[Dict[str, Any]] = []
    for pos in order:
        i = int(cand[pos])
        out.append({
            "table": str(TABLES[i]),
            "id": int(IDs[i]),
            "path": (None if PATHS[i] is None else str(PATHS[i])),
            "hamming": {"ahash": int(da[i]), "dhash": int(dd[i]), "phash": int(dp[i])},
            "score": float(score[pos])
        })

    return (out, (float(w[0]), float(w[1]), float(w[2]))) if return_weights else out

# ---------- Query-Hashes berechnen ----------
def compute_query_hashes(path: str, hash_size: int = 8) -> Tuple[int, int, int]:
    """Erzeugt aHash/dHash/pHash (Standard: 64 Bit bei hash_size=8)."""
    img = Image.open(path).convert("RGB")
    ah = imagehash.average_hash(img, hash_size=hash_size)
    dh = imagehash.dhash(img,       hash_size=hash_size)
    ph = imagehash.phash(img,       hash_size=hash_size)
    return int(str(ah), 16), int(str(dh), 16), int(str(ph), 16)

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(
        description="Hash-basierte Bildsuche (aHash/dHash/pHash) über mehrere Tabellen."
    )
    p.add_argument("--db", dest="db_path", required=True, help="Pfad zur SQLite-DB")
    p.add_argument("--QUERY_IMG", dest="query_img", required=True, help="Pfad zum Query-Bild")
    p.add_argument("--final_k", dest="final_k", type=int, default=6, help="Anzahl der finalen Treffer")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Query-Hashes berechnen
    q_ah, q_dh, q_ph = compute_query_hashes(args.query_img)

    # Suche (nur die gewünschten Flags; Rest intern konfiguriert)
    results, w = search_by_hash_voting_multitables(
        db_path=args.db_path,
        q_ahash=q_ah, q_dhash=q_dh, q_phash=q_ph,
        weights="auto",          # dynamische, query-abhängige Gewichte
        topk_per_hash=300,       # intern
        final_k=args.final_k,
        return_weights=True
    )

    print("Gewichte (a,d,p):", tuple(round(x, 4) for x in w))
    print(json.dumps(results, ensure_ascii=False, indent=2))



#python src/similarity/hash.py --db "C:\BIG_DATA\data\database.db" --QUERY_IMG "Z:\CODING\UNI\BIG_DATA\data\TEST_IMAGES\ga-traisen-katze-emma-am-9937-scaled.jpg" --final_k 6
