import sqlite3, os, numpy as np

# --- Config ---
DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]
HIST_COL = "color_hist"  # ggf. anpassen
OUT_NPZ = r"C:\BIG_DATA\data\avg_rgb_points_from_hist.npz"

BINS = 32  # pro Kanal
DIM = 3 * BINS  # 96
MODE = "avg"  # 'avg' oder 'dominant' (naiv, pro Kanal argmax)

# Bin-Kanten/Zentren gemäß calc_histogram-Range [0,256]
edges = np.linspace(0, 256, BINS + 1, dtype=np.float32)
centers = (edges[:-1] + edges[1:]) * 0.5  # 32 Zentren in [4, 252]


def parse_hist_text(s: str) -> np.ndarray:
    v = np.fromstring(s, sep=",", dtype=np.float32)
    if v.size != DIM:
        return None
    # Sicherheits-L1
    ssum = v.sum()
    if ssum > 0:
        v /= ssum
    return v


def hist_to_rgb(h: np.ndarray, mode: str = "avg") -> np.ndarray:
    """h: (96,) = [B(32), G(32), R(32)] L1-normalisiert -> RGB (0..255)"""
    B = h[0:BINS]
    G = h[BINS : 2 * BINS]
    R = h[2 * BINS : 3 * BINS]
    if mode == "avg":
        # gewichteter Mittelwert der Zentren
        b = float((B * centers).sum())
        g = float((G * centers).sum())
        r = float((R * centers).sum())
    elif mode == "dominant":
        # naiv: pro Kanal die massereichste Bin (Achtung: keine Kanal-Korrelationen)
        b = float(centers[int(np.argmax(B))])
        g = float(centers[int(np.argmax(G))])
        r = float(centers[int(np.argmax(R))])
    else:
        raise ValueError("MODE must be 'avg' or 'dominant'")
    # in [0,255] clampen
    return np.array([r, g, b], dtype=np.float32).clip(0, 255)


def main():
    pos = []  # RGB Positionen 0..255
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for t in TABLES:
            cur.execute(f"PRAGMA table_info({t})")
            cols = {r[1] for r in cur.fetchall()}
            if HIST_COL not in cols:
                continue
            cur.execute(f"SELECT {HIST_COL} FROM {t} WHERE {HIST_COL} IS NOT NULL")
            for (txt,) in cur.fetchall():
                h = parse_hist_text(txt)
                if h is None:
                    continue
                pos.append(hist_to_rgb(h, MODE))

    if not pos:
        raise RuntimeError("Keine Histogrameinträge gefunden.")
    pos = np.vstack(pos).astype(np.float32)  # (N,3) 0..255
    colors = (pos / 255.0).clip(0, 1).astype(np.float32)  # Markerfarbe

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, pos=pos, colors=colors)
    print(f"Gespeichert: {OUT_NPZ}  | Punkte: {pos.shape[0]}  | Modus: {MODE}")


if __name__ == "__main__":
    main()
