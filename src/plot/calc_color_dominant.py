# build_rgb_cache_from_hist_bold.py
import sqlite3, os, numpy as np

DB_PATH   = r"C:\BIG_DATA\data\database.db"
TABLES    = [f"image_features_part_{i}" for i in range(1, 8)]
HIST_COL  = "color_hist"                      # ggf. anpassen
OUT_NPZ   = r"C:\BIG_DATA\data\avg_rgb_points_from_hist_bold_mode_maxed.npz"

BINS = 32
DIM  = 3*BINS

# ---- Farbverstärkung nur fürs Rendering ----
SAT_BOOST = 1.7       # >1 macht bunter (z.B. 1.2..1.8)
VAL_BOOST = 1.15      # >1 macht heller (z.B. 1.05..1.2)
GAMMA     = 1.3       # <1 dunkler, >1 heller in sRGB-ähnlichem Sinn (z.B. 1.3)

# ---- Dominanz-Modus ----
MODE  = "soft"        # "mean" | "soft" | "mode"
ALPHA = 3.0           # Softmax-Schärfe (2..5 ist gut)

# ---- Positions-Stretch (optional, für mehr Abstand) ----
STRETCH = 1.2        # 1.0 = aus; >1 dehnt um 128 herum (z.B. 1.2..1.5)

# Bin-Zentren für [0,256] in 32 Bins
edges   = np.linspace(0, 256, BINS+1, dtype=np.float32)
centers = (edges[:-1] + edges[1:]) * 0.5      # 32 Werte ~[4,252]

def parse_hist_text(s: str) -> np.ndarray:
    v = np.fromstring(s, sep=',', dtype=np.float32)
    if v.size != DIM:
        return None
    ssum = v.sum()
    if ssum > 0: v /= ssum  # L1
    return v

def channel_from_hist(hc: np.ndarray, mode: str) -> float:
    """hc: (32,) einer Farbe -> Rückgabe float in 0..255 (Zentrum/Erwartungswert)."""
    if mode == "mean":
        return float((hc * centers).sum())
    if mode == "mode":
        return float(centers[int(np.argmax(hc))])
    if mode == "soft":
        w = np.power(hc + 1e-12, ALPHA)
        w /= w.sum()
        return float((w * centers).sum())
    raise ValueError

def hist_to_rgb(h: np.ndarray, mode: str = MODE) -> np.ndarray:
    B = h[0:BINS]; G = h[BINS:2*BINS]; R = h[2*BINS:3*BINS]
    b = channel_from_hist(B, mode)
    g = channel_from_hist(G, mode)
    r = channel_from_hist(R, mode)
    return np.array([r, g, b], dtype=np.float32).clip(0, 255)

def rgb_to_hsv01(rgb01):
    r,g,b = rgb01[...,0], rgb01[...,1], rgb01[...,2]
    cmax, cmin = np.max(rgb01, axis=-1), np.min(rgb01, axis=-1)
    delta = cmax - cmin + 1e-12
    # Hue
    h = np.zeros_like(cmax)
    mask = (cmax == r)
    h[mask] = ((g-b)[mask]/delta[mask]) % 6
    mask = (cmax == g)
    h[mask] = ((b-r)[mask]/delta[mask]) + 2
    mask = (cmax == b)
    h[mask] = ((r-g)[mask]/delta[mask]) + 4
    h = (h/6.0) % 1.0
    # Sat & Val
    s = delta / (cmax + 1e-12)
    v = cmax
    return np.stack([h,s,v], axis=-1)

def hsv01_to_rgb01(hsv):
    h,s,v = hsv[...,0], hsv[...,1], hsv[...,2]
    h6 = h*6.0
    i  = np.floor(h6).astype(int)
    f  = h6 - i
    p  = v*(1-s)
    q  = v*(1-s*f)
    t  = v*(1-s*(1-f))
    i_mod = i % 6
    out = np.zeros(hsv.shape, dtype=np.float32)
    lut = {
        0: (v,t,p), 1: (q,v,p), 2: (p,v,t),
        3: (p,q,v), 4: (t,p,v), 5: (v,p,q),
    }
    for k in range(6):
        m = (i_mod == k)
        out[m,0], out[m,1], out[m,2] = [arr[m] for arr in lut[k]]
    return out

def apply_visual_boost(rgb255: np.ndarray) -> np.ndarray:
    """Nur für Marker-Farbe: S/V boosten + Gamma-Aufhellung."""
    x = (rgb255 / 255.0).clip(0,1).astype(np.float32)
    hsv = rgb_to_hsv01(x)
    hsv[:,1] = np.clip(hsv[:,1] * SAT_BOOST, 0, 1)
    hsv[:,2] = np.clip(hsv[:,2] * VAL_BOOST, 0, 1)
    y = hsv01_to_rgb01(hsv)
    if GAMMA != 1.0:
        y = np.clip(y, 0, 1) ** (1.0 / GAMMA)  # GAMMA>1 -> heller
    return y

def stretch_positions(pos255: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return pos255
    c = 128.0
    return np.clip(c + factor*(pos255 - c), 0, 255)

def main():
    pos = []
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for t in TABLES:
            cur.execute(f"PRAGMA table_info({t})")
            cols = {r[1] for r in cur.fetchall()}
            if HIST_COL not in cols:
                continue
            for (txt,) in cur.execute(f"SELECT {HIST_COL} FROM {t} WHERE {HIST_COL} IS NOT NULL"):
                h = parse_hist_text(txt)
                if h is None: continue
                pos.append(hist_to_rgb(h, MODE))
    if not pos:
        raise RuntimeError("Keine Histogramme gefunden.")

    pos = np.vstack(pos).astype(np.float32)  # (N,3) 0..255

    # (optional) Positionen spreizen
    pos_stretched = stretch_positions(pos, STRETCH)

    # Marker-Farben boosten (HSV + Gamma)
    colors = apply_visual_boost(pos).astype(np.float32)  # 0..1

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(OUT_NPZ, pos=pos_stretched, colors=colors)
    print(f"Gespeichert: {OUT_NPZ}  | Punkte: {pos.shape[0]}  | mode={MODE}, alpha={ALPHA}, stretch={STRETCH}")

if __name__ == "__main__":
    main()
