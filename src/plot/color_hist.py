import sqlite3, os, numpy as np
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]
THUMB = 128
CENTER_CROP = 0.7
N_THREADS = 32

# ---- Sampling-Parameter ----
SAMPLE_N = 10000  # << feste Anzahl (None = aus)
SAMPLE_FRAC = None  # << Anteil z.B. 0.1 (=10%) (None = aus)
EVERY_K = None  # << jedes k-te Bild nehmen (None = aus)
RNG_SEED = 42  # Reproduzierbarkeit

POINT_SIZE = 3.0
ALPHA = 0.9


def collect_paths():
    paths = []
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        for t in TABLES:
            cur.execute(f"PRAGMA table_info({t})")
            cols = {r[1] for r in cur.fetchall()}
            if "path" in cols or "filepath" in cols:
                col = "path" if "path" in cols else "filepath"
                cur.execute(f"SELECT {col} FROM {t} WHERE {col} IS NOT NULL")
                paths += [row[0] for row in cur.fetchall()]
    return paths


def avg_rgb_from_path(p: str):
    try:
        if not p or not os.path.exists(p):
            return None
        img = cv.imread(p, cv.IMREAD_COLOR)  # BGR
        if img is None:
            return None
        if THUMB:
            h0, w0 = img.shape[:2]
            if max(h0, w0) > THUMB:
                s = THUMB / max(h0, w0)
                img = cv.resize(
                    img, (int(w0 * s), int(h0 * s)), interpolation=cv.INTER_AREA
                )
        if CENTER_CROP and 0 < CENTER_CROP < 1:
            h, w = img.shape[:2]
            k = int(min(h, w) * CENTER_CROP)
            y0 = (h - k) // 2
            x0 = (w - k) // 2
            img = img[y0 : y0 + k, x0 : x0 + k]
        mean_bgr = img.reshape(-1, 3).mean(axis=0)
        r, g, b = mean_bgr[2], mean_bgr[1], mean_bgr[0]
        return np.float32([r, g, b])
    except Exception:
        return None


# --- 1) Pfade holen + SAMPLING anwenden ---
paths_all = collect_paths()
print("Gefundene Pfade:", len(paths_all))

paths = paths_all
rng = np.random.default_rng(RNG_SEED)

if SAMPLE_N is not None:
    k = min(SAMPLE_N, len(paths_all))
    idx = rng.choice(len(paths_all), size=k, replace=False)
    paths = [paths_all[i] for i in idx]
elif SAMPLE_FRAC is not None:
    k = max(1, int(len(paths_all) * float(SAMPLE_FRAC)))
    idx = rng.choice(len(paths_all), size=k, replace=False)
    paths = [paths_all[i] for i in idx]
elif EVERY_K is not None and EVERY_K > 1:
    paths = paths_all[::EVERY_K]

print("Verwende für Plot:", len(paths))

# --- 2) Durchschnittsfarben berechnen (Threads) ---
RGBs = []
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    futs = [ex.submit(avg_rgb_from_path, p) for p in paths]
    for i, f in enumerate(as_completed(futs), 1):
        v = f.result()
        if v is not None:
            RGBs.append(v)
        if i % 1000 == 0:
            print(f"{i}/{len(futs)} Bilder bearbeitet...")

if not RGBs:
    raise RuntimeError("Keine Durchschnittsfarben berechnet (Pfade/Dateien prüfen).")

RGBs = np.vstack(RGBs).astype(np.float32)  # (N,3), 0..255
colors = (RGBs / 255.0).clip(0, 1)  # (N,3) für Face-Color

# --- 3) VisPy 3D-Scatter ---
from vispy import scene, app

canvas = scene.SceneCanvas(
    keys="interactive", bgcolor="white", size=(1100, 800), show=True
)
view = canvas.central_widget.add_view()
view.camera = "turntable"  # Maus: drehen/zoomen/pannen

markers = scene.visuals.Markers(parent=view.scene)
markers.set_data(
    pos=RGBs,
    face_color=np.c_[colors, np.full((colors.shape[0], 1), ALPHA)],
    size=POINT_SIZE,
    edge_width=0.0,
)

scene.visuals.XYZAxis(parent=view.scene)  # optional Achsen
view.camera.set_range(x=(0, 255), y=(0, 255), z=(0, 255))

if __name__ == "__main__":
    app.run()
