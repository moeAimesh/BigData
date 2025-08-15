# rgb_viewer.py
import numpy as np
from vispy import scene, app

# === Datei & Anzeige ===
NPZ_PATH    = r"C:\BIG_DATA\data\avg_rgb_points_from_hist_bold_mode_maxed.npz"
ALPHA       = 1.0            # Marker-Transparenz (0..1)
POINT_SIZE  = 4.0            # Marker-Größe (Fallback)
BG_COLOR    = "grey"         # neutraler Hintergrund

# LOD (Level of Detail) per Voxel (None = aus)
VOXEL_BINS  = None           # z.B. 64 oder 32

def voxel_downsample_rgb(pos, bins_per_axis=64):
    """
    RGB-Würfel in bins^3 Voxel; pro Voxel Mittelwert & Count.
    Rückgabe:
      positions: (N,3) 0..255  | colors: (N,3) 0..1  | sizes: (N,)
    """
    pos = np.clip(pos, 0, 255).astype(np.float32)
    step = 256.0 / bins_per_axis

    # Voxel-Index je Punkt
    idx = np.minimum(np.floor(pos / step).astype(np.int32), bins_per_axis - 1)
    keys = (idx[:, 0]
            + bins_per_axis * idx[:, 1]
            + (bins_per_axis ** 2) * idx[:, 2]).astype(np.int64)

    # Gruppieren nach Voxel-Key (stabil sortieren)
    order = np.argsort(keys, kind="mergesort")
    keys_sorted = keys[order]
    pos_sorted  = pos[order]

    uniq, start_idx = np.unique(keys_sorted, return_index=True)
    end_idx = np.r_[start_idx[1:], keys_sorted.size]

    means = np.empty((uniq.size, 3), dtype=np.float32)
    sizes = np.empty(uniq.size, dtype=np.float32)

    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        block = pos_sorted[s:e]
        means[i] = block.mean(axis=0)
        count = e - s
        # dichtere Voxel -> größere Marker
        sizes[i] = 2.0 + 3.0 * np.log1p(count)

    colors = (means / 255.0).clip(0, 1).astype(np.float32)
    return means, colors, sizes
