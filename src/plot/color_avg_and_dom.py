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
