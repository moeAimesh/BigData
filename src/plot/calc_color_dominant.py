import sqlite3, os, numpy as np

DB_PATH   = r"C:\BIG_DATA\data\database.db"
TABLES    = [f"image_features_part_{i}" for i in range(1, 8)]
HIST_COL  = "color_hist"
OUT_NPZ   = r"C:\BIG_DATA\data\avg_rgb_points_from_hist_bold_mode_maxed.npz"

BINS = 32
DIM  = 3*BINS

# Farbverstärkung
SAT_BOOST = 1.7
VAL_BOOST = 1.15
GAMMA     = 1.3

# Dominanz-Modus
MODE  = "soft"
ALPHA = 3.0

# Positions-Stretch
STRETCH = 1.2

# Bin-Zentren
edges   = np.linspace(0, 256, BINS+1, dtype=np.float32)
centers = (edges[:-1] + edges[1:]) * 0.5
