import json
import numpy as np
import sqlite3
from typing import List, Dict, Any
import cv2
# from features.color_vec import calc_histogram   # später erneut testen

# === Konstanten ===
DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]

# Histogramm-Parameter
PARTS = 7
BINS = 32
CH = 3
DIM = CH * BINS   # 96
