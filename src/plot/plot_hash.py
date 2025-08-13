import sqlite3
import numpy as np
import matplotlib.pyplot as plt

# === Konstanten ===
DB_PATH = r"Z:\CODING\UNI\BIG_DATA\data\database.db"
TABLE_NAME = "image_features_part_6"

# === Hashes laden ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(f"""
    SELECT image_hash FROM {TABLE_NAME}
    WHERE image_hash IS NOT NULL
""")
rows = cursor.fetchall()
conn.close()
