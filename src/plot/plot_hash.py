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

# === Hash-Strings in Binärvektoren umwandeln ===
# Beispiel: Hex-String 'a1b2c3' → Binärstring
hash_bits = []
for h in rows:
    hexstr = h[0]  # Hash als Hex-String
    binstr = bin(int(hexstr, 16))[2:].zfill(len(hexstr) * 4)  # Hex → Binär (mit führenden Nullen)
    bit_array = np.array([int(b) for b in binstr])
    hash_bits.append(bit_array)

# Matrix: Zeilen = Bilder, Spalten = Bits
hash_matrix = np.stack(hash_bits)  # Shape: [N, B]
