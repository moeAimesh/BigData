# 2D/3D feature vektors plot with UMAP, t-SNE, TMAP



import sqlite3
import matplotlib.pyplot as plt

# === CONFIG ===
DB_PATH = r"Z:\CODING\UNI\BIG_DATA\data\database.db"
TABLE_NAME = "image_features_test"

# === Verbindung zur Datenbank ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# === Koordinaten und Dateinamen abfragen ===
cursor.execute(f"""
    SELECT filename, umap_x, umap_y FROM {TABLE_NAME}
    WHERE umap_x IS NOT NULL AND umap_y IS NOT NULL
""")
rows = cursor.fetchall()

# === Schließen ===
conn.close()

# === Daten aufteilen ===
filenames = [row[0] for row in rows]
x_coords = [row[1] for row in rows]
y_coords = [row[2] for row in rows]

# === Visualisierung ===
plt.figure(figsize=(10, 8))
plt.scatter(x_coords, y_coords, s=5, alpha=0.6)
plt.title("UMAP Visualisierung der Bilder")
plt.xlabel("UMAP X")
plt.ylabel("UMAP Y")
plt.grid(True)
plt.show()
