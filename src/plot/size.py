import sqlite3
import matplotlib.pyplot as plt
import numpy as np

# === Pfad zur SQLite-Datenbank ===
DB_PATH = r"Z:\CODING\UNI\BIG_DATA\data\database.db"

# === Verbindung öffnen ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# SQL: alle Tabellen mit Name 'image_features_part_%'
cursor.execute("""
    SELECT name
      FROM sqlite_master
     WHERE type='table'
       AND name LIKE 'image_features_part_%'
    ORDER BY name
""")
tables = [row[0] for row in cursor.fetchall()]

sizes_global = []     # alle Dateigrößen (in MB)
table_boundaries = [] # Anzahl Bilder pro Tabelle

for table in tables:
    cursor.execute(f"SELECT file_size FROM {table} ORDER BY id")
    # Bytes → MB umrechnen
    sizes = [row[0] / (1024 * 1024) for row in cursor.fetchall()]
    table_boundaries.append(len(sizes))
    sizes_global.extend(sizes)

conn.close()
# Gleitender Durchschnitt mit Fenstergröße 1000
window = 1000
rolling_avg = np.convolve(sizes_global, np.ones(window)/window, mode='valid')

# Gesamtdurchschnitt aller Bilder
mean_all = np.mean(sizes_global)
