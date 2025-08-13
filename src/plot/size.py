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

plt.figure(figsize=(14, 6))
x = np.arange(len(sizes_global))

# Rohdaten
plt.plot(x, sizes_global, label='Bildgröße (MB)', alpha=0.3)

# Gleitender Durchschnitt
plt.plot(np.arange(window-1, len(sizes_global)), rolling_avg,
         color='orange', label=f'Ø über {window} Bilder')

# Gesamtdurchschnitt als Linie
plt.axhline(mean_all, color='r', linestyle='--', label='Gesamt-Ø')

# Vertikale Linien zur Trennung der Tables
cumsum = np.cumsum(table_boundaries)
for boundary in cumsum[:-1]:
    plt.axvline(boundary, color='grey', linestyle=':', alpha=0.5)

# Achsen + Titel
plt.xlabel('Globaler Bildindex')
plt.ylabel('Dateigröße (MB)')
plt.title('Dateigrößenverlauf über alle image_features_part_ Tabellen')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
