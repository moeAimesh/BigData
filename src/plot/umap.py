import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# === Verbindung öffnen (Pfad anpassen) ===
conn = sqlite3.connect(r"Z:\CODING\UNI\BIG_DATA\data\database.db")

# Tabellenliste manuell (1 bis 6)
tables = [f"image_features_part_{i}" for i in range(1, 7)]

dfs = []  # einzelne DataFrames pro Tabelle

for tbl in tables:
    # Nur Spalten umap_x und umap_y abrufen
    df_part = pd.read_sql_query(
        f"SELECT umap_x AS x, umap_y AS y FROM {tbl}",
        conn
    )
    dfs.append(df_part)

# Alles zu einem DataFrame zusammenfügen
df = pd.concat(dfs, ignore_index=True)

plt.figure(figsize=(10, 8))
plt.scatter(df['x'], df['y'], s=2, alpha=0.5)

plt.title(f"UMAP-Embedding von {len(df)} Bildern (aus {len(tables)} Tabellen)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.tight_layout()
plt.show()

conn.close()

