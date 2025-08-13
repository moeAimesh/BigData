import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# 1) Verbindung öffnen (pfad anpassen)
conn = sqlite3.connect(r"C:\BIG_DATA\data\database.db")

# 2) Alle Teile laden und zusammenführen
tables = [f"image_features_part_{i}" for i in range(1, 6)]
dfs = []
for tbl in tables:
    df_part = pd.read_sql_query(
        f"SELECT umap_x AS x, umap_y AS y FROM {tbl}",
        conn
    )
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)

# 3) Plot
plt.figure(figsize=(10,8))
plt.scatter(df['x'], df['y'], s=2, alpha=0.5)
plt.title(f"UMAP-Embedding von {len(df)} Bildern (aus {len(tables)} Tabellen)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4) Verbindung schließen
conn.close()
