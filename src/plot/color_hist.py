from sklearn.cluster import DBSCAN
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import math

DB_PATH = r"Z:\CODING\UNI\BIG_DATA\data\database.db"
TABLE_NAME = "image_features_test"

# === DB-Verbindung ===
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute(f"""
    SELECT color_hist, umap_x, umap_y FROM {TABLE_NAME}
    WHERE color_hist IS NOT NULL AND umap_x IS NOT NULL AND umap_y IS NOT NULL
""")
rows = cursor.fetchall()
conn.close()

# === Extrahieren ===
hists = [np.array([float(v) for v in row[0].split(",")]) for row in rows]
coords = np.array([[row[1], row[2]] for row in rows])

# === DBSCAN-Clustering auf UMAP-Koordinaten ===
dbscan = DBSCAN(eps=1.5, min_samples=5).fit(coords)
labels = dbscan.labels_  # Cluster-IDs, -1 = Rauschen

# === Mittelwert-Histogramme pro Cluster ===
cluster_histograms = {}
for i in range(len(hists)):
    cluster = labels[i]
    if cluster == -1:
        continue  # Rauschen ignorieren
    if cluster not in cluster_histograms:
        cluster_histograms[cluster] = []
    cluster_histograms[cluster].append(hists[i])

# === Plots: alle Cluster zusammen in Grid ===
num_clusters = len(cluster_histograms)
cols = 3
rows = math.ceil(num_clusters / cols)
fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axs = axs.flatten()

for i, (cluster_id, hist_list) in enumerate(cluster_histograms.items()):
    mean_hist = np.mean(hist_list, axis=0)
    r_bins = mean_hist[0:32]
    g_bins = mean_hist[32:64]
    b_bins = mean_hist[64:96]

    ax = axs[i]
    ax.plot(r_bins, color='red', label='Rot')
    ax.plot(g_bins, color='green', label='Grün')
    ax.plot(b_bins, color='blue', label='Blau')
    ax.set_title(f"Cluster {cluster_id}")
    ax.set_xlabel("Bin (innerhalb R/G/B)")
    ax.set_ylabel("Normierte Häufigkeit")
    ax.grid(True)
    ax.legend()

# Leere Plots ausblenden (falls weniger Cluster als Subplots)
for j in range(i + 1, len(axs)):
    fig.delaxes(axs[j])

plt.suptitle("Durchschnitts-Farbhistogramme pro Cluster (DBSCAN auf UMAP)", fontsize=16)
plt.tight_layout()
plt.show()
