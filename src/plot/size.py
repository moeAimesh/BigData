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
