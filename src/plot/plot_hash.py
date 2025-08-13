import sqlite3
import numpy as np
import matplotlib.pyplot as plt

DB_PATH = r"C:\BIG_DATA\data\database.db"
TABLES = [f"image_features_part_{i}" for i in range(1, 8)]
COLS = ["image_hash", "dhash", "phash"]
BITS = 64  # bei Bedarf auf 128/256 anpassen

# ---------- Helper ----------
def _parse_to_uint64_list(vals):
    """
    Nimmt eine Liste aus (int/str/bytes) und gibt np.uint64 Array zurück.
    - str: Hex ('0x..' oder a-f) oder Dezimal
    - bytes: big-endian interpretiert (links mit Nullen auf 8 Bytes auffüllen)
    """
    out = np.empty(len(vals), dtype=np.uint64)
    for i, v in enumerate(vals):
        if v is None:
            out[i] = np.uint64(0)
            continue
        if isinstance(v, (np.integer, int)):
            out[i] = np.uint64(int(v))
        elif isinstance(v, (bytes, bytearray)):
            b = bytes(v)
            if len(b) < 8:
                b = b'\x00' * (8 - len(b)) + b
            else:
                b = b[-8:]  # falls länger, nur die letzten 8 Bytes (LSB) nehmen
            out[i] = np.frombuffer(b, dtype='>u8')[0]  # big-endian lesen
        elif isinstance(v, str):
            s = v.strip().lower()
            base = 16 if (s.startswith("0x") or any(c in s for c in "abcdef")) else 10
            out[i] = np.uint64(int(s, base))
        else:
            out[i] = np.uint64(int(v))
    return out

def _bits_msb_to_lsb_uint64(arr_uint64):
    """
    Wandelt np.uint64 Array in Bitmatrix (N, 64) um, MSB→LSB.
    Tipp: byteswap() sorgt für big-endian Byte-Reihenfolge; unpackbits gibt pro Byte MSB→LSB aus.
    """
    be = arr_uint64.byteswap().view(np.uint8).reshape(-1, 8)  # big-endian Bytes
    bits = np.unpackbits(be, axis=1)  # (N, 8*8)
    return bits

# ---------- Aggregation ----------
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# Zähler für '1'-Bits pro Spalte
sum_bits = {c: np.zeros(BITS, dtype=np.int64) for c in COLS}
count_rows = {c: 0 for c in COLS}

for table in TABLES:
    # Nur vorhandene Spalten ziehen (robuster)
    cur.execute(f"PRAGMA table_info({table})")
    cols_in_table = {row[1] for row in cur.fetchall()}
    needed = [c for c in COLS if c in cols_in_table]
    if not needed:
        continue

    # Alle drei Spalten (sofern vorhanden) in einem Rutsch selektieren
    sel_cols = ", ".join(needed)
    cur.execute(f"SELECT {sel_cols} FROM {table} WHERE " + " OR ".join([f"{c} IS NOT NULL" for c in needed]))
    rows = cur.fetchall()
    if not rows:
        continue

    # Spaltenweise sammeln
    col_data = {c: [] for c in needed}
    for row in rows:
        for j, c in enumerate(needed):
            val = row[j]
            if val is not None:
                col_data[c].append(val)

    # Für jede vorhandene Spalte: in uint64 -> Bits -> aufsummieren
    for c in needed:
        if not col_data[c]:
            continue
        arr = _parse_to_uint64_list(col_data[c])

        # Falls du sicher 64-Bit hast: direkt umwandeln.
        bits = _bits_msb_to_lsb_uint64(arr)  # (N, 64)
        if bits.shape[1] != BITS:
            # Bei abweichender Länge auf gewünschte BITS bringen (links mit Nullen oder abschneiden)
            if bits.shape[1] > BITS:
                bits = bits[:, -BITS:]  # nur die letzten BITS (LSB-seitig) behalten
            else:
                pad = np.zeros((bits.shape[0], BITS - bits.shape[1]), dtype=bits.dtype)
                bits = np.hstack([pad, bits])

        sum_bits[c] += bits.sum(axis=0, dtype=np.int64)
        count_rows[c] += bits.shape[0]

conn.close()

# ---------- Mittelwerte berechnen ----------
bit_means = {c: (sum_bits[c] / count_rows[c]) if count_rows[c] > 0 else None for c in COLS}

# ---------- Plot ----------
fig, axes = plt.subplots(nrows=len(COLS), ncols=1, figsize=(12, 3*len(COLS)), sharex=True)
if len(COLS) == 1:
    axes = [axes]

for ax, c in zip(axes, COLS):
    if bit_means[c] is None:
        ax.set_title(f"{c}: keine Daten gefunden")
        ax.set_xlabel("Bit-Position (MSB → LSB)")
        ax.set_ylabel("Anteil '1'")
        ax.grid(True, alpha=0.3)
        continue
    x = np.arange(BITS)
    ax.bar(x, bit_means[c])
    ax.set_title(f"Häufigkeit von '1'-Bits – {c}")
    ax.set_xlabel("Bit-Position (MSB → LSB)")
    ax.set_ylabel("Anteil '1'")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
