import os
import sqlite3
import tempfile
import unittest
import numpy as np
from src.similarity import color as mod



def hist_rgb(bins, r=0.0, g=0.0, b=0.0, r_bin=None, g_bin=None, b_bin=None):
    """
    L1-normalisiertes Histogramm in Reihenfolge [R|G|B] (3*bins).
    r/g/b = Masse pro Kanal. Falls *_bin gesetzt, genau dort; sonst letzter Bin.
    """
    R = np.zeros(bins, dtype=np.float32)
    G = np.zeros(bins, dtype=np.float32)
    B = np.zeros(bins, dtype=np.float32)
    if r > 0:
        R[r_bin if r_bin is not None else bins - 1] = r
    if g > 0:
        G[g_bin if g_bin is not None else bins - 1] = g
    if b > 0:
        B[b_bin if b_bin is not None else bins - 1] = b
    v = np.concatenate([R, G, B]).astype(np.float32)
    s = float(v.sum())
    return v / s if s > 0 else v


def to_csv(v: np.ndarray) -> str:
    """Vektor -> CSV-String mit 8 Nachkommastellen."""
    return ",".join(f"{float(x):.8f}" for x in v.astype(np.float32))


class TestUtils(unittest.TestCase):
    def test_to_uint8_shapes(self):
        # Grau -> BGR (3 Kanäle)
        gray = np.full((5, 5), 128, dtype=np.uint8)
        bgr = mod.to_uint8(gray)
        self.assertEqual(bgr.shape, (5, 5, 3))
        # Float [0,1] -> uint8
        f = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        f = np.stack([f, f, f], axis=-1)  # 1x3x3
        u8 = mod.to_uint8(f)
        self.assertEqual(u8.dtype, np.uint8)

    def test_swap_rb_hist(self):
        bins = mod.BINS
        # Start als [B|G|R] für die Funktion: setze B=1 im ersten Bin
        v_bgr = np.zeros(3 * bins, dtype=np.float32)
        v_bgr[0] = 1.0  # B-Kanal, Bin 0
        out = mod.swap_rb_hist(v_bgr, bins)  # -> [R|G|B]
        r_sum = float(out[:bins].sum())
        g_sum = float(out[bins : 2 * bins].sum())
        b_sum = float(out[2 * bins :].sum())
        self.assertAlmostEqual(r_sum, 0.0, places=6)
        self.assertAlmostEqual(g_sum, 0.0, places=6)
        self.assertAlmostEqual(b_sum, 1.0, places=6)

    def test_parse_hist_text_errors(self):
        wrong = "0.1,0.2,0.3"  # zu kurz
        with self.assertRaises(ValueError):
            mod.parse_hist_text(wrong)


class TestDistancesAndMapping(unittest.TestCase):
    def test_to_similarity_ranges(self):
        d = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        # 1 - d für diese Metriken -> in [0,1]
        for m in ("hellinger", "emd", "intersect"):
            s = mod.to_similarity(m, d)
            self.assertTrue(np.all(s >= 0.0) and np.all(s <= 1.0))
        # chi2 wird robust skaliert -> ebenfalls in [0,1]
        s_chi = mod.to_similarity("chi2", d)
        self.assertTrue(np.all(s_chi >= 0.0) and np.all(s_chi <= 1.0))

    def test_channel_constraints_penalize_wrong_colors(self):
        bins = mod.BINS
        # Query: reines Rot (als [R|G|B])
        q = hist_rgb(bins, r=1.0)
        # Kandidaten: Rot vs. Blau
        x_red = hist_rgb(bins, r=1.0)
        x_blue = hist_rgb(bins, b=1.0)
        X = np.vstack([x_red, x_blue]).astype(np.float32)

        fused = np.array([0.9, 0.9], dtype=np.float32)  # gleicher Startscore
        adjusted = mod.apply_channel_constraints_universal(fused, X, q, bins=bins)
        # Rot darf nicht schlechter als Blau werden
        self.assertGreaterEqual(float(adjusted[0]), float(adjusted[1]))


class TestSearchEndToEnd(unittest.TestCase):
    def setUp(self):
        # temporäre SQLite-DB
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "test.db")
        self.conn = sqlite3.connect(self.db_path)
        cur = self.conn.cursor()

        # Tabelle mit Präfix aus dem Modul
        self.table = f"{mod.TABLE_PREFIX}1"
        cur.execute(
            f"""
            CREATE TABLE {self.table} (
                id INTEGER PRIMARY KEY,
                {mod.HIST_COL} TEXT,
                filepath TEXT
            );
            """
        )

        bins = mod.BINS
        # Drei Bilder: Rot, Grün, Blau – alle in [R|G|B]
        h_red = hist_rgb(bins, r=1.0)
        h_grn = hist_rgb(bins, g=1.0)
        h_blu = hist_rgb(bins, b=1.0)

        cur.executemany(
            f"INSERT INTO {self.table} (id, {mod.HIST_COL}, filepath) VALUES (?, ?, ?)",
            [
                (1, to_csv(h_red), "red.png"),
                (2, to_csv(h_grn), "green.png"),
                (3, to_csv(h_blu), "blue.png"),
            ],
        )
        self.conn.commit()
        self.conn.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_search_prefers_correct_color(self):
        bins = mod.BINS
        q = hist_rgb(bins, r=1.0)  # Query: Rot

        weights = {"chi2": 1.8, "hellinger": 1.0, "intersect": 0.2, "emd": 1.3}
        results = mod.search_color_voting(
            db_path=self.db_path,
            q_hist=q,
            hist_col=mod.HIST_COL,
            metrics=mod.DEFAULT_METRICS,
            weight_map=weights,
            topk=3,
        )

        # 3 Ergebnisse erwartet
        self.assertEqual(len(results), 3)
        # Bestes Ergebnis: id=1 (rot)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["path"], "red.png")
        # Score[0] > Score[1]
        self.assertGreater(
            float(results[0]["fused_similarity"]),
            float(results[1]["fused_similarity"]),
        )

    def test_search_handles_missing_tables(self):
        # leere DB-Datei (ohne Tabellen) -> keine Ergebnisse
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.close()
        res = mod.search_color_voting(
            db_path=tmp.name,
            q_hist=hist_rgb(mod.BINS, r=1.0),
            hist_col=mod.HIST_COL,
            metrics=("chi2",),
            weight_map={"chi2": 1.0},
            topk=5,
        )
        os.unlink(tmp.name)
        self.assertEqual(res, [])


if __name__ == "__main__":
    unittest.main()
