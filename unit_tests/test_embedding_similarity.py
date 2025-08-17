import os
import io
import sys
import cv2
import json
import pickle
import sqlite3
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from unittest import mock
from src.similarity import embedding as mod


class TestUtils(unittest.TestCase):
    def test_l2_normalize_rows_shapes(self):
        # Vektor -> Ergebnis bleibt 1D, Norm ≈ 1
        v = np.array([3.0, 4.0], dtype=np.float32)
        out = mod.l2_normalize_rows(v)
        self.assertEqual(out.shape, (2,))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=6)

        # Matrix -> jede Zeile Norm ≈ 1
        M = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        out2 = mod.l2_normalize_rows(M)
        self.assertEqual(out2.shape, (2, 2))
        self.assertAlmostEqual(float(np.linalg.norm(out2[0])), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(out2[1])), 1.0, places=6)

    def test_fast_load_bytes_bgr_and_preprocess(self):
        # kleines Dummy-Bild erzeugen und laden
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "img.png"
            bgr = np.full((10, 12, 3), 200, dtype=np.uint8)
            ok, buf = cv2.imencode(".png", bgr)
            self.assertTrue(ok)
            p.write_bytes(buf.tobytes())

            img = mod._fast_load_bytes_bgr(p)
            self.assertEqual(img.shape[2], 3)

            arr = mod.preprocess_for_model(img)
            self.assertEqual(arr.shape, (mod.TARGET_SIZE[1], mod.TARGET_SIZE[0], 3))
            self.assertEqual(arr.dtype, np.float32)
            self.assertTrue(arr.max() <= 1.0 and arr.min() >= 0.0)

    def test_to_embed_dtype(self):
        x = (np.ones((5, 5, 3)) * 0.5).astype(np.float64)
        y = mod.to_embed_dtype(x)
        self.assertEqual(y.dtype, np.float32)


class TestPCAandHNSW(unittest.TestCase):
    def test_pca_transformer_uses_joblib(self):
        # Fake-IPCA: transform -> 64 Einsen
        class FakeIPCA:
            def transform(self, X):
                n = X.shape[0]
                return np.ones((n, 64), dtype=np.float32)

        with tempfile.TemporaryDirectory() as td:
            fake_model = Path(td) / "ipca.joblib"
            fake_model.write_bytes(b"dummy")

            with mock.patch.object(mod, "load", return_value=FakeIPCA()):
                tr = mod.PCATransformer(fake_model)
                vec512 = np.zeros(512, dtype=np.float32)
                out64 = tr.transform64(vec512)
                self.assertEqual(out64.shape, (64,))
                self.assertTrue(np.allclose(out64, 1.0))

    def test_hnsw32_basic(self):
        # Mini-Index mit 2 Vektoren
        idx = mod.HNSW32(dim=32)
        idx.init_new(max_elements=2)

        v1 = np.ones(32, dtype=np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = np.zeros(32, dtype=np.float32)
        v2[0] = 1.0
        v2 /= np.linalg.norm(v2)

        idx.add_batch(
            np.vstack([v1, v2]),
            metas=[
                ("t", 1, "a", "p32a", "p64a", "p512a"),
                ("t", 2, "b", "p32b", "p64b", "p512b"),
            ],
        )

        labels, sims = idx.knn(v1, k=1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(int(labels[0]), 0)  # erster Eintrag
        self.assertGreaterEqual(float(sims[0]), 0.9)


class TestDBUtils(unittest.TestCase):
    def test_db_table_helpers_and_fetch(self):
        with tempfile.TemporaryDirectory() as td:
            db_path = Path(td) / "tmp.db"
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()

            tname = f"{mod.TABLE_PREFIX}1"
            cur.execute(
                f"""
                CREATE TABLE "{tname}" (
                    {mod.COL_ID} INTEGER PRIMARY KEY,
                    {mod.COL_IMAGE_PATH} TEXT,
                    {mod.COL_EMB32_PATH} TEXT,
                    {mod.COL_EMB64_PATH} TEXT,
                    {mod.COL_EMB512_PATH} TEXT,
                    umap32_x REAL, umap32_y REAL,
                    umap_x REAL, umap_y REAL,
                    umap512_x REAL, umap512_y REAL
                );
                """
            )

            # Dummy-Embeddings ablegen
            p32 = Path(td) / "v32.npy"
            p64 = Path(td) / "v64.npy"
            p512 = Path(td) / "v512.npy"
            np.save(p32, np.ones(32, dtype=np.float32))
            np.save(p64, np.ones(64, dtype=np.float32))
            np.save(p512, np.ones(512, dtype=np.float32))

            cur.execute(
                f'INSERT INTO "{tname}" ({mod.COL_ID},{mod.COL_IMAGE_PATH},{mod.COL_EMB32_PATH},{mod.COL_EMB64_PATH},{mod.COL_EMB512_PATH}, umap32_x, umap32_y) VALUES (?,?,?,?,?,?,?)',
                (1, "img.png", str(p32), str(p64), str(p512), 0.1, 0.2),
            )
            con.commit()

            # Tests
            tables = mod.list_feature_tables(con)
            self.assertIn(tname, tables)

            cnt = mod.count_rows(con, tname)
            self.assertEqual(cnt, 1)

            has_cols = mod.table_has_columns(con, tname, ("umap32_x", "umap32_y"))
            self.assertTrue(has_cols)

            rows = list(mod.fetch_rows_iter(con, tname))
            self.assertEqual(len(rows), 1)
            rid, img_path, emb32, emb64, emb512 = rows[0]
            self.assertEqual(rid, 1)
            self.assertTrue(
                emb32.endswith(".npy")
                and emb64.endswith(".npy")
                and emb512.endswith(".npy")
            )

            coords = mod.fetch_umap_coords(con, tname, [1], ("umap32_x", "umap32_y"))
            self.assertIn(1, coords)
            self.assertAlmostEqual(coords[1][0], 0.1, places=6)

            con.close()


class TestSearchImageHappyPath(unittest.TestCase):
    def setUp(self):
        # Temp-Workspace + Dummy-Embeddings
        self.td = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.td.name)

        self.v32 = np.ones(32, dtype=np.float32)
        self.v32 /= np.linalg.norm(self.v32)
        self.v64 = np.ones(64, dtype=np.float32)
        self.v64 /= np.linalg.norm(self.v64)
        self.v512 = np.ones(512, dtype=np.float32)
        self.v512 /= np.linalg.norm(self.v512)

        self.p32 = self.tmpdir / "x32.npy"
        np.save(self.p32, self.v32)
        self.p64 = self.tmpdir / "x64.npy"
        np.save(self.p64, self.v64)
        self.p512 = self.tmpdir / "x512.npy"
        np.save(self.p512, self.v512)

        # leere Fake-DB (Index wird gemockt)
        self.db_path = self.tmpdir / "db.sqlite"
        sqlite3.connect(str(self.db_path)).close()

        # Dummy-Query-Bild
        self.img_path = self.tmpdir / "q.png"
        ok, buf = cv2.imencode(".png", (np.ones((8, 8, 3), dtype=np.uint8) * 127))
        self.assertTrue(ok)
        self.img_path.write_bytes(buf.tobytes())

        # vorbereiteter HNSW-Index mit genau einem perfekten Treffer
        self.idx = mod.HNSW32(dim=32)
        self.idx.init_new(max_elements=1)
        self.idx.add_batch(
            np.vstack([self.v32]),
            metas=[
                (
                    "image_features_part_1",
                    42,
                    "match.png",
                    str(self.p32),
                    str(self.p64),
                    str(self.p512),
                )
            ],
        )

    def tearDown(self):
        self.td.cleanup()

    def test_search_image_with_mocks(self):
        # Fake-IPCA -> 64 Einsen
        class FakeIPCA:
            def transform(self, X):
                n = X.shape[0]
                return np.ones((n, 64), dtype=np.float32)

        # PCATransformer durch Fake ersetzen
        class FakePCATransformer(mod.PCATransformer):
            def __init__(self, _):
                self.ipca = FakeIPCA()

        # Embedding-Extraktion mocken -> 512 Einsen
        def fake_extract_embeddings(batch):
            return [np.ones(512, dtype=np.float32)]

        with mock.patch.object(
            mod, "PCATransformer", FakePCATransformer
        ), mock.patch.object(
            mod, "extract_embeddings", side_effect=fake_extract_embeddings
        ), mock.patch.object(
            mod, "build_index_if_needed", return_value=self.idx
        ):

            out = mod.search_image(self.img_path, self.db_path)

            # Struktur & Inhalte prüfen
            self.assertIn("results", out)
            final_k = out["results"]["final_top_k"]
            self.assertTrue(len(final_k) >= 1)

            self.assertEqual(final_k[0]["id"], 42)
            self.assertEqual(final_k[0]["table"], "image_features_part_1")
            self.assertEqual(final_k[0]["image_path"], "match.png")
            self.assertEqual(final_k[0]["path_32"], str(self.p32))
            self.assertEqual(final_k[0]["path_64"], str(self.p64))
            self.assertEqual(final_k[0]["path_512"], str(self.p512))


if __name__ == "__main__":
    unittest.main(verbosity=2)
