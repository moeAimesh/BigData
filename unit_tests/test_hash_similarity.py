import os
import sqlite3
import tempfile
import unittest
import numpy as np
from src.similarity import hash as mod


class TestHelpers(unittest.TestCase):
    def test_parse_hash_value_variants(self):
        # Dezimal
        self.assertEqual(int(mod._parse_hash_value(15)), 15)
        self.assertEqual(int(mod._parse_hash_value("15")), 15)
        # Hex
        self.assertEqual(int(mod._parse_hash_value("0x0f")), 15)
        self.assertEqual(int(mod._parse_hash_value("FF")), 255)
        # Binär (mind. 32 Zeichen) -> 15
        bin_15 = "0" * 60 + "1111"
        self.assertEqual(int(mod._parse_hash_value(bin_15)), 15)
        # Bytes (big-endian)
        self.assertEqual(int(mod._parse_hash_value(b"\x00\x10")), 16)
        # None bleibt None
        self.assertIsNone(mod._parse_hash_value(None))

    def test_bit_norm_returns_bucket_64_128_256(self):
        self.assertEqual(mod._bit_norm(32, (1 << 40)), 64)
        self.assertEqual(mod._bit_norm(80, (1 << 100)), 128)
        self.assertEqual(mod._bit_norm(200, (1 << 200)), 256)

    def test_hamming_uint64_vec_popcount(self):
        arr = np.array([0, 1, 3, 0xF], dtype=np.uint64)
        d = mod._hamming_uint64_vec(arr, np.uint64(0))
        self.assertTrue(np.array_equal(d, np.array([0, 1, 2, 4])))

    def test_topk_indices_smallest_first(self):
        arr = np.array([5, 2, 8, 1, 3], dtype=np.int64)
        idx = mod._topk_indices(arr, 3)
        self.assertTrue(np.array_equal(idx, np.array([3, 1, 4])))

    def test_column_exists(self):
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "t.db")
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute("CREATE TABLE demo (id INTEGER, path TEXT)")
            self.assertTrue(mod._column_exists(cur, "demo", "path"))
            self.assertFalse(mod._column_exists(cur, "demo", "missing"))
            con.close()


class TestSearchHashVoting(unittest.TestCase):
    def setUp(self):
        # temporäre DB mit zwei Tabellen
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmpdir.name, "hash.db")
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        for i in (1, 2):
            cur.execute(
                f"""
                CREATE TABLE image_features_part_{i} (
                    id INTEGER PRIMARY KEY,
                    image_hash BLOB,
                    dhash BLOB,
                    phash BLOB,
                    path TEXT
                );
                """
            )

        # Query-Hashes: alle 0 (64 Bit)
        self.q_a = 0
        self.q_d = 0
        self.q_p = 0

        # Kandidaten
        cur.executemany(
            "INSERT INTO image_features_part_1 (id, image_hash, dhash, phash, path) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    1,
                    np.uint64(0).tobytes(),
                    np.uint64(0).tobytes(),
                    np.uint64(0).tobytes(),
                    "match.png",
                ),
                (
                    2,
                    np.uint64(1).tobytes(),
                    np.uint64(3).tobytes(),
                    np.uint64(7).tobytes(),
                    "near.png",
                ),
                (4, None, None, None, "bad.png"),
            ],
        )
        big = (1 << 10) | (1 << 20) | (1 << 30) | (1 << 40) | (1 << 50)
        cur.executemany(
            "INSERT INTO image_features_part_2 (id, image_hash, dhash, phash, path) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    3,
                    np.uint64(big).tobytes(),
                    np.uint64(big).tobytes(),
                    np.uint64(big).tobytes(),
                    "far.png",
                ),
            ],
        )
        con.commit()
        con.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_search_fixed_weights_returns_sorted(self):
        res = mod.search_by_hash_voting_multitables(
            db_path=self.db_path,
            q_ahash=self.q_a,
            q_dhash=self.q_d,
            q_phash=self.q_p,
            weights=(1.0, 1.0, 1.0),
            topk_per_hash=10,
            final_k=3,
            parts=2,  # wichtig!
            return_weights=False,
        )
        self.assertEqual(len(res), 3)
        self.assertEqual(res[0]["id"], 1)
        self.assertEqual(res[0]["path"], "match.png")
        self.assertGreaterEqual(float(res[0]["score"]), float(res[1]["score"]))

    def test_search_auto_weights_and_return_weights(self):
        res, w = mod.search_by_hash_voting_multitables(
            db_path=self.db_path,
            q_ahash=self.q_a,
            q_dhash=self.q_d,
            q_phash=self.q_p,
            weights="auto",
            topk_per_hash=10,
            final_k=3,
            parts=2,  # wichtig!
            return_weights=True,
        )
        self.assertTrue(len(res) >= 1)
        self.assertEqual(len(w), 3)
        self.assertAlmostEqual(sum(w), 1.0, places=6)

    def test_search_handles_no_results(self):
        with tempfile.TemporaryDirectory() as td:
            empty_db = os.path.join(td, "empty.db")
            con = sqlite3.connect(empty_db)
            cur = con.cursor()
            cur.execute(
                "CREATE TABLE image_features_part_1 (id, image_hash BLOB, dhash BLOB, phash BLOB)"
            )
            con.commit()
            con.close()

            res = mod.search_by_hash_voting_multitables(
                db_path=empty_db,
                q_ahash=0,
                q_dhash=0,
                q_phash=0,
                topk_per_hash=5,
                final_k=5,
                parts=1,  # wichtig!
            )
            self.assertEqual(res, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
