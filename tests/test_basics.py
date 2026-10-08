"""Quick checks of the pure logic (no network, no models):  python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cache import EmbeddingCache  # noqa: E402
from app.drive import DriveError, DriveFile, normalize_service_account, parse_folder_id  # noqa: E402
from app.pipeline import safe_dirname, safe_filename  # noqa: E402

FID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"


class FolderLinks(unittest.TestCase):
    def test_common_shapes(self):
        for link in (f"https://drive.google.com/drive/folders/{FID}?usp=sharing",
                     f"https://drive.google.com/drive/u/2/folders/{FID}",
                     f"https://drive.google.com/open?id={FID}",
                     f"https://drive.google.com/folderview?id={FID}&usp=sharing", FID, f"  {FID}  "):
            self.assertEqual(parse_folder_id(link), FID, link)

    def test_rejects_garbage(self):
        for link in ("", "not a link", "https://example.com/x"):
            with self.assertRaises(DriveError):
                parse_folder_id(link)


class Names(unittest.TestCase):
    def test_person_folder(self):
        self.assertEqual(safe_dirname("  Priya   Sharma "), "Priya Sharma")
        self.assertEqual(safe_dirname("a/b\\c:d"), "a_b_c_d")
        escaped = safe_dirname("../../etc")  # can't climb out of the output folder
        self.assertNotIn("/", escaped)
        self.assertFalse(escaped.startswith("."))
        self.assertEqual(safe_dirname("CON"), "_CON")
        with self.assertRaises(ValueError):
            safe_dirname("  . ")

    def test_file_name(self):
        self.assertEqual(safe_filename("IMG_1.jpg", "image/jpeg"), "IMG_1.jpg")
        self.assertEqual(safe_filename("IMG_2", "image/png"), "IMG_2.png")
        self.assertEqual(safe_filename('a:b?.jpg', "image/jpeg"), "a_b_.jpg")


class Credentials(unittest.TestCase):
    def test_fills_defaults_and_fixes_escaped_newlines(self):
        info = normalize_service_account({"client_email": " a@b.iam.gserviceaccount.com ",
                                          "private_key": "-----BEGIN PRIVATE KEY-----\\nAAAA\\n-----END PRIVATE KEY-----\\n"})
        self.assertEqual(info["client_email"], "a@b.iam.gserviceaccount.com")
        self.assertIn("\n", info["private_key"])
        self.assertEqual(info["type"], "service_account")
        self.assertTrue(info["token_uri"].startswith("https://"))

    def test_reports_missing_fields(self):
        with self.assertRaises(DriveError):
            normalize_service_account({"client_email": "a@b.c"})
        with self.assertRaises(DriveError):
            normalize_service_account({"client_email": "a@b.c", "private_key": "nope"})


class Cache(unittest.TestCase):
    def test_roundtrip_including_photos_without_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = EmbeddingCache(Path(tmp) / "c.sqlite3")
            a, b = DriveFile("a", "a.jpg", "image/jpeg", 1, "v1", None), DriveFile("b", "b.jpg", "image/jpeg", 1, "v1", None)
            embs = np.random.rand(3, 512).astype(np.float32)
            cache.put("a", "p", "v1", embs)
            cache.put("b", "p", "v1", np.zeros((0, 512), np.float32))
            cache.flush()
            hits = cache.get_many("p", [a, b])
            np.testing.assert_array_equal(hits["a"], embs)
            self.assertEqual(hits["b"].shape, (0, 512))
            changed = DriveFile("a", "a.jpg", "image/jpeg", 1, "v2", None)  # file edited since: must be rescanned
            self.assertNotIn("a", cache.get_many("p", [changed]))
            self.assertNotIn("a", cache.get_many("other-profile", [a]))
            cache.close()


if __name__ == "__main__":
    unittest.main()
