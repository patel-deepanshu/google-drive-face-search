"""Remembers the face embeddings of every photo already scanned (SQLite).

Scanning is the slow part. With this cache, searching the same Drive folder for a second person
(or re-running a search) needs no downloads and no detection: it is just a dot product.
"""
import sqlite3
import threading

import numpy as np


class EmbeddingCache:
    def __init__(self, path):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS scans (file_id TEXT, profile TEXT, version TEXT, embs BLOB,"
            " PRIMARY KEY (file_id, profile))"
        )
        self._lock = threading.Lock()
        self._pending = 0

    def get_many(self, profile: str, files: list) -> dict:
        """Return {file_id: embeddings (n, 512)} for files scanned before and unchanged since."""
        want = {f.id: f.version for f in files}
        ids, hits = list(want), {}
        with self._lock:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                rows = self._db.execute(
                    f"SELECT file_id, version, embs FROM scans WHERE profile=? AND file_id IN ({','.join('?' * len(chunk))})",
                    [profile, *chunk],
                ).fetchall()
                for file_id, version, blob in rows:
                    if version == want[file_id]:
                        hits[file_id] = np.frombuffer(blob, np.float32).reshape(-1, 512)
        return hits

    def put(self, file_id: str, profile: str, version: str, embs: np.ndarray) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO scans VALUES (?,?,?,?)",
                (file_id, profile, version, np.ascontiguousarray(embs, np.float32).tobytes()),
            )
            self._pending += 1
            if self._pending >= 100:
                self._flush()

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def close(self) -> None:
        with self._lock:
            self._flush()
            self._db.close()

    def _flush(self) -> None:
        self._db.commit()
        self._pending = 0
