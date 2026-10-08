"""One search = reference photos + a Drive folder -> the matching photos saved on disk.

    list folder -> (cached? skip) -> download resized copy -> detect + embed faces -> compare
                                                                   |
                                         match -> download original -> output/<Name>/
"""
import mimetypes
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np

from .cache import EmbeddingCache
from .config import CPU_WORKERS, DOWNLOAD_THREADS, OUTPUT_DIR, THUMB_DIR
from .drive import DriveClient, DriveError, DriveFile
from .faces import FaceEngine

_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_run_one_at_a_time = threading.Lock()  # a second search waits instead of fighting for the CPU


def safe_dirname(name: str) -> str:
    """Folder name for a person: keeps the name as typed, minus characters Windows can't store."""
    name = re.sub(r"\s+", " ", _BAD_CHARS.sub("_", name or "")).strip(" .")[:80].strip(" .")
    if not name:
        raise ValueError("Please enter a name.")
    return f"_{name}" if name.split(".")[0].upper() in _RESERVED else name


def safe_filename(name: str, mime: str) -> str:
    name = _BAD_CHARS.sub("_", name).strip(" .") or "image"
    if not Path(name).suffix:
        name += mimetypes.guess_extension(mime) or ".jpg"
    if Path(name).stem.upper() in _RESERVED:
        name = "_" + name
    return name[:150]


# ---- reference photos ----------------------------------------------------------------------
def build_references(engine: FaceEngine, files: list) -> tuple:
    """files = [(filename, bytes)]. Returns (reference embeddings, per-file report, warnings)."""
    embs, report, warnings = [], [], []
    for filename, data in files:
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            report.append({"file": filename, "faces": 0, "used": False, "note": "Couldn't read this image"})
            continue
        h, w = img.shape[:2]
        if max(h, w) > 2000:
            s = 2000 / max(h, w)
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        faces = None
        for size in (960, 1280):  # try harder if nothing was found
            faces = engine.analyze(img, size, min_face=24, max_faces=20)
            if len(faces.boxes):
                break
        n = len(faces.boxes)
        if n == 0:
            report.append({"file": filename, "faces": 0, "used": False, "note": "No face found"})
            continue
        areas = (faces.boxes[:, 2] - faces.boxes[:, 0]) * (faces.boxes[:, 3] - faces.boxes[:, 1])
        embs.append(faces.embs[int(np.argmax(areas * faces.scores))])
        report.append({"file": filename, "faces": n, "used": True,
                       "note": f"{n} faces found, used the largest" if n > 1 else ""})
    if not embs:
        return np.zeros((0, 512), np.float32), report, warnings

    embs = np.stack(embs)
    if len(embs) >= 2:  # a photo of somebody else would poison the search: flag it
        sims = embs @ embs.T
        np.fill_diagonal(sims, np.nan)
        used = [r for r in report if r["used"]]
        for r, mean_sim in zip(used, np.nanmean(sims, axis=1)):
            if mean_sim < 0.2:
                warnings.append(f"{r['file']} doesn't look like the same person as your other photos. "
                                "Remove it if it's a different person.")
    centroid = embs.mean(axis=0)
    refs = np.vstack([embs, centroid / np.linalg.norm(centroid)]).astype(np.float32)
    return refs, report, warnings


# ---- job -----------------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    name: str
    folder: str
    out_dir: Path
    status: str = "queued"  # queued | listing | scanning | saving | done | cancelled | error
    message: str = ""
    folder_name: str = ""
    total: int = 0
    scanned: int = 0   # photos fully processed (including cache hits)
    cached: int = 0    # ...of which were answered from the cache
    todo: int = 0      # photos that need a download
    fetched: int = 0   # ...of which are done
    failed: int = 0
    faces: int = 0
    matched: int = 0   # matches found
    saved: int = 0     # matches saved to disk
    started: float = field(default_factory=time.time)
    scan_started: float = 0.0
    finished: float = 0.0
    matches: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    abort: str = ""    # set when the job gives up by itself (e.g. every download fails)
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    claimed: set = field(default_factory=set)

    def snapshot(self, since: int = 0) -> dict:
        with self.lock:
            end = self.finished or time.time()
            return {
                "id": self.id, "name": self.name, "status": self.status, "message": self.message,
                "folder": self.folder, "folder_name": self.folder_name, "output_folder": str(self.out_dir),
                "total": self.total, "scanned": self.scanned, "cached": self.cached, "todo": self.todo,
                "fetched": self.fetched, "failed": self.failed, "faces": self.faces,
                "matched": self.matched, "saved": self.saved,
                "elapsed": round(end - self.started, 1),
                "scan_elapsed": round(end - self.scan_started, 1) if self.scan_started else 0,
                "matches": self.matches[since:], "match_count": len(self.matches),
                "errors": self.errors[-10:], "warnings": self.warnings,
            }

    def add(self, **counters) -> None:
        with self.lock:
            for k, v in counters.items():
                setattr(self, k, getattr(self, k) + v)

    def set(self, **fields) -> None:
        with self.lock:
            for k, v in fields.items():
                setattr(self, k, v)

    def fail(self, what: str, err: Exception, processed: bool = True) -> None:
        with self.lock:
            self.failed += 1
            self.scanned += processed
            self.errors.append(f"{what}: {err}")


JOBS: dict = {}


def new_job(name: str) -> Job:
    folder = safe_dirname(name)
    out_dir = OUTPUT_DIR / folder
    if out_dir.resolve().parent != OUTPUT_DIR.resolve():
        raise ValueError("Invalid name.")
    out_dir.mkdir(parents=True, exist_ok=True)
    job = Job(id=uuid.uuid4().hex[:12], name=name.strip(), folder=folder, out_dir=out_dir)
    for old in [j for j in JOBS.values() if j.finished][:-20]:  # keep the registry small
        JOBS.pop(old.id, None)
    JOBS[job.id] = job
    return job


def _claim_path(job: Job, f: DriveFile) -> tuple:
    """Pick the file name to save under. Returns (path, needs_download)."""
    base = safe_filename(f.name, f.mime)
    stem, ext = Path(base).stem, Path(base).suffix
    candidates = [base, f"{stem}_{f.id[:6]}{ext}"] + [f"{stem}_{f.id[:6]}_{i}{ext}" for i in range(2, 50)]
    with job.lock:
        for cand in candidates:
            p = job.out_dir / cand
            if p in job.claimed:
                continue
            if p.exists():
                if f.size and p.stat().st_size == f.size:  # saved by an earlier run
                    job.claimed.add(p)
                    return p, False
                continue
            job.claimed.add(p)
            return p, True
    raise DriveError("too many files with the same name")


def run_job(job: Job, drive: DriveClient, folder_id: str, refs: np.ndarray, threshold: float, det_size: int,
            recursive: bool, engine: FaceEngine, cache: EmbeddingCache) -> None:
    pool = save_pool = None
    try:
        with _run_one_at_a_time:
            if job.cancel.is_set():
                job.set(status="cancelled", message="Cancelled", finished=time.time())
                return
            job.set(status="listing", message="Reading the Drive folder…")
            files = drive.list_images(folder_id, recursive, lambda n: job.set(message=f"Reading the Drive folder… {n} photos"),
                                      job.cancel.is_set)
            if job.cancel.is_set():
                job.set(status="cancelled", message="Cancelled", finished=time.time())
                return
            job.set(total=len(files))
            if not files:
                job.set(status="done", message="No images found in that folder.", finished=time.time())
                return

            profile = f"{engine.pack}-{det_size}"
            hits = cache.get_many(profile, files)
            todo = [f for f in files if f.id not in hits]
            job.set(status="scanning", message="Scanning photos…", todo=len(todo), scan_started=time.time())

            save_pool = ThreadPoolExecutor(6, thread_name_prefix="save")
            cpu_slots = threading.Semaphore(CPU_WORKERS)
            thumb_dir = THUMB_DIR / job.folder
            thumb_dir.mkdir(parents=True, exist_ok=True)
            url_dir = quote(job.folder)

            def save_match(f: DriveFile, score: float) -> None:
                try:
                    path, needs_download = _claim_path(job, f)
                    if needs_download:
                        drive.download_to(f, path)
                    thumb = thumb_dir / (path.name + ".jpg")
                    if not thumb.exists():
                        data = drive.fetch_thumbnail(f)
                        if data:
                            thumb.write_bytes(data)
                    with job.lock:
                        job.saved += 1
                        job.matches.append({
                            "file": path.name, "score": round(score, 3),
                            "url": f"/results/{url_dir}/{quote(path.name)}",
                            "thumb": f"/thumbs/{url_dir}/{quote(thumb.name)}" if thumb.exists() else None,
                        })
                except Exception as e:  # keep going; report it
                    job.fail(f"saving {f.name}", e, processed=False)

            def judge(f: DriveFile, embs: np.ndarray) -> None:
                job.add(scanned=1, faces=len(embs))
                if len(embs):
                    score = float((embs @ refs.T).max())
                    if score >= threshold:
                        job.add(matched=1)
                        save_pool.submit(save_match, f, score)

            # 1) photos we've scanned before: instant
            for f in files:
                if f.id in hits:
                    job.add(cached=1)
                    judge(f, hits[f.id])

            # 2) new photos: download (many at once) -> analyse (CPU_WORKERS at once)
            def scan(f: DriveFile) -> None:
                if job.cancel.is_set():
                    return
                try:
                    data = drive.fetch_scan(f, det_size)
                    with cpu_slots:
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            raise DriveError("not an image this tool can read")
                        embs = engine.analyze(img, det_size).embs
                    cache.put(f.id, profile, f.version, embs)
                    job.add(fetched=1)
                    judge(f, embs)
                except Exception as e:
                    job.add(fetched=1)
                    job.fail(f.name, e)
                    if job.failed >= 25 and job.failed == job.scanned - job.cached:  # nothing has worked so far
                        job.set(abort=f"Too many photos failed to download. Last error: {e}")
                        job.cancel.set()

            pool = ThreadPoolExecutor(DOWNLOAD_THREADS, thread_name_prefix="scan")
            for f in todo:
                pool.submit(scan, f)
            pool.shutdown(wait=True)
            pool = None
            cache.flush()

            job.set(status="saving", message="Saving matched photos…")
            save_pool.shutdown(wait=True)
            save_pool = None

        if job.abort:
            job.set(status="error", message=job.abort, finished=time.time())
        elif job.cancel.is_set():
            job.set(status="cancelled", message="Cancelled", finished=time.time())
        else:
            job.set(status="done", finished=time.time(),
                    message=f"Found {job.saved} photo{'s' if job.saved != 1 else ''} of {job.name}.")
    except Exception as e:
        job.set(status="error", message=str(e) if isinstance(e, DriveError) else f"Unexpected error: {e}", finished=time.time())
    finally:
        for p in (pool, save_pool):
            if p:
                p.shutdown(wait=False, cancel_futures=True)
        try:
            cache.flush()
        except Exception:
            pass
