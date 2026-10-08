"""Web server: serves the page and a small JSON API.

    POST /api/search          start a search (multipart form: name, folder link, credentials, photos)
    GET  /api/jobs/{id}       progress + matches so far
    POST /api/jobs/{id}/cancel
    GET  /api/results         folders of saved photos
    GET  /api/results/{name}/zip
    /results/<Name>/<file>    the saved photos themselves
"""
import json
import tempfile
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from . import faces, pipeline
from .cache import EmbeddingCache
from .config import (CACHE_DB, DEFAULT_QUALITY, FACE_MODEL, ORT_THREADS, OUTPUT_DIR, QUALITY, STATIC_DIR,
                     THUMB_DIR)
from .drive import DriveClient, DriveError, parse_folder_id

MAX_REF_PHOTOS = 30
MAX_REF_BYTES = 30 * 1024 * 1024

engine = None
cache = EmbeddingCache(CACHE_DB)


def _load_engine() -> None:
    global engine
    try:
        engine = faces.FaceEngine(FACE_MODEL, ORT_THREADS)
    except Exception as e:  # shown on the page instead of crashing the server
        faces.status.update(state="error", message=f"Couldn't load the face model: {e}")
        print(f"[model] ERROR: {e}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=_load_engine, daemon=True).start()  # first run downloads the model
    yield


app = FastAPI(title="Face Search", lifespan=lifespan)


@app.get("/api/health")
def health():
    return faces.status


@app.post("/api/search")
def start_search(
    name: str = Form(...),
    folder_link: str = Form(...),
    auth_mode: str = Form("service_account"),
    credentials: str = Form(""),
    threshold: float = Form(0.40),
    quality: str = Form(DEFAULT_QUALITY),
    include_subfolders: bool = Form(True),
    images: List[UploadFile] = File(...),
):
    if engine is None:
        raise HTTPException(503, faces.status["message"])
    if quality not in QUALITY:
        raise HTTPException(400, f"quality must be one of {', '.join(QUALITY)}")
    threshold = min(max(threshold, 0.1), 0.9)
    try:
        pipeline.safe_dirname(name)
        folder_id = parse_folder_id(folder_link)
    except (ValueError, DriveError) as e:
        raise HTTPException(400, str(e))

    # 1) the face photos: cheap and local, so check them before touching the network
    if len(images) > MAX_REF_PHOTOS:
        raise HTTPException(400, f"Please attach at most {MAX_REF_PHOTOS} photos.")
    uploads = []
    for up in images:
        data = up.file.read(MAX_REF_BYTES + 1)
        if len(data) > MAX_REF_BYTES:
            raise HTTPException(400, f"{up.filename} is larger than {MAX_REF_BYTES // 2**20} MB.")
        uploads.append((up.filename or "photo", data))
    refs, report, warnings = pipeline.build_references(engine, uploads)
    if not len(refs):
        raise HTTPException(400, "No usable face found in the attached photos. "
                                 "Use clear photos where the face is visible. " +
                                 "; ".join(f"{r['file']}: {r['note']}" for r in report if r["note"]))

    # 2) Google Drive: make sure the credentials work and the folder is visible
    try:
        if auth_mode == "service_account":
            try:
                info = json.loads(credentials)
            except ValueError:
                raise DriveError("The service account credentials aren't valid JSON.")
            drive = DriveClient(service_account_info=info)
        elif auth_mode == "api_key":
            drive = DriveClient(api_key=credentials)
        else:
            raise DriveError("Unknown credential type.")
        folder_name = drive.get_folder_name(folder_id)
    except DriveError as e:
        raise HTTPException(400, str(e))

    job = pipeline.new_job(name)
    job.folder_name, job.warnings = folder_name, warnings
    threading.Thread(
        target=pipeline.run_job, name=f"job-{job.id}", daemon=True,
        args=(job, drive, folder_id, refs, threshold, QUALITY[quality], include_subfolders, engine, cache),
    ).start()
    return {"job_id": job.id, "references": report, "warnings": warnings, "folder_name": folder_name}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    job = pipeline.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job (the server may have been restarted).")
    return job.snapshot(since)


@app.post("/api/jobs/{job_id}/cancel")
def cancel(job_id: str):
    job = pipeline.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job.")
    job.cancel.set()
    return {"ok": True}


def _result_dir(name: str) -> Path:
    path = (OUTPUT_DIR / name).resolve()
    if path.parent != OUTPUT_DIR.resolve() or not path.is_dir():
        raise HTTPException(404, "No saved photos with that name.")
    return path


def _photos(folder: Path) -> list:
    return [p for p in folder.iterdir() if p.is_file() and not p.name.endswith(".part")]


@app.get("/api/results")
def list_results():
    return [{"name": d.name, "count": len(_photos(d)), "path": str(d)}
            for d in sorted(OUTPUT_DIR.iterdir()) if d.is_dir()]


@app.get("/api/results/{name}/zip")
def download_zip(name: str):
    folder = _result_dir(name)
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # photos are already compressed
        for p in _photos(folder):
            z.write(p, p.name)
    tmp.close()
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{folder.name}.zip",
                        background=BackgroundTask(lambda: Path(tmp.name).unlink(missing_ok=True)))


app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")
app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
