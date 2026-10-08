"""Paths and tunables. Everything can be overridden with environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "output"          # matched photos: output/<Name>/...
DATA_DIR = ROOT / "data"
THUMB_DIR = DATA_DIR / "thumbs"       # small previews shown in the web page
CACHE_DB = DATA_DIR / "embeddings.sqlite3"
STATIC_DIR = ROOT / "static"

# "buffalo_sc" = small and fast, "buffalo_l" = larger and more accurate (see README).
FACE_MODEL = os.getenv("FACE_MODEL", "buffalo_sc")

# Measured fastest: many images in parallel, each ONNX call single-threaded. Giving every call
# several threads oversubscribes the CPU and was ~3x slower on a 2-core laptop.
CPU_WORKERS = int(os.getenv("FACE_WORKERS", os.cpu_count() or 2))  # images analysed at once
ORT_THREADS = int(os.getenv("FACE_THREADS", 1))                    # threads inside one ONNX call
DOWNLOAD_THREADS = int(os.getenv("FACE_DL_THREADS", 16))           # parallel Drive downloads

# Face-detector input size per quality preset. Drive thumbnails are requested at the same size,
# so a lower preset also means smaller (faster) downloads.
QUALITY = {"fast": 640, "balanced": 960, "accurate": 1280}
DEFAULT_QUALITY = "balanced"

for d in (MODEL_DIR, OUTPUT_DIR, DATA_DIR, THUMB_DIR):
    d.mkdir(parents=True, exist_ok=True)
