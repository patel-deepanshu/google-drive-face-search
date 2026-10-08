"""Face detection (SCRFD) and face embeddings (ArcFace) running on plain ONNX Runtime.

These are the same models InsightFace ships ("buffalo" packs). We call them directly so the
project needs no compiler and no heavy dependencies: just onnxruntime, numpy and opencv.
"""
import threading
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime as ort
import requests

from .config import MODEL_DIR

cv2.setNumThreads(1)  # we parallelise across images instead; OpenCV's own threads would just fight for the CPU

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/{pack}.zip"
PACKS = {  # pack -> (detector, recognizer)
    "buffalo_sc": ("det_500m.onnx", "w600k_mbf.onnx"),
    "buffalo_l": ("det_10g.onnx", "w600k_r50.onnx"),
}

# Where the 5 landmarks (eyes, nose, mouth corners) should land in a 112x112 ArcFace crop.
_ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

status = {"state": "starting", "message": "Starting..."}  # shown on the web page while models load


def _set_status(state: str, message: str) -> None:
    status["state"], status["message"] = state, message
    print(f"[model] {message}", flush=True)


def ensure_models(pack: str) -> tuple:
    """Return paths of the detector and recognizer, downloading the pack once if missing."""
    if pack not in PACKS:
        raise ValueError(f"Unknown FACE_MODEL '{pack}'. Choose one of: {', '.join(PACKS)}")
    folder = MODEL_DIR / pack
    paths = [folder / name for name in PACKS[pack]]
    if all(p.exists() for p in paths):
        return tuple(paths)

    folder.mkdir(parents=True, exist_ok=True)
    zip_path = folder / f"{pack}.zip.part"
    _set_status("downloading", f"Downloading face model '{pack}' (one time only)...")
    with requests.get(PACK_URL.format(pack=pack), stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done, last = 0, -1
        with open(zip_path, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done * 10 // total != last:
                    last = done * 10 // total
                    _set_status("downloading", f"Downloading face model '{pack}': {done * 100 // total}%")
    with zipfile.ZipFile(zip_path) as z:
        for p in paths:  # only the two files we need, ignore the rest of the pack
            (folder / p.name).write_bytes(z.read(p.name))
    zip_path.unlink()
    return tuple(paths)


def _session(path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.enable_mem_pattern = False  # input sizes vary per image
    so.log_severity_level = 3      # the detector declares fixed output shapes; ORT warns on every call otherwise
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), so, providers=providers)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (scale + rotation + shift) as a 2x3 matrix."""
    sm, dm = src.mean(0), dst.mean(0)
    s, d = src - sm, dst - dm
    u, sv, vt = np.linalg.svd(d.T @ s / len(src))
    flip = np.array([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rot = u @ np.diag(flip) @ vt
    scale = (sv * flip).sum() / ((s ** 2).sum() / len(src))
    m = np.zeros((2, 3), dtype=np.float32)
    m[:, :2] = scale * rot
    m[:, 2] = dm - scale * rot @ sm
    return m


@dataclass
class Faces:
    boxes: np.ndarray   # (n, 4) x1, y1, x2, y2 in the coordinates of the image that was analysed
    scores: np.ndarray  # (n,) detector confidence
    embs: np.ndarray    # (n, 512) L2-normalised embeddings


class FaceEngine:
    def __init__(self, pack: str, threads: int = 2):
        det_path, rec_path = ensure_models(pack)
        _set_status("loading", "Loading face model...")
        self.pack = pack
        self._det = _session(det_path, threads)
        self._rec = _session(rec_path, threads)
        self._det_in = self._det.get_inputs()[0].name
        self._det_out = [o.name for o in self._det.get_outputs()]
        self._rec_in = self._rec.get_inputs()[0].name
        n = len(self._det_out)  # how many output heads the detector has tells us its layout
        self._fmc = 3 if n in (6, 9) else 5
        self._strides = (8, 16, 32) if self._fmc == 3 else (8, 16, 32, 64, 128)
        self._anchors = 2 if self._fmc == 3 else 1
        if n not in (9, 15):
            raise RuntimeError("Detector model has no landmark outputs; it can't be used for face alignment")
        self._centers = {}
        self._centers_lock = threading.Lock()
        self.analyze(np.zeros((320, 320, 3), np.uint8), 320)  # warm-up so the first real image isn't slow
        _set_status("ready", f"Face model '{pack}' ready")

    # ---- detection -------------------------------------------------------------------------
    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        c = self._centers.get(key)
        if c is None:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32).reshape(-1, 2) * stride
            if self._anchors > 1:
                c = np.repeat(c, self._anchors, axis=0)
            with self._centers_lock:
                self._centers[key] = c
        return c

    def detect(self, img: np.ndarray, size: int, thresh: float = 0.5):
        """Find faces. Returns boxes (n,4), scores (n,), landmarks (n,5,2) in `img` coordinates."""
        h, w = img.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        pw, ph = -(-nw // 32) * 32, -(-nh // 32) * 32  # network wants multiples of 32
        canvas = np.zeros((ph, pw, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 128.0, (pw, ph), (127.5, 127.5, 127.5), swapRB=True)
        outs = self._det.run(self._det_out, {self._det_in: blob})

        all_boxes, all_scores, all_kps = [], [], []
        for i, stride in enumerate(self._strides):
            scores = outs[i].reshape(-1)
            keep = scores >= thresh
            if not keep.any():
                continue
            centers = self._anchor_centers(ph // stride, pw // stride, stride)[keep]
            dist = outs[i + self._fmc].reshape(-1, 4)[keep] * stride
            kps = outs[i + 2 * self._fmc].reshape(-1, 10)[keep] * stride
            all_boxes.append(np.column_stack([centers - dist[:, :2], centers + dist[:, 2:]]))
            all_kps.append((kps.reshape(-1, 5, 2) + centers[:, None, :]))
            all_scores.append(scores[keep])
        if not all_boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros((0, 5, 2), np.float32)

        boxes, scores, kps = np.concatenate(all_boxes), np.concatenate(all_scores), np.concatenate(all_kps)
        keep = _nms(boxes, scores, 0.4)
        return boxes[keep] / scale, scores[keep], kps[keep] / scale

    # ---- embeddings ------------------------------------------------------------------------
    def embed(self, img: np.ndarray, kps: np.ndarray) -> np.ndarray:
        crops = [cv2.warpAffine(img, _umeyama(k.astype(np.float32), _ARCFACE_DST), (112, 112), borderValue=0) for k in kps]
        blob = cv2.dnn.blobFromImages(crops, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        out = np.concatenate([self._rec.run(None, {self._rec_in: blob[i:i + 32]})[0] for i in range(0, len(blob), 32)])
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def analyze(self, img: np.ndarray, size: int, min_face: int = 20, max_faces: int = 64) -> Faces:
        """Detect every face in the image and return an embedding for each."""
        boxes, scores, kps = self.detect(img, size)
        if len(boxes):
            big = ((boxes[:, 2] - boxes[:, 0]) >= min_face) & ((boxes[:, 3] - boxes[:, 1]) >= min_face)
            boxes, scores, kps = boxes[big], scores[big], kps[big]
        if len(boxes) > max_faces:  # crowd shots: keep the largest faces
            order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))[:max_faces]
            boxes, scores, kps = boxes[order], scores[order], kps[order]
        embs = self.embed(img, kps) if len(boxes) else np.zeros((0, 512), np.float32)
        return Faces(boxes, scores, embs.astype(np.float32))


def _nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order, keep = scores.argsort()[::-1], []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1, yy1 = np.maximum(x1[i], x1[order[1:]]), np.maximum(y1[i], y1[order[1:]])
        xx2, yy2 = np.minimum(x2[i], x2[order[1:]]), np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1 + 1) * np.maximum(0, yy2 - yy1 + 1)
        order = order[1:][inter / (areas[i] + areas[order[1:]] - inter) <= thresh]
    return keep
