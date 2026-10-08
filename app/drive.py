"""Minimal Google Drive client: list the images in a shared folder and download them.

Talks to the Drive v3 REST API directly (no google-api-python-client) so that many downloads
can safely run in parallel threads.

Speed trick: to *find* faces we don't download the full photo. Drive can serve a resized copy of
any image (its `thumbnailLink`), which is 10-50x smaller. Only photos that match are downloaded in
full resolution.
"""
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import requests
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from requests.adapters import HTTPAdapter

API = "https://www.googleapis.com/drive/v3"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
FOLDER_MIME = "application/vnd.google-apps.folder"
SKIP_MIME = {"image/svg+xml", "image/gif", "image/x-icon", "image/vnd.microsoft.icon"}
RETRY_REASONS = {"rateLimitExceeded", "userRateLimitExceeded", "sharingRateLimitExceeded", "backendError"}
MAX_FULL_SCAN_BYTES = 60 * 1024 * 1024  # fallback path (no thumbnail): don't pull absurdly large files


class DriveError(Exception):
    """Anything that should be shown to the user as a plain message."""


@dataclass
class DriveFile:
    id: str
    name: str
    mime: str
    size: int
    version: str          # changes when the file changes (used by the embedding cache)
    thumb: Optional[str]  # short-lived resized-image link


def parse_folder_id(link: str) -> str:
    """Accept a full Drive folder URL (any of the common shapes) or a bare folder ID."""
    link = (link or "").strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{10,}", link):
        return link
    m = re.search(r"/folders/([A-Za-z0-9_-]{10,})", link) or re.search(r"[?&]id=([A-Za-z0-9_-]{10,})", link)
    if not m:
        raise DriveError("Couldn't find a folder ID in that link. Paste the folder's share link "
                         "(https://drive.google.com/drive/folders/…).")
    return m.group(1)


def normalize_service_account(info: dict) -> dict:
    """Validate a service-account credential and fill in defaults."""
    if not isinstance(info, dict):
        raise DriveError("Service account credentials must be a JSON object.")
    info = {k: v.strip() if isinstance(v, str) else v for k, v in info.items()}
    info["private_key"] = (info.get("private_key") or "").replace("\\n", "\n")
    missing = [k for k in ("client_email", "private_key") if not info.get(k)]
    if missing:
        raise DriveError(f"Service account credentials are missing: {', '.join(missing)}.")
    if "BEGIN PRIVATE KEY" not in info["private_key"]:
        raise DriveError("The private_key doesn't look right: it should start with -----BEGIN PRIVATE KEY-----.")
    info.setdefault("type", "service_account")
    info.setdefault("token_uri", "https://oauth2.googleapis.com/token")
    return info


def _api_error(r: requests.Response) -> tuple:
    """(reason, message) from a Google API error response."""
    try:
        err = r.json()["error"]
        reason = (err.get("errors") or [{}])[0].get("reason", "")
        return reason, err.get("message") or f"HTTP {r.status_code}"
    except Exception:
        return "", f"HTTP {r.status_code}"


class DriveClient:
    def __init__(self, service_account_info: Optional[dict] = None, api_key: Optional[str] = None,
                 api_base: str = API, pool_size: int = 32):
        self.api = api_base.rstrip("/")
        self.params = {}
        if service_account_info:
            info = normalize_service_account(service_account_info)
            try:
                creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
            except (ValueError, KeyError) as e:
                raise DriveError(f"The service account credentials can't be read ({e}). "
                                 "Check client_email and private_key.") from None
            self.session = AuthorizedSession(creds)
            self.who = info["client_email"]
            self.share_hint = f"the folder is shared with {self.who}"
        elif api_key:
            self.session = requests.Session()
            self.params = {"key": api_key.strip()}
            self.who = "the API key"
            self.share_hint = "the folder is shared as 'Anyone with the link'"
        else:
            raise DriveError("No Google credentials were provided.")
        for scheme in ("https://", "http://"):
            self.session.mount(scheme, HTTPAdapter(pool_connections=4, pool_maxsize=pool_size))
        self._api_host = urlparse(self.api).hostname
        self._thumbs_ok = True
        self._thumb_failures = 0

    # ---- low level -------------------------------------------------------------------------
    def _request(self, url: str, params: Optional[dict] = None, stream: bool = False, retries: int = 4):
        """GET with retries on rate limits and transient errors."""
        for attempt in range(retries + 1):
            last = attempt == retries
            try:
                r = self.session.get(url, params={**self.params, **(params or {})}, stream=stream, timeout=(10, 60))
            except GoogleAuthError as e:
                raise DriveError(f"Google rejected the service account credentials: {e}. "
                                 "(Also check that this computer's clock is correct.)") from None
            except requests.RequestException as e:
                if last:
                    raise DriveError(f"Network problem talking to Google Drive: {e}") from None
                time.sleep(2 ** attempt + random.random())
                continue
            retryable = r.status_code in (429, 500, 502, 503, 504)
            if r.status_code == 403 and _api_error(r)[0] in RETRY_REASONS:
                retryable = True
            if retryable and not last:
                wait = float(r.headers.get("Retry-After") or 2 ** attempt) + random.random()
                r.close()
                time.sleep(min(wait, 30))
                continue
            return r

    def _trusted(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return host == self._api_host or host.endswith((".googleusercontent.com", ".google.com", ".googleapis.com"))

    @staticmethod
    def _sized(thumb: str, size: int) -> str:
        return re.sub(r"=s\d+(-\w+)*$", "", thumb) + f"=s{size}"

    # ---- folder access ---------------------------------------------------------------------
    def get_folder_name(self, folder_id: str) -> str:
        """Check we can see the folder; returns its name. Gives a clear error otherwise."""
        r = self._request(f"{self.api}/files/{folder_id}", {"fields": "id,name,mimeType", "supportsAllDrives": "true"})
        if r.status_code == 404:
            raise DriveError(f"Folder not found. Check the link, and make sure {self.share_hint}.")
        if r.status_code != 200:
            raise DriveError(f"Google Drive refused access: {_api_error(r)[1]}")
        meta = r.json()
        if meta.get("mimeType") != FOLDER_MIME:
            raise DriveError("That link points to a file, not a folder.")
        return meta["name"]

    def _list_children(self, folder_id: str) -> tuple:
        q = (f"'{folder_id}' in parents and trashed = false and "
             f"(mimeType contains 'image/' or mimeType = '{FOLDER_MIME}')")
        params = {
            "q": q, "pageSize": "1000", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            "fields": "nextPageToken,files(id,name,mimeType,size,md5Checksum,modifiedTime,thumbnailLink)",
        }
        folders, images = [], []
        while True:
            r = self._request(f"{self.api}/files", params)
            if r.status_code != 200:
                raise DriveError(f"Couldn't list the folder: {_api_error(r)[1]}")
            body = r.json()
            for f in body.get("files", []):
                if f["mimeType"] == FOLDER_MIME:
                    folders.append(f["id"])
                elif f["mimeType"] not in SKIP_MIME:
                    size = int(f.get("size") or 0)
                    version = f.get("md5Checksum") or f"{f.get('modifiedTime')}:{size}"
                    images.append(DriveFile(f["id"], f["name"], f["mimeType"], size, version, f.get("thumbnailLink")))
            if not body.get("nextPageToken"):
                return folders, images
            params["pageToken"] = body["nextPageToken"]

    def list_images(self, folder_id: str, recursive: bool = True,
                    on_progress: Callable[[int], None] = lambda n: None,
                    cancelled: Callable[[], bool] = lambda: False) -> list:
        """All images in the folder (and its subfolders), listed level by level in parallel."""
        images, seen, level = {}, {folder_id}, [folder_id]
        with ThreadPoolExecutor(8) as pool:
            while level and not cancelled():
                nxt = []
                for folders, imgs in pool.map(self._list_children, level):
                    images.update({f.id: f for f in imgs})
                    nxt += [f for f in folders if f not in seen] if recursive else []
                seen.update(nxt)
                level = nxt
                on_progress(len(images))
        return list(images.values())

    # ---- downloads -------------------------------------------------------------------------
    def fetch_scan(self, f: DriveFile, size: int) -> bytes:
        """A copy of the image about `size` px on its long side: a Drive thumbnail if possible."""
        if self._thumbs_ok and f.thumb and self._trusted(f.thumb):
            try:
                r = self.session.get(self._sized(f.thumb, size), timeout=(10, 60))
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                    self._thumb_failures = 0
                    return r.content
            except (requests.RequestException, GoogleAuthError):
                pass
            self._thumb_failures += 1
            if self._thumb_failures >= 8:  # thumbnails clearly aren't working; stop trying them
                self._thumbs_ok = False
        if f.size > MAX_FULL_SCAN_BYTES:
            raise DriveError(f"{f.name} is too large to scan without a thumbnail")
        r = self._request(f"{self.api}/files/{f.id}", {"alt": "media", "supportsAllDrives": "true"})
        if r.status_code != 200:
            raise DriveError(f"{f.name}: {_api_error(r)[1]}")
        return r.content

    def fetch_thumbnail(self, f: DriveFile, size: int = 480) -> Optional[bytes]:
        """Small preview for the web page. Best effort."""
        if not (f.thumb and self._trusted(f.thumb)):
            return None
        try:
            r = self.session.get(self._sized(f.thumb, size), timeout=(10, 60))
            return r.content if r.status_code == 200 else None
        except (requests.RequestException, GoogleAuthError):
            return None

    def download_to(self, f: DriveFile, dest: Path) -> None:
        """Download the original file to `dest` (written to a temp name first, then renamed)."""
        tmp = dest.parent / (dest.name + ".part")
        last = None
        for attempt in range(3):
            try:
                r = self._request(f"{self.api}/files/{f.id}", {"alt": "media", "supportsAllDrives": "true"}, stream=True)
                if r.status_code != 200:
                    raise DriveError(_api_error(r)[1])
                with r, open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 18):
                        fh.write(chunk)
                if f.size and tmp.stat().st_size != f.size:
                    raise DriveError("incomplete download")
                tmp.replace(dest)
                return
            except (requests.RequestException, DriveError, OSError) as e:
                last = e
                time.sleep(1 + attempt)
        tmp.unlink(missing_ok=True)
        raise DriveError(f"Couldn't download {f.name}: {last}")
