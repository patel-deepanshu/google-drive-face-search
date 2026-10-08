# Face Search

Give it a few photos of one person and a Google Drive folder. It scans every image in the folder and
saves only the photos that person appears in, into `output/<Name>/`.

## Run it

Windows: double-click **`run.bat`** (first run creates the `venv` and installs the packages), or:

```
venv\Scripts\python run.py
```

The page opens at http://127.0.0.1:8000. The first start downloads the face model (about 14 MB, once).

## Use it

1. Type the person's **name** and add one or more **photos of their face**. Several photos (different
   angles, lighting, ages) find more. If a photo has several faces, the largest is used.
2. Paste the **Drive folder link** and give credentials (below).
3. Click **Find photos**. Matches appear as they are found. Everything is saved in
   `output/<Name>/` and can be downloaded as a ZIP. Stop is safe: photos found so far stay saved.

Searching the same folder again for **another person is nearly instant**: scanned photos are remembered
(`data/embeddings.sqlite3`), so nothing is downloaded or analysed twice.

### Google credentials

**Service account (works for private folders, recommended)**

1. Google Cloud Console: create a project and enable the **Google Drive API**.
2. IAM & Admin > Service Accounts > create one > **Keys > Add key > JSON**. A `.json` file downloads.
3. In Google Drive, **share the folder with the service account's email** (`client_email` in the file) as Viewer.
4. In the page, **drop the `.json` file** (or paste its contents). The page reads it and fills in the
   fields. It copes with the usual copy-paste damage: smart quotes, trailing commas, line breaks inside
   the key, a key on one line. You can also type the fields by hand.

**API key (only for folders shared as "Anyone with the link")**

Create an API key in Google Cloud (Credentials > Create credentials > API key), enable the Drive API for
that project, and use the *API key* tab. Don't add an HTTP-referrer restriction; requests come from this server.

## Why it is fast

- **Resized copies for scanning.** Drive serves a resized version of every image; the scan downloads
  that (about 100 KB) instead of the multi-MB original. Only matches are downloaded in full resolution.
- **Parallel everything.** 16 downloads in flight while every CPU core analyses an image.
- **Small, accurate models.** InsightFace's SCRFD detector and ArcFace embeddings through ONNX Runtime
  (no PyTorch, no compiler needed).
- **Remembered results** (see above), and the folder listing is paged and parallel across subfolders.

Measured on a 2-core i3 laptop: roughly 10-20 photos per second after listing. On Google Drive the
network is usually the limit.

## Settings

On the page: **Match strictness** (lower finds more but with more wrong people; higher is stricter;
0.40 is a good start) and **Speed** (Fastest / Balanced / Most thorough: the face detector's working
size. Higher finds small faces in group photos but is slower).

Environment variables (optional):

| Variable | Default | Meaning |
|---|---|---|
| `FACE_MODEL` | `buffalo_sc` | `buffalo_l` is more accurate but about 10x slower on CPU (and a ~275 MB one-time download) |
| `FACE_PORT` / `FACE_HOST` | `8000` / `127.0.0.1` | Keep the host as is: the page accepts Google credentials |
| `FACE_WORKERS` | CPU count | Images analysed at once |
| `FACE_DL_THREADS` | `16` | Parallel Drive downloads |

## Notes

- **Review the results.** Face matching is not perfect (sunglasses, masks, very small or blurry faces,
  look-alikes). Raise the strictness if you see wrong people; lower it if people are missed.
- Skipped: non-image files, GIF/SVG, Drive shortcuts, and images Drive can't make a preview of
  *and* that OpenCV can't read (for example HEIC with no preview). They show up under "Skipped".
- Credentials are used only in memory for the running search and are never written to disk. Photos and
  face data stay on this computer. Faces are biometric data: only search for people who agreed to it.
- Folder layout: `output/<Name>/` photos, `data/thumbs/` page previews, `data/embeddings.sqlite3`
  scan memory, `models/` face models. Delete any of them freely.
- Tests: `venv\Scripts\python -m unittest discover tests`.

## Layout

```
app/main.py      web server + API          static/   the page (index.html, app.js, creds.js, style.css)
app/drive.py     Google Drive client       app/faces.py      face detection + embeddings
app/pipeline.py  the search job            app/cache.py      scan memory
```
