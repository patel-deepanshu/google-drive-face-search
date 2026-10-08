"""Start Face Search and open it in the browser:  python run.py"""
import os
import threading
import webbrowser

import uvicorn

from app.main import app

HOST = os.getenv("FACE_HOST", "127.0.0.1")  # keep 127.0.0.1: the page accepts Google credentials
PORT = int(os.getenv("FACE_PORT", 8000))

if __name__ == "__main__":
    url = f"http://{HOST}:{PORT}"
    print(f"Face Search is starting at {url}  (Ctrl+C to stop)", flush=True)
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
