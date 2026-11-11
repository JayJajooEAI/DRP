"""HTTP API for the impersonation detector.

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health   -> status + the loaded model's thresholds
    GET  /clients  -> the protected clients that can be checked
    POST /predict  -> multipart form: file = the profile screenshot, client = a client id from /clients, use_luna = true/false
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make the `pipeline` package importable no matter where uvicorn is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pipeline.engine import BadImage, Engine, UnknownClient
from pipeline.settings import MAX_UPLOAD_MB

ENGINE = {}


@asynccontextmanager
async def lifespan(app):
    # Load all models once when the server starts (takes a few seconds), not on the first request
    ENGINE["e"] = Engine()
    yield
    ENGINE.clear()


app = FastAPI(title="Impersonation detector", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return dict(status="ok", **ENGINE["e"].info())


@app.get("/clients")
def clients():
    return [dict(id=c["id"], name=c["name"], brand=c["brand"], domain=c["domain"]) for c in ENGINE["e"].clients.values()]


# A plain `def` (not async) runs in a worker thread, so a slow request does not block the server
@app.post("/predict")
def predict(file: UploadFile = File(...), client: str = Form(...), use_luna: bool = Form(True)):
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB:g} MB")
    try:
        return ENGINE["e"].predict(data, client, use_luna)
    except UnknownClient:
        raise HTTPException(404, f"unknown client '{client}'; see GET /clients")
    except BadImage as e:
        raise HTTPException(400, str(e))
