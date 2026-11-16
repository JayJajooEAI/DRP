"""One HTTP server for both incident types.

    uvicorn app:app --host 0.0.0.0 --port 8000

Endpoints
    GET  /health                   -> which incident types are loaded
    GET  /impersonation/clients    -> the clients that can be checked for impersonation
    POST /impersonation/predict    -> multipart form: file = the profile screenshot, client = a client id, use_luna = true/false
    GET  /webphish/clients         -> the clients that can be checked for web phishing
    POST /webphish/predict         -> JSON body: {"client": "<id>", "url": "<link>"}  (or "content_id": <n> instead of "url")

Each incident type lives in its own folder (impersonation/, WebPhish/) with its own code and models; this file only routes requests.
Set INCIDENT_TYPES=impersonation or INCIDENT_TYPES=WebPhish to serve just one. If one of them fails to load, the other keeps working
and the failing one answers 503 with the reason.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Make the incident-type packages importable no matter where uvicorn is started from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

ENGINES = {}   # incident type -> its loaded engine
ERRORS = {}    # incident type -> why it did not load
WANTED = [t.strip() for t in os.environ.get("INCIDENT_TYPES", "impersonation,WebPhish").split(",") if t.strip()]


def _make(name):
    if name == "impersonation":
        from impersonation.engine import Engine
    elif name == "WebPhish":
        from WebPhish.engine import Engine
    else:
        raise ValueError(f"unknown incident type '{name}'")
    return Engine()


@asynccontextmanager
async def lifespan(app):
    # Load every model once when the server starts (about a minute), not on the first request
    for name in WANTED:
        try:
            ENGINES[name] = _make(name)
        except Exception as e:   # keep the other incident type serving
            ERRORS[name] = f"{type(e).__name__}: {e}"
            print(f"[{name}] failed to load: {ERRORS[name]}", flush=True)
    yield
    ENGINES.clear()


app = FastAPI(title="DRP incident detectors", version="1.0", lifespan=lifespan)


def engine(name):
    e = ENGINES.get(name)
    if e is None:
        raise HTTPException(503, f"{name} is not loaded: {ERRORS.get(name, 'not in INCIDENT_TYPES')}")
    return e


@app.get("/health")
def health():
    loaded = {n: dict(loaded=True, **e.info()) for n, e in ENGINES.items()}
    failed = {n: dict(loaded=False, error=m) for n, m in ERRORS.items()}
    return dict(status="ok" if not failed else "degraded", incident_types={**loaded, **failed})


# ---------------------------------------------------------------- impersonation
imp = APIRouter(prefix="/impersonation", tags=["impersonation"])


@imp.get("/clients")
def imp_clients():
    from impersonation.clients import RdsUnavailable
    try:
        return engine("impersonation").clients.ids()
    except RdsUnavailable as e:
        raise HTTPException(503, str(e))


# A plain `def` (not async) runs in a worker thread, so a slow request does not block the server
@imp.post("/predict")
def imp_predict(file: UploadFile = File(...), client: str = Form(...), use_luna: bool = Form(True)):
    from impersonation.clients import RdsUnavailable
    from impersonation.engine import BadImage, UnknownClient
    from impersonation.settings import MAX_UPLOAD_MB
    e = engine("impersonation")
    data = file.file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file is larger than {MAX_UPLOAD_MB:g} MB")
    try:
        return e.predict(data, client, use_luna)
    except UnknownClient:
        raise HTTPException(404, f"unknown client '{client}'; see GET /impersonation/clients")
    except BadImage as err:
        raise HTTPException(400, str(err))
    except RdsUnavailable as err:
        raise HTTPException(503, str(err))


# ---------------------------------------------------------------- web phish
wp = APIRouter(prefix="/webphish", tags=["webphish"])


class WebPhishRequest(BaseModel):
    client: str
    url: Optional[str] = None
    content_id: Optional[int] = None


@wp.get("/clients")
def wp_clients():
    from WebPhish.clients import RdsUnavailable
    try:
        return engine("WebPhish").clients.ids()
    except RdsUnavailable as e:
        raise HTTPException(503, str(e))


@wp.post("/predict")
def wp_predict(req: WebPhishRequest):
    from WebPhish.clients import RdsUnavailable
    from WebPhish.engine import BadInput, NoCapture, UnknownClient
    e = engine("WebPhish")
    try:
        return e.predict(req.client, url=req.url, content_id=req.content_id)
    except UnknownClient:
        raise HTTPException(404, f"unknown client '{req.client}'; see GET /webphish/clients")
    except NoCapture:
        raise HTTPException(404, "no stored page for this link and client")
    except BadInput as err:
        raise HTTPException(400, str(err))
    except RdsUnavailable as err:
        raise HTTPException(503, str(err))


app.include_router(imp)
app.include_router(wp)
