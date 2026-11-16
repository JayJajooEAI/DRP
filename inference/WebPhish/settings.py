"""Paths and switches. Everything is relative to this incident type's folder, and can be overridden with environment variables."""

import os
from pathlib import Path

# ROOT = this incident type's folder (inference/WebPhish/); its model/ folder is here
ROOT = Path(__file__).resolve().parent

# Settings can also come from a git-ignored file `inference/.env` (lines like KEY=value). Variables already set in the environment win.
_env = ROOT.parent / ".env"   # inference/.env, shared by both incident types
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# The models stored locally (everything about the clients and the pages comes from RDS)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model"))
CHECK_DIRS = {2: MODEL_DIR / "check2", 3: MODEL_DIR / "check3"}   # the fine-tuned mmBERT models (branded / data capture)
PHISHPEDIA_DIR = MODEL_DIR / "phishpedia"                         # logo detector (Faster R-CNN) and logo matcher (siamese BiT)
URL_MODEL = MODEL_DIR / "url_model" / "url_model.joblib"          # suspicious-URL model (LightGBM)

# DEVICE = "cuda" if a GPU is available, else "cpu" (set DEVICE=cpu to force CPU)
DEVICE = os.environ.get("DEVICE", "auto")

# HTML larger than this is cut before parsing (same limit as the batch pipeline)
MAX_HTML_BYTES = 5_000_000

# A model's answer counts as "branded" when its probability is at least this (the same as taking the larger of the two)
BRANDED_THRESHOLD = 0.5

# ---- RDS: where the clients, the logos and the stored page captures (HTML + screenshot) are fetched from.
# The password is only ever read from the environment (never written in code or files).
RDS = dict(
    host=os.environ.get("RDS_HOST"),
    port=int(os.environ.get("RDS_PORT", "5432")),
    dbname=os.environ.get("RDS_DBNAME", "customerdata"),
    user=os.environ.get("RDS_USER"),
    password=os.environ.get("RDS_PASSWORD"),
    sslmode=os.environ.get("RDS_SSLMODE", "require"),
)
RDS_SCHEMA = os.environ.get("RDS_SCHEMA", "drp_clients")
