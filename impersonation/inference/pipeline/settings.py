"""Paths and switches. Everything is relative to the inference folder, and can be overridden with environment variables."""

import os
from pathlib import Path

# Settings can also come from a git-ignored file `inference/.env` (lines like KEY=value). Variables already set in the environment win.
_env = Path(__file__).resolve().parents[1] / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# ROOT = the inference folder (this file is inference/pipeline/settings.py)
ROOT = Path(__file__).resolve().parents[1]

# MODEL_FILE = the frozen LightGBM model + thresholds (the only thing stored locally; everything about the clients comes from RDS)
MODEL_FILE = Path(os.environ.get("MODEL_FILE", ROOT / "model" / "frozen.joblib"))

# DEVICE = "cuda" if a GPU is available, else "cpu" (set DEVICE=cpu to force CPU)
DEVICE = os.environ.get("DEVICE", "auto")

# LUNA_ENABLED = 0 turns the LLM step off (unsure pages then keep the ML model's best guess)
LUNA_ENABLED = os.environ.get("LUNA_ENABLED", "1") not in ("0", "false", "False")

# Luna model name, per-call timeout (seconds)
LUNA_MODEL = os.environ.get("LUNA_MODEL", "gpt-6-luna")
LUNA_TIMEOUT_S = float(os.environ.get("LUNA_TIMEOUT_S", "60"))

# Largest accepted upload, in megabytes
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))

# The screenshots the parser was built for are 900 x 900 pixels
SHOT_SIZE = 900

# ---- RDS: where the client profiles, logos, executive photos and embeddings are fetched from.
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

# A client's data is cached in memory this long (seconds) before it is fetched from RDS again
CLIENT_CACHE_TTL_S = float(os.environ.get("CLIENT_CACHE_TTL_S", "300"))
