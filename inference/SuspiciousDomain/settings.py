"""Paths and switches. Everything is relative to this incident type's folder, and can be overridden with environment variables."""

import os
from pathlib import Path

# ROOT = this incident type's folder (inference/SuspiciousDomain/); its model/ folder is here
ROOT = Path(__file__).resolve().parent

# Settings can also come from a git-ignored file `inference/.env` (lines like KEY=value). Variables already set in the environment win.
_env = ROOT.parent / ".env"   # inference/.env, shared by all incident types
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# The two small LightGBM models and their thresholds (everything about the clients and the domains comes from RDS)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "model"))
STRING_MODEL = MODEL_DIR / "string_model.txt"    # Step 4: unclear name matches
ALERT_MODEL = MODEL_DIR / "alert_model.txt"      # Step 7: 'brand + extra word' alerts
THRESHOLDS = MODEL_DIR / "thresholds.json"

# ---- RDS: where the clients and the stored domain lookups are read from. The password is only ever read from the environment.
RDS = dict(
    host=os.environ.get("RDS_HOST"),
    port=int(os.environ.get("RDS_PORT", "5432")),
    dbname=os.environ.get("RDS_DBNAME", "customerdata"),
    user=os.environ.get("RDS_USER"),
    password=os.environ.get("RDS_PASSWORD"),
    sslmode=os.environ.get("RDS_SSLMODE", "require"),
)
RDS_SCHEMA = os.environ.get("RDS_SCHEMA", "drp_clients")
