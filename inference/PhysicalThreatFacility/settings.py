"""Paths and switches. Everything is relative to this incident type's folder, and can be overridden with environment variables."""

import os
from pathlib import Path

# ROOT = this incident type's folder (inference/PhysicalThreatFacility/); its model/ folder is here
ROOT = Path(__file__).resolve().parent

# Settings can also come from a git-ignored file `inference/.env` (lines like KEY=value). Variables already set in the environment win.
_env = ROOT.parent / ".env"   # inference/.env, shared by all incident types
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))

# The two local models (Rung 1: where? Rung 2: threat or not). Rung 3 is the GPT-6 Luna API.
LOCATION_MODEL_DIR = Path(os.environ.get("LOCATION_MODEL_DIR", ROOT / "model" / "rung1_location"))
THREAT_MODEL_DIR = Path(os.environ.get("THREAT_MODEL_DIR", ROOT / "model" / "rung2_distilbert"))

# DEVICE = "cuda" if a GPU is available, else "cpu" (set DEVICE=cpu to force CPU)
DEVICE = os.environ.get("DEVICE", "auto")

# LUNA_ENABLED = 0 turns Rung 3 off (unsure messages then use Rung 2's best guess)
LUNA_ENABLED = os.environ.get("LUNA_ENABLED", "1") not in ("0", "false", "False")

# Longest accepted message, in characters (the models read the first 3,000 / 128 tokens; Luna the first 4,000 characters)
MAX_TEXT_CHARS = int(os.environ.get("MAX_TEXT_CHARS", "20000"))
