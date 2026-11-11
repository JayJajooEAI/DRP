"""Paths and switches. Everything is relative to the inference folder, and can be overridden with environment variables."""

import os
from pathlib import Path

# ROOT = the inference folder (this file is inference/pipeline/settings.py)
ROOT = Path(__file__).resolve().parents[1]

# MODEL_FILE = the frozen LightGBM model + thresholds; CLIENTS_DIR = one sub-folder per protected client
MODEL_FILE = Path(os.environ.get("MODEL_FILE", ROOT / "model" / "frozen.joblib"))
CLIENTS_DIR = Path(os.environ.get("CLIENTS_DIR", ROOT / "clients"))

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
