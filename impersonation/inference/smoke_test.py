"""Smoke test for a running server: lists the clients and sends each sample screenshot to /predict.
Usage: python smoke_test.py [http://localhost:8000]      (uses only the standard library)"""

import json
import sys
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HERE = Path(__file__).resolve().parent


def post(path, fields, filename, data):
    """Send a multipart/form-data POST (a file plus form fields) and return the parsed JSON answer."""
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode()
    body += data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req))


print("health:", json.load(urllib.request.urlopen(BASE + "/health")))
for s in json.load(open(HERE / "samples" / "samples.json")):
    r = post("/predict", {"client": s["client"]}, s["file"], (HERE / "samples" / s["file"]).read_bytes())
    ok = "OK " if r["prediction"] == s["truth"] else "BAD"
    print(ok, s["type"], "->", r["prediction"], "| decided by", r["decided_by"], "| rule", r["rule_score"], "| p_ml", r["ml_probability"], "|", r["timings_ms"]["total"], "ms")
