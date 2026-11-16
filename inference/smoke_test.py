"""Smoke test for a running server: checks /health, then sends each sample to every incident type that is loaded.
Usage: python smoke_test.py [http://localhost:8000]      (uses only the standard library)"""

import json
import sys
import urllib.request
import uuid
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
HERE = Path(__file__).resolve().parent


def post_json(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def post_file(path, fields, filename, data):
    """Send a multipart/form-data POST (a file plus form fields) and return the parsed JSON answer."""
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode()
    body += data + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req))


health = json.load(urllib.request.urlopen(BASE + "/health"))
print("health:", health["status"], {k: v["loaded"] for k, v in health["incident_types"].items()})

if health["incident_types"].get("impersonation", {}).get("loaded"):
    print("\nimpersonation")
    d = HERE / "impersonation" / "samples"
    for s in json.load(open(d / "samples.json")):
        r = post_file("/impersonation/predict", {"client": s["client"]}, s["file"], (d / s["file"]).read_bytes())
        ok = "OK " if r["prediction"] == s["truth"] else "BAD"
        print(" ", ok, s["type"], "->", r["prediction"], "| decided by", r["decided_by"], "|", r["timings_ms"]["total"], "ms")

if health["incident_types"].get("WebPhish", {}).get("loaded"):
    print("\nwebphish")
    for s in json.load(open(HERE / "WebPhish" / "samples" / "samples.json")):
        r = post_json("/webphish/predict", {"client": s["client"], "url": s["url"]})
        print(" ", s["client"], s["url"][:55], "-> records:", r["records"], "| web_phish records:", r["web_phish_records"])
        for x in r["results"]:
            ok = "OK " if x["prediction"] == s["truth"] else "BAD"
            print("    ", ok, "content_id", x["capture"]["content_id"], "->", x["prediction"], "| case",
                  x["checks"]["data_capture"]["case"], "|", x["timings_ms"]["total"], "ms")

if health["incident_types"].get("PhysicalThreatFacility", {}).get("loaded"):
    print("\nphysicalthreat")
    for s in json.load(open(HERE / "PhysicalThreatFacility" / "samples" / "samples.json")):
        r = post_json("/physicalthreat/predict", {"text": s["text"]})
        ok = "OK " if r["prediction"] == s["truth"] else "BAD"
        print(" ", ok, s["expected_rung"], "->", r["prediction"], "| decided by", r["decided_by"], "| p_threat", r["p_threat"], "|", r["timings_ms"]["total"], "ms")
