"""The client registry: what each protected client supplies (brand, official domain, logo, protected executives with photos).

Layout of the clients folder:
    clients/<client_id>/client.json      {"id", "name", "brand", "domain", "executives": [{"name", "role", "photo"}]}
    clients/<client_id>/logo.png         the client's avatar / logo
    clients/<client_id>/executives/*.jpg executive photos (paths are listed in client.json)
To protect a new client, add a folder like this and restart the service."""

import json
from pathlib import Path

from pipeline.settings import CLIENTS_DIR


def load_clients(root=CLIENTS_DIR):
    """Read every client folder -> {client_id: profile}. The profile has the keys the signals code expects: brand, domain, logo, execs."""
    out = {}
    for d in sorted(Path(root).iterdir()):
        f = d / "client.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())

        # execs = protected executives; photo is an absolute path, or None when no photo is available
        execs = [
            dict(name=e["name"], role=e["role"], photo=(str(d / e["photo"]) if e.get("photo") else None))
            for e in j.get("executives", [])
        ]
        out[j["id"]] = dict(
            id=j["id"], name=j["name"], brand=j["brand"], domain=j["domain"], logo=str(d / "logo.png"), execs=execs
        )
    return out
