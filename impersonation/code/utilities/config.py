"""Paths and the client-side reference data (what a client would supply: brand, domain, logo, protected executives with photos).

Layout of the project (all paths are relative to the project root, which is found from this file):
    data/      screenshots, train/val/test csv, classes.json, assets (logos, wordmarks, AI faces), reference (client homepages)
    code/      numbered scripts 01..09 in pipeline order + utilities/
    outputs/   everything the scripts write (created on first run): work/, reports/, cache/
Set IMPERSONATION_OUT to write the outputs somewhere else."""

import json, csv, re, os
from pathlib import Path

# ROOT = the project folder (two levels above this file: utilities -> code -> project)
ROOT = Path(__file__).resolve().parents[2]

# DATA = the data folder
DATA = str(ROOT / "data")

# OUT = where results are written: the env var IMPERSONATION_OUT if set, else <project>/outputs
OUT = Path(os.environ.get("IMPERSONATION_OUT", ROOT / "outputs"))

# WORK = intermediate files (OCR, signals, model); REPORTS = final tables/pages; CACHE_DIR = cached Luna answers
WORK, REPORTS, CACHE_DIR = str(OUT / "work"), str(OUT / "reports"), OUT / "cache"

# Create the work and reports folders if they do not exist yet
for _p in (WORK, REPORTS):
    os.makedirs(_p, exist_ok=True)

# SHOT = folder with the screenshots (<id>.png)
SHOT = f"{DATA}/screenshots"

# _cls = the list of clients (brand, domain, protected executives) from classes.json
_cls = json.load(open(f"{DATA}/classes.json"))["customers"]

# FACEMAP = executive name -> face image file
FACEMAP = json.load(open(f"{DATA}/assets/faces/map.json"))


def key(c):
    return re.sub(r"[^A-Za-z0-9]+", "_", c)  # client name -> safe folder name (spaces/symbols become "_")


def slug(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())  # lowercase letters+digits only


# CLIENT[name] = everything the client would give us: brand, official domain, logo + wordmark image paths, and each protected executive (name, role, photo path or None)
CLIENT = {}
for c, v in _cls.items():
    k = key(c)
    CLIENT[c] = dict(
        brand=v["brand"],
        domain=v["domain"],
        logo=f"{DATA}/assets/{k}/avatar.png",
        wordmark=f"{DATA}/assets/{k}/wordmark.png",
        execs=[
            dict(
                name=e["name"],
                role=e["role"],
                photo=(f"{DATA}/assets/faces/{FACEMAP[e['name']]}" if e["name"] in FACEMAP else None),
            )
            for e in v["protected_executives"]
        ],
    )


def load_split(s):
    return list(
        csv.DictReader(open(f"{DATA}/{s}.csv"))
    )  # read train / val / test csv as a list of dicts (one per page)


# The pipeline reads only: id, customer (to pick the client profile) and the screenshot. Every other CSV column is ground truth, used only to score the result.
