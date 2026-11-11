"""Step 1d: signals per screenshot (parsed fields + avatar scores + client profile) -> outputs/work/signals_<split>.csv.
Usage: python 04_build_signals.py train val test"""

import sys, json, pandas as pd

# Pool = run the work on several CPU cores at once
from multiprocessing import Pool
from PIL import Image
from utilities.config import *

# signals() turns one page (OCR lines + avatar scores + screenshot) into ~100 numbers
from utilities.signals import signals


def work(a):
    """Compute the signals of ONE page (runs inside a worker process)."""
    # r = csv row, ol = OCR lines, av = avatar scores. Only id and customer are copied from the row; the screenshot is opened for the verified-badge check.
    r, ol, av = a
    return dict(id=r["id"], customer=r["customer"], **signals(r, ol, av, Image.open(f"{SHOT}/{r['id']}.png")))


# Needed so the worker processes do not re-run this block when they start
if __name__ == "__main__":
    # One pass per split named on the command line
    for sp in sys.argv[1:]:
        # R = rows, O = OCR output (step 01), A = avatar scores (step 02)
        R = load_split(sp)
        O = json.load(open(f"{WORK}/ocr_{sp}.json"))
        A = json.load(open(f"{WORK}/avatar_{sp}.json"))

        # Spread the pages over 8 processes (40 pages per task)
        with Pool(8) as p:
            out = p.map(work, [(r, O[r["id"]], A[r["id"]]) for r in R], chunksize=40)

        # Save one row per page; the rules and the ML model read this file
        pd.DataFrame(out).to_csv(f"{WORK}/signals_{sp}.csv", index=False)
        print(sp, len(out))
