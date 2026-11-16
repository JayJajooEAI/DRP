"""Step 1b check: how exactly the layout parser recovers each field from the OCR lines, against the CSV truth (diagnostic only; nothing downstream uses it).
Usage: python 03_check_parser_accuracy.py train val [test]"""

import json, re, sys
from utilities.config import *
from utilities.layout_parser import parse

# rapidfuzz.fuzz.ratio = text similarity 0..100 (100 = identical)
from rapidfuzz import fuzz


def sl(s):
    return re.sub(
        r"[^a-z0-9]", "", (s or "").lower()
    )  # "slug": lowercase letters+digits only, so punctuation/spaces do not matter in comparisons


# One pass per split given on the command line
for sp in sys.argv[1:]:
    # R = true rows (csv); O = OCR output; n = pages; ok = count of correct fields; bad = examples of mistakes per field
    R = load_split(sp)
    O = json.load(open(f"{WORK}/ocr_{sp}.json"))
    n = len(R)
    ok = dict(plat=0, handle=0, name=0, bio=0, link=0, fol=0, posts=0)
    bad = {"name": [], "bio": [], "link": [], "fol": []}
    for r in R:
        # Parse the page from OCR only, then check the platform
        p = parse(O[r["id"]])
        ok["plat"] += p["platform"] == r["platform"]

        # Handle read from the URL bar: correct if >= 90% similar to the true handle
        ok["handle"] += fuzz.ratio(sl(p["handle_url"]), sl(r["handle"])) >= 90

        # Display name: correct if >= 88% similar; remember the miss
        c = fuzz.ratio(sl(p["name"]), sl(r["display_name"])) >= 88
        ok["name"] += c
        if not c:
            bad["name"].append((r["id"], r["display_name"], p["name"]))

        # Bio: tb = true, pb = parsed. Correct if both are empty-ish, or >= 85% similar
        tb = sl(r["bio"])
        pb = sl(p["bio"])
        c = (not tb and len(pb) < 3) or (fuzz.ratio(tb, pb) >= 85)
        ok["bio"] += c
        if not c:
            bad["bio"].append((r["id"], r["bio"], p["bio"]))

        # Link: tl = true link without https:// and trailing slash. Correct if both empty, or >= 88% similar
        tl = r["link"].replace("https://", "").lower().strip("/")
        c = (not tl and not p["link"]) or bool(tl and fuzz.ratio(sl(tl), sl(p["link"])) >= 88)
        ok["link"] += c
        if not c:
            bad["link"].append((r["id"], r["link"], p["link"]))

        # Followers: tf = true; pf = parsed (LinkedIn shows "connections" instead). Correct if within 6% (or 1)
        tf = float(r["followers"])
        pf = p["followers"] if p["followers"] is not None else p["connections"]
        c = pf is not None and abs(pf - tf) <= max(1, tf * 0.06)
        ok["fol"] += c
        if not c:
            bad["fol"].append((r["id"], tf, pf))

        # Posts: did the parser find the same NUMBER of posts as the truth ("||" separates posts in the csv)?
        ok["posts"] += len(p["posts"]) == len([x for x in r["recent_posts"].split(" || ") if x])

    # Print accuracy per field, then up to 4 example mistakes per field
    print(sp, n, {k: round(v / n, 3) for k, v in ok.items()})
    for k, v in bad.items():
        print(k, len(v), v[:4])
