#!/usr/bin/env python3
"""
Step 1 of training the Rung 1 location model: GLiNER (urchade/gliner_multi-v2.1) marks the places in every train and
val message. These marks are the training labels (GLiNER is the teacher, MiniLMv2 the student).

Two GLiNER passes per chunk, both kept:
  - base:  6 location labels at score >= 0.4
  - extra: the same + 5 place-type labels at >= 0.3 on a title-cased view of all-caps text, keeping only named places
           (not "the school", "a building", or lower-case words)

    pip install gliner
    python code/01_make_teacher_labels.py [--limit N] [--out-dir DIR]   -> DIR/teacher_train.jsonl, DIR/teacher_val.jsonl
    Reads data/splits/{train,val}.csv. Default DIR: outputs/01_make_teacher_labels/, so the shipped labels in data/teacher_labels/ are never overwritten.
"""
import argparse, csv, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); from location import clean, chunks
csv.field_size_limit(10**9)
GLINER = "urchade/gliner_multi-v2.1"
LOC_LABELS = ["street address", "city", "state or province", "country", "landmark or venue", "neighborhood or district"]
EXTRA = ["school or university", "building", "road or highway", "island", "region"]
GENERIC = re.compile(r"^(the |a |an |our |my |their )?(house|home|bedroom|room|building|apartment|garage|basement|bar|office|store|shop|school|"
                     r"church|car|vehicle|roof|kitchen|street|road|area|scene|residence|structure|casas|warehouse)s?$", re.I)
VENUE = {"building", "landmark or venue", "road or highway", "region", "school or university"}
def named(e): return not (e["label"] in VENUE and (GENERIC.match(e["text"].strip()) or e["text"][:1].islower()))
def caps(t):
    letters = [c for c in t if c.isalpha() and c.isascii()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > 0.6

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int); ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "01_make_teacher_labels")); a = ap.parse_args()
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    import torch; from gliner import GLiNER
    m = GLiNER.from_pretrained(GLINER).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    for split in ("train", "val"):
        rows = list(csv.DictReader(open(ROOT / f"data/splits/{split}.csv", encoding="utf-8")))[: a.limit]
        pieces = []
        for i, r in enumerate(rows):
            c = clean(r["text"]); wc = caps(c)
            pieces += [(i, ch, ch.replace("_", " ").title() if wc else ch.replace("_", " ")) for ch in chunks(c)]
        with torch.no_grad():
            base = m.inference([p[1] for p in pieces], LOC_LABELS, threshold=0.4, batch_size=16)
            ext = m.inference([p[2] for p in pieces], LOC_LABELS + EXTRA, threshold=0.3, batch_size=16)
        per = {}
        for (i, ch, _), b, x in zip(pieces, base, ext):
            spans = [(e["start"], e["end"], e["label"]) for e in b] + [(e["start"], e["end"], e["label"]) for e in x if named(e)]
            per.setdefault(i, []).append({"text": ch, "spans": sorted(set(spans))})
        with open(OUT / f"teacher_{split}.jsonl", "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                f.write(json.dumps({"id": r["id"], "label": int(r["label"]), "chunks": per.get(i, [])}, ensure_ascii=False) + "\n")
        print(split, len(rows), "messages,", len(pieces), "chunks", flush=True)
