"""Score the detector's predictions.jsonl against the case-code ground truth.

Three classes: suspicious_domain_registration / domains_parked / none. The 181 questionable rows (A1, A2) are left
out. Train and val are printed; test only with --test (look at it once, at
the end).

    python code/03_evaluate.py          # train + val
    python code/03_evaluate.py --test   # also test
"""
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORLD = HERE.parent / "data"
OUT = HERE.parent / "outputs" / "02_train_and_run"      # --out DIR reads and writes somewhere else
CLASSES = ["suspicious_domain_registration", "domains_parked", "none"]
SHORT = {"suspicious_domain_registration": "Suspicious", "domains_parked": "Parked", "none": "None"}
TYPES = {"Suspicious Domain Registration": "suspicious_domain_registration", "Domains - Parked": "domains_parked"}


def truth_of(r):
    return TYPES[r["incident_type"]] if r["label"] == "relevant" else "none"


def score(pairs):
    """pairs: [(truth, predicted)] -> accuracy, per-class P/R/F1, alert-level P/R/F1, confusion."""
    conf = Counter(pairs)
    out = {"rows": len(pairs), "accuracy": round(sum(t == p for t, p in pairs) / max(len(pairs), 1), 4)}
    for c in CLASSES:
        tp = conf[(c, c)]
        pp, ap = sum(v for (t, p), v in conf.items() if p == c), sum(v for (t, p), v in conf.items() if t == c)
        pr, rc = (tp / pp if pp else 0.0), (tp / ap if ap else 0.0)
        out[c] = {"precision": round(pr, 4), "recall": round(rc, 4),
                  "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}
    alert = lambda c: c != "none"
    tp = sum(alert(t) and alert(p) for t, p in pairs)
    fp = sum(not alert(t) and alert(p) for t, p in pairs)
    fn = sum(alert(t) and not alert(p) for t, p in pairs)
    pr, rc = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
    out["alert"] = {"precision": round(pr, 4), "recall": round(rc, 4),
                    "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}
    out["confusion"] = {f"{t} -> {p}": v for (t, p), v in sorted(conf.items())}
    return out, conf


def print_split(name, res, conf):
    print(f"\n{name}: {res['rows']} rows, accuracy {res['accuracy']:.3f}, alert P {res['alert']['precision']:.3f} "
          f"R {res['alert']['recall']:.3f} F1 {res['alert']['f1']:.3f}")
    print(f"  {'truth \\ predicted':22}" + "".join(f"{SHORT[c]:>12}" for c in CLASSES))
    for t in CLASSES:
        print(f"  {SHORT[t]:22}" + "".join(f"{conf[(t, p)]:>12}" for p in CLASSES))
    print("  " + "  ".join(f"{SHORT[c]} P {res[c]['precision']:.3f} R {res[c]['recall']:.3f} F1 {res[c]['f1']:.3f}"
                           for c in CLASSES))


def main(out=None):
    out = Path(out) if out else OUT
    rows = {r["id"]: r for r in json.loads((WORLD / "domains.json").read_text())["rows"]}
    preds = [json.loads(line) for line in (out / "predictions.jsonl").open()]
    kept = [(rows[p["id"]], p["choice"]) for p in preds
            if rows[p["id"]]["case"] not in ("A1", "A2") and rows[p["id"]]["expected"].get("scored", True)]
    splits = ["train", "val", "train+val"] + (["test"] if "--test" in sys.argv else [])
    report = {}
    for s in splits:
        members = {"train+val": ("train", "val")}.get(s, (s,))
        res, conf = score([(truth_of(r), p) for r, p in kept if r["split"] in members])
        report[s] = res
        print_split(s, res, conf)
    (out / "eval.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    a = sys.argv
    main(a[a.index("--out") + 1] if "--out" in a else None)
