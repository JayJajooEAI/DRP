#!/usr/bin/env python3
"""
Physical-threat pipeline: each message goes down the ladder and stops at the first rung that decides it.

    (enrichment, only if the message has a link)  ->  RUNG 1 location?  --no-->  not a threat
                                                          | yes
                                                   RUNG 2 DistilBERT  >= 0.6 threat · < 0.3 not a threat
                                                          | 0.3-0.6
                                                   RUNG 3 GPT-6 Luna (prompt v14) decides

    python code/04_run_ladder.py data/splits/test.csv                   # all rungs (Rung 3 calls the API: needs OPENAI_API_KEY)
    python code/04_run_ladder.py data/splits/test.csv --no-gpt          # Rungs 1-2 only; unsure messages are left "unsure"
    python code/04_run_ladder.py my.csv --enrich                        # also open links in the messages (needs Playwright)
    python code/04_run_ladder.py --text "Shots fired at 5th Ave, Seattle"

Input: a CSV with a "text" column (a "label" column, if present, is used to score the run).
Reads the trained models from outputs/02_train_location_model/model and outputs/03_train_threat_model/model.
Output, in outputs/04_run_ladder/:
        <input name>_pipeline.csv (one row per message: rung that decided, places, P(threat), GPT answer, final label)
        and <input name>_pipeline_summary.json (counts per rung, time, cost, and metrics when labels are given).
"""
import argparse, csv, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))              # location.py, threat.py, luna.py, enrich.py sit next to this script
OUT = ROOT / "outputs" / "04_run_ladder"
csv.field_size_limit(10**9)
GPU_USD_PER_HOUR = 0.526         # AWS g4dn.xlarge (Tesla T4), on-demand us-east-1: Rungs 1-2 are costed as GPU time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?"); ap.add_argument("--text", action="append")
    ap.add_argument("--no-gpt", action="store_true"); ap.add_argument("--enrich", action="store_true")
    ap.add_argument("--workers", type=int, default=16, help="parallel GPT-6 Luna calls")
    a = ap.parse_args()
    if a.text: rows = [{"text": t} for t in a.text]
    elif a.csv: rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    else: ap.error("give a CSV or --text")
    texts = [r["text"] for r in rows]; T = {}

    links = [[] for _ in texts]
    if a.enrich:
        from enrich import Enricher
        t0 = time.time()
        with Enricher() as e:
            for i, t in enumerate(texts):
                if e.x.has_urls(t): texts[i], links[i] = e.enrich(t)
        T["enrichment_s"] = time.time() - t0

    from location import LocationModel
    from threat import ThreatModel, decide
    import torch
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    loc_m, thr_m = LocationModel(), ThreatModel()

    sync(); t0 = time.time(); loc = loc_m.predict(texts); sync(); T["rung1_s"] = time.time() - t0
    r2_idx = [i for i, l in enumerate(loc) if l["has_location"]]
    sync(); t0 = time.time(); p = dict(zip(r2_idx, thr_m.score([texts[i] for i in r2_idx]))); sync(); T["rung2_s"] = time.time() - t0
    r3_idx = [i for i in r2_idx if decide(p[i]) == "unsure"]

    gpt = {}
    if r3_idx and not a.no_gpt:
        from luna import Luna
        L = Luna(); t0 = time.time()
        with ThreadPoolExecutor(a.workers) as pool: gpt = dict(zip(r3_idx, pool.map(lambda i: L.classify(texts[i]), r3_idx)))
        T["rung3_s"] = time.time() - t0

    out = []
    for i, r in enumerate(rows):
        o = {k: r[k] for k in r}
        o["links_opened"] = sum(l["ok"] for l in links[i]); o["has_location"] = int(loc[i]["has_location"])
        o["places"] = " | ".join(loc[i]["places"]); o["location_by"] = loc[i]["by"]
        o["p_threat"] = round(p[i], 4) if i in p else ""
        g = gpt.get(i)
        o["gpt_label"], o["gpt_subtype"], o["gpt_reason"] = (g["label"], g["subtype"], g["reason"]) if g else ("", "", "")
        if not loc[i]["has_location"]: o["decided_at"], o["prediction"] = "rung1", 0
        elif i not in r3_idx: o["decided_at"], o["prediction"] = "rung2", int(decide(p[i]) == "threat")
        elif g: o["decided_at"], o["prediction"] = "rung3", g["label"]
        else: o["decided_at"], o["prediction"] = "rung3 (not run)", ""
        out.append(o)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / (Path(a.csv).stem if a.csv else "text")
    with open(f"{stem}_pipeline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(out[0])); w.writeheader(); w.writerows(out)

    gpu = GPU_USD_PER_HOUR / 3600
    S = {"messages": len(rows), "rung1_filtered_no_location": len(rows) - len(r2_idx), "rung2_decided": len(r2_idx) - len(r3_idx),
         "rung2_threat": sum(1 for i in r2_idx if decide(p[i]) == "threat"), "rung2_not_threat": sum(1 for i in r2_idx if decide(p[i]) == "not a threat"),
         "rung3_sent": len(r3_idx), "rung3_answered": len(gpt),
         "seconds": {k: round(v, 3) for k, v in T.items()},
         "cost_usd": {"rung1_gpu": T["rung1_s"] * gpu, "rung2_gpu": T["rung2_s"] * gpu,
                      "rung3_gpt": sum(g["cost_usd"] for g in gpt.values()), "rung3_gpt_no_cache": sum(g["cost_usd_no_cache"] for g in gpt.values())},
         "gpt_usage": {"calls": len(gpt), "input_tokens": sum(g["usage"].get("prompt_tokens") or 0 for g in gpt.values()),
                       "output_tokens": sum(g["usage"].get("completion_tokens") or 0 for g in gpt.values()),
                       "request_ids": [g["meta"]["request_id"] for g in gpt.values()]}}
    if rows and "label" in rows[0]:
        y = [int(r["label"]) for r in rows]; pr = [o["prediction"] for o in out]
        done = [(t, q) for t, q in zip(y, pr) if q != ""]
        tp = sum(t == 1 and q == 1 for t, q in done); fp = sum(t == 0 and q == 1 for t, q in done)
        fn = sum(t == 1 and q == 0 for t, q in done); tn = sum(t == 0 and q == 0 for t, q in done)
        S["metrics"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "undecided": len(y) - len(done),
                        "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
                        "missed_threats_at_rung1": sum(1 for i, o in enumerate(out) if y[i] == 1 and o["decided_at"] == "rung1")}
    json.dump(S, open(f"{stem}_pipeline_summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in S.items() if k != "gpt_usage"}, indent=1))
    if a.text:
        for o in out: print(o["decided_at"], o["prediction"], o["places"], o["p_threat"], o["gpt_reason"][:120], "|", o["text"][:80])

if __name__ == "__main__":
    main()
