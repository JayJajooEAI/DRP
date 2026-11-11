"""Run the FROZEN pipeline once on the test split. Nothing here is tuned: the code and model are checked against the checksums written by 05.
Usage: python 08_run_test.py   (needs 01, 02, 04 run on test first)"""

import json, joblib, pandas as pd, numpy as np
from utilities.config import *
from utilities.cascade import *
from utilities.luna import decide_many

# Safety check: stop immediately if any pipeline code or the saved model changed since step 05 froze it
check_manifest()

# F = the frozen model + thresholds saved by step 05; art = just the thresholds
F = joblib.load(f"{WORK}/frozen.joblib")
art = F["art"]

# Score the test signals with the frozen rules + model, decide the stage of each page, and save the table
te = cascade(apply(load("test"), F), art)
te.to_csv(f"{WORK}/test_stage.csv", index=False)

# Show how the pages split between stages and how many go to Luna
print(
    "stage share",
    te.stage.value_counts(normalize=True).round(3).to_dict(),
    "| to Luna:",
    int((te.stage == "llm").sum()),
    flush=True,
)

# Ask Luna about the unsure test pages and save its answers
res = decide_many(te[te.stage == "llm"], "test")
json.dump(res, open(f"{WORK}/luna_test.json", "w"))
print(
    "Luna: cost $%.4f, cache hits %d, errors %d"
    % (
        sum(o["cost"] for o in res.values() if not o["cache_hit"]),
        sum(o["cache_hit"] for o in res.values()),
        sum(o["error"] is not None for o in res.values()),
    )
)

# d = final predictions for test; y = true labels (used only for scoring)
d = combine(f"{WORK}/test_stage.csv", f"{WORK}/luna_test.json")
y = d.label.values

# Accuracy of the pages decided before Luna, then of each stage on its own pages
pre = d[d.stage != "llm"]
print("before-Luna decided:", metrics(pre.label.values, pre.pred.values))
for st in ("rules", "ml", "llm"):
    g = d[d.stage == st]
    print(f"  {st}: n={len(g)} acc={(g.pred==g.label).mean():.4f}")

# Final test metrics (the headline number)
M = metrics(y, d.pred.values)
print("FINAL TEST", M)

# Save one prediction row per page
d.to_csv(
    f"{REPORTS}/test_predictions.csv",
    index=False,
    columns=["id", "customer", "concept", "tier", "label", "stage", "rule_score", "p_ml", "pred"],
)

# Accuracy per page type (concept): count, true label, accuracy, errors
rows = [
    dict(
        concept=c,
        n=len(g),
        label=int(g.label.iloc[0]),
        accuracy=round((g.pred == g.label).mean(), 3),
        errors=int((g.pred != g.label).sum()),
    )
    for c, g in d.groupby("concept")
]
pd.DataFrame(rows).to_csv(f"{REPORTS}/test_by_concept.csv", index=False)

# Print the wrong pages and save the metrics
print("errors:", d[d.pred != d.label][["id", "concept", "label", "stage", "p_ml"]].to_string(index=False))
json.dump(M, open(f"{REPORTS}/test_metrics.json", "w"), indent=1)
