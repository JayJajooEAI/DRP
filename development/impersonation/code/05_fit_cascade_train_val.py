"""Steps 2-3: fit the rule thresholds (train) and LightGBM (train), run the cascade on train (out-of-fold) and val, and FREEZE.
Writes outputs/work/dev_{train,val}_stage.csv, outputs/work/frozen.joblib, outputs/reports/frozen_thresholds.json and frozen_manifest.json.
Uses train and val only; test is never read here.  Usage: python 05_fit_cascade_train_val.py"""

import json, numpy as np, pandas as pd

# cascade.py has fit_all, load, apply, cascade, metrics, write_manifest
from utilities.cascade import *

# Fit on TRAIN only: finds the rule thresholds and trains LightGBM. tr = train table with rule_score and out-of-fold ML probability; art = the thresholds
tr, art = fit_all()
print(art)

# va = validation signals scored by the frozen rules + model; then decide which stage answers each page (rules / ml / llm) for both tables
va = apply(load("val"))
tr = cascade(tr, art)
va = cascade(va, art)

# Report for train and for val
for n, d in (("train(OOF ml)", tr), ("val", va)):
    # Share of pages answered by each stage
    print("\n==", n, "stage share:", d.stage.value_counts(normalize=True).round(3).to_dict())

    # Accuracy of the rules stage and of the ML stage on the pages each one decided
    for st in ("rules", "ml"):
        g = d[d.stage == st]
        if len(g):
            print(f"  {st}: n={len(g)} acc={(g.pred_pre_llm==g.label).mean():.4f}")

    # Confusion-matrix metrics for everything decided before Luna
    dec = d[d.stage != "llm"]
    print("  decided-before-Luna:", metrics(dec.label.values, dec.pred_pre_llm.values))

    # The pages left unsure (sent to Luna): how many, what share are impersonation, and which page types
    u = d[d.stage == "llm"]
    print(
        "  to Luna:",
        len(u),
        "pos share",
        round(u.label.mean(), 2) if len(u) else None,
        u.concept.value_counts().head(8).to_dict(),
    )

# Save the per-page stage tables (step 06 reads them to know which pages go to Luna)
tr.to_csv(f"{WORK}/dev_train_stage.csv", index=False)
va.to_csv(f"{WORK}/dev_val_stage.csv", index=False)

# FREEZE: store checksums of the code + model so step 08 can prove nothing changed before the test run
write_manifest()
print("frozen (checksums written; test is checked against them before it is run)")
