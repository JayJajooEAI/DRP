"""Final answer on train and val: stage answers + Luna answers -> confusion matrix and errors.  Usage: python 07_final_metrics_train_val.py"""

import pandas as pd
from utilities.config import *

# metrics = confusion matrix + precision/recall/accuracy; combine = merge Luna's answers into the stage table
from utilities.cascade import metrics, combine

for sp in ("train", "val"):
    # d = one row per page with the FINAL prediction (rules / ML answer, or Luna's answer for unsure pages); y = true labels
    d = combine(f"{WORK}/dev_{sp}_stage.csv", f"{WORK}/luna_{sp}.json")
    y = d.label.values

    # Final metrics for the whole cascade and the share of pages each stage answered
    print(
        sp,
        "FINAL",
        metrics(y, d.pred.values),
        "| stage share",
        d.stage.value_counts(normalize=True).round(3).to_dict(),
    )

    # List the wrong pages, grouped by page type (concept)
    e = d[d.pred != d.label]
    print("  errors:", len(e), e.concept.value_counts().to_dict())
