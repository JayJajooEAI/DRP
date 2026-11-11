"""Turn a signals table into the feature table the frozen LightGBM model expects (same logic as training, no fitting here)."""

import pandas as pd

from pipeline.rules import rule_scores, to01

# Columns that identify the page but are not features
META = ["id", "customer", "platform", "layout"]


def feats(d):
    """Signals table -> numeric feature table (before picking the model's exact column order)."""
    # Drop private "_" columns and identifiers (ground-truth columns are not present at inference time)
    X = d.drop(
        columns=[
            c
            for c in d.columns
            if c.startswith("_") or c in META + ["label", "concept", "tier", "expected_incident_type"]
        ]
    ).copy()

    # Add each rule's points and the combined 0..1 rule score as extra features
    R = rule_scores(d)
    X = pd.concat([X, R, to01(R.sum(axis=1)).rename("rule_score")], axis=1)

    # Platform as a number (unknown = -1)
    X["plat_code"] = d.platform.map(
        {"X": 0, "Facebook": 1, "Instagram": 2, "LinkedIn": 3, "Telegram": 4, "TikTok": 5}
    ).fillna(-1)
    return X.astype(float)
