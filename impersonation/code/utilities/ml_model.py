"""Stage 2: LightGBM on signals (+ rule-score components). Trained on train only; grouped-by-customer OOF for threshold tuning."""

import pandas as pd, numpy as np, lightgbm as lgb

# GroupKFold = cross-validation where all pages of one customer stay in the same fold (so we test on unseen customers)
from sklearn.model_selection import GroupKFold
from utilities.rules import rule_scores, to01

# Columns that identify the page but are not features
META = ["id", "customer", "platform", "layout"]


def feats(d):
    """Signals table -> numeric feature table for LightGBM."""
    # Drop private "_" columns, identifiers and every ground-truth column (so the model can never see the answer)
    X = d.drop(
        columns=[
            c
            for c in d.columns
            if c.startswith("_") or c in META + ["label", "concept", "tier", "expected_incident_type"]
        ]
    ).copy()

    # Add each individual rule's points plus the combined 0..1 rule score as extra features
    R = rule_scores(d)
    X = pd.concat([X, R, to01(R.sum(axis=1)).rename("rule_score")], axis=1)

    # Platform as a number (unknown = -1)
    X["plat_code"] = d.platform.map(
        {"X": 0, "Facebook": 1, "Instagram": 2, "LinkedIn": 3, "Telegram": 4, "TikTok": 5}
    ).fillna(-1)
    return X.astype(float)


# LightGBM settings: 300 small trees (15 leaves), slow learning rate, row/column subsampling and L2 regularisation to avoid overfitting; fixed seed = repeatable
PARAMS = dict(
    objective="binary",
    n_estimators=300,
    learning_rate=0.04,
    num_leaves=15,
    min_child_samples=25,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    verbose=-1,
    random_state=7,
)


def fit(X, y):
    return lgb.LGBMClassifier(**PARAMS).fit(X, y)  # train one model on features X and labels y


def oof(X, y, groups, k=5):
    """Out-of-fold probabilities: every page is scored by a model that never saw its customer (honest probabilities for choosing thresholds)."""
    # p = probability per page, filled fold by fold
    p = np.zeros(len(y))

    # Train on k-1 groups of customers, predict the held-out group
    for a, b in GroupKFold(k).split(X, y, groups):
        p[b] = fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
    return p
