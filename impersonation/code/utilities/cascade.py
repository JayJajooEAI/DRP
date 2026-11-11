"""Utility: the cascade. Step 2 rules -> Step 3 LightGBM -> Step 4 Luna (unsure only).  fit() uses TRAIN only; apply() is frozen."""

import json, joblib, numpy as np, pandas as pd, warnings

warnings.filterwarnings("ignore")
from utilities.config import *
from utilities.rules import rule_score, decide
from utilities.ml_model import feats, fit as ml_fit, oof

# RULE_TARGET = rules must be 99.5% precise on the pages they decide; ML_CONF = ML decides only if its probability is <= 2% or >= 98%; MIN_N = a threshold must cover at least 20 train pages
RULE_TARGET = 0.995
ML_CONF = 0.02
MIN_N = 20  # ML decides only when p<=ML_CONF or p>=1-ML_CONF (chosen on train-OOF + val)


def load(sp):
    """Signals csv + the truth columns, joined by page id (the truth is only for scoring / fitting; the ML features drop it)."""
    # S = signals from step 04; T = ground truth columns from the split csv
    S = pd.read_csv(f"{WORK}/signals_{sp}.csv")
    T = pd.DataFrame(load_split(sp))[["id", "label", "concept", "tier", "expected_incident_type"]]
    d = S.merge(T, on="id")
    d["label"] = d["label"].astype(int)
    return d


def band(score, y, target, grid):
    """smallest-coverage-loss thresholds: hi = lowest score with precision>=target among {score>=hi}; lo = highest score with npv>=target among {score<=lo}."""
    # hi: scan thresholds upward, take the first where pages scoring >= hi are at least `target` impersonation (and there are enough of them)
    hi = next((h for h in grid if (score >= h).sum() >= MIN_N and y[score >= h].mean() >= target), None)

    # lo: scan thresholds downward, take the first where pages scoring <= lo are at least `target` NOT impersonation
    lo = next(
        (l for l in grid[::-1] if (score <= l).sum() >= MIN_N and (1 - y[score <= l]).mean() >= target), None
    )

    # If no threshold qualifies, use infinity so that side never decides
    return (hi if hi is not None else np.inf), (lo if lo is not None else -np.inf)


def fit_all():
    """Fit everything on TRAIN: rule thresholds, out-of-fold ML probabilities, the final model. Saves frozen.joblib."""
    # tr = train table; y = labels; s = rule score per page
    tr = load("train")
    y = tr.label.values
    s = rule_score(tr).values

    # Rule thresholds searched on a 0.001 grid
    rhi, rlo = band(s, y, RULE_TARGET, np.round(np.arange(0, 1.0001, 0.001), 3))

    # X = features; po = honest (out-of-fold) ML probabilities; model = LightGBM fitted on all of train
    X = feats(tr)
    po = oof(X, y, tr.customer.values)
    model = ml_fit(X, y)

    # art = the frozen settings (thresholds + feature column order)
    art = dict(rule_hi=float(rhi), rule_lo=float(rlo), ml_hi=1 - ML_CONF, ml_lo=ML_CONF, cols=list(X.columns))

    # Save the model and settings
    joblib.dump(dict(model=model, art=art), f"{WORK}/frozen.joblib")
    json.dump(art, open(f"{REPORTS}/frozen_thresholds.json", "w"), indent=1)
    tr["rule_score"] = s
    tr["p_ml"] = po
    return tr, art


def apply(d, F=None):
    """Score a signals table with the FROZEN rules + model (nothing is fitted here)."""
    # Load the frozen model if not given; X = features in the exact column order used at training
    F = F or joblib.load(f"{WORK}/frozen.joblib")
    a = F["art"]
    X = feats(d)[a["cols"]]

    # Add the rule score and the ML probability to a copy of the table
    d = d.copy()
    d["rule_score"] = rule_score(d).values
    d["p_ml"] = F["model"].predict_proba(X)[:, 1]
    return d


def cascade(d, art):
    """Decide which stage answers each page and what it says (before Luna)."""
    # rd = rule decision (1 / 0 / -1 unsure); md = ML decision (1 / 0 / -1 unsure)
    rd = decide(d.rule_score.values, art["rule_hi"], art["rule_lo"])
    md = np.where(d.p_ml >= art["ml_hi"], 1, np.where(d.p_ml <= art["ml_lo"], 0, -1))

    # stage: rules if they decided; otherwise ml if it decided; otherwise llm (Luna). pred_pre_llm = the answer given by that stage (-1 if Luna must answer).
    d["stage"] = np.where(rd >= 0, "rules", np.where(md >= 0, "ml", "llm"))
    d["pred_pre_llm"] = np.where(rd >= 0, rd, md)
    return d


def metrics(y, p):
    """Confusion matrix and scores. y = true labels, p = predictions (1 = impersonation)."""
    # tp / fp / fn / tn = true positives, false positives, false negatives, true negatives
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())

    # precision, recall, F1 and accuracy (max(1, ...) avoids dividing by zero)
    pr = tp / max(1, tp + fp)
    rc = tp / max(1, tp + fn)
    return dict(
        n=len(y),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=round(pr, 4),
        recall=round(rc, 4),
        f1=round(2 * pr * rc / max(1e-9, pr + rc), 4),
        accuracy=round((tp + tn) / len(y), 4),
    )


def combine(stage_csv, luna_json):
    """Merge the Luna answers into a stage csv -> final prediction per page (Luna failed -> the model's guess)."""
    # d = stage table; L = Luna answers by page id; pred = predictions before Luna
    d = pd.read_csv(stage_csv)
    L = json.load(open(luna_json))
    pred = d.pred_pre_llm.copy()
    for i, o in L.items():
        # Luna answered: 1 if it said "impersonation", else 0
        if o["choice"] is not None:
            pred[d.index[d.id == i][0]] = int(o["choice"] == "impersonation")

        # Luna failed (error / over budget): fall back to the ML model's best guess (p >= 0.5)
        else:
            pred[d.index[d.id == i][0]] = int(d.loc[d.id == i, "p_ml"].iloc[0] >= 0.5)
    d["pred"] = pred.astype(int)
    return d


# ---- freeze: checksums of the code + fitted model, written when the pipeline is frozen (05) and checked before the test run (08)
import hashlib, glob, os
from pathlib import Path as _P

# _CODE = the code folder
_CODE = _P(__file__).resolve().parents[1]


def _frozen_files():
    """Every file whose content must not change between freezing and the test run: utilities, the key scripts, and the fitted model."""
    return sorted(
        str(p)
        for p in list(_CODE.glob("utilities/*.py"))
        + [
            _CODE / n
            for n in (
                "01_ocr_read_screenshots.py",
                "02_avatar_matching.py",
                "04_build_signals.py",
                "05_fit_cascade_train_val.py",
                "06_luna_unsure_pages.py",
            )
        ]
    ) + [f"{WORK}/frozen.joblib"]


def _md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()  # fingerprint of a file's bytes


def write_manifest():
    """Record the fingerprint of every frozen file (called at the end of step 05)."""
    m = {os.path.relpath(p, str(_CODE.parent)): _md5(p) for p in _frozen_files()}
    json.dump(m, open(f"{REPORTS}/frozen_manifest.json", "w"), indent=1)
    return m


def check_manifest():
    """Stop with an error if any frozen file changed since step 05 (called at the start of step 08)."""
    # m = recorded fingerprints; bad = files whose current fingerprint differs
    m = json.load(open(f"{REPORTS}/frozen_manifest.json"))
    bad = [k for k, v in m.items() if _md5(str(_CODE.parent / k)) != v]
    if bad:
        raise SystemExit(
            "Frozen pipeline changed since it was frozen: "
            + ", ".join(bad)
            + ". Re-run 05 (train/val only) and freeze again before touching test."
        )
    print("frozen manifest OK:", len(m), "files")
