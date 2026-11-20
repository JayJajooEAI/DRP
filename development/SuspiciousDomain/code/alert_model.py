"""Step 7: small model check on the 'brand + extra word' alerts (LightGBM, no LLM, no cost).

Runs last, only on domains the pipeline alerts on (Suspicious or Parked) whose typosquat check was 'brand + extra
word': on train + val that check gives 129 of the 130 false alarms (fan / complaint sites, brand inside a word,
similar-name businesses); every other check has none, so they are never sent. Below the threshold the alert is
cleared; otherwise the label from Steps 1-6 stays (Step 6 still decides Suspicious or Parked).

The model is the Step 4 LightGBM (04_string_model.py: same 28 name + domain-fact features, same settings), trained on
the train-company alerts sent here, with the threshold picked on the val-company alerts. Test is never used.
"""
from importlib import import_module

M = import_module("string_model")
SENT = {"brand + extra word"}


def train(sent_rows):
    """sent_rows: {split: [(features, truly relevant 0/1)]} for the alerts this step checks -> fitted model."""
    X, y = zip(*sent_rows["train"])
    model = M.Model(list(X), list(y))
    Xv, yv = zip(*sent_rows["val"])
    f1, t = model.pick_threshold(list(Xv), list(yv))
    print(f"alert model (Step 7): train {len(y)} alerts ({len(y) - sum(y)} harmless), val {len(yv)} alerts "
          f"({len(yv) - sum(yv)} harmless); threshold {t} (val F1 {f1:.3f})")
    print("  top features:", ", ".join(f"{k} {v}" for k, v in model.importance()[:8]))
    return model


def check(model, feats):
    """-> (keep the alert?, note)."""
    p = model.score(feats)
    return p >= model.threshold, f"p={p:.2f} (threshold {model.threshold}) -> {'keep' if p >= model.threshold else 'none'}"
