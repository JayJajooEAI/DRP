"""Suspicious-domain detector: a step-by-step cascade. Each (domain, client) row gets one of three classes.

    python code/02_train_and_run.py          # train the two small models, run the detector over every row, write predictions, score train and val (add --test to score test too, --out DIR to write elsewhere)

The detector sees only what production would: the client profile and the domain's facts from data_api.py (no case,
label or expected path). Truth is read only by the scorer (03_evaluate.py).

Steps, in the order they run in detect():
    1. quick ownership check (ownership.py): owned list / registrant is the client / listed partner -> none
    2. typosquat rules (typosquat.py): a match -> relevant
    3. string match, clear rules (string_match.py), when no typosquat rule fires
    4. string model (string_model.py), unclear names only: LightGBM, trained on train, threshold on val
    5. legitimate third parties (legitimacy.py), on relevant domains before the label
    6. parked check (parked.py): parking nameservers or parking page -> domains_parked, else suspicious
    7. alert model (alert_model.py), only on 'brand + extra word' alerts: LightGBM, trained on train, threshold on
       val; below the threshold -> none
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from importlib import import_module  # noqa: E402   (03_evaluate starts with a digit)

A = import_module("data_api")
evaluate = import_module("03_evaluate")
typosquat = import_module("typosquat")
ownership = import_module("ownership")
parked = import_module("parked")
string_match = import_module("string_match")
string_model = import_module("string_model")
legitimacy = import_module("legitimacy")
alert_model = import_module("alert_model")
MODEL = None                                  # set in main(): trained on train, threshold on val
ALERT_MODEL = None                            # Step 7, set in main() the same way

OUT = HERE.parent / "outputs" / "02_train_and_run"


def inputs(domain, profile):
    """Everything the detector may look at for one (domain, client) pair."""
    return {"domain": domain, "profile": profile, "rdap": A.get_rdap(domain), "nameservers": A.get_nameservers(domain),
            "http": A.get_http(domain), "cert": A.get_cert(domain), "page": A.render(domain),
            "tranco_rank": A.tranco_rank(domain), "on_phishing_feed": A.is_known_phish(domain)}


def detect(x):
    """-> (class, trace). Class is suspicious_domain_registration / domains_parked / none."""
    own = ownership.check(x["domain"], x["profile"], x["rdap"])
    if own:
        return "none", [f"ownership {own[0]}: {own[1]}"]
    hit = typosquat.check(x["domain"], x["profile"])
    if hit:
        trace, flagged_by = [f"typosquat: {hit[0]} ({hit[1]})"], hit[0]
    else:
        flagged_by = "string match"
        verdict, reason, words = string_match.check(x["domain"], x["profile"])
        if reason.startswith("executive full name"):
            flagged_by = "executive full name"     # never cleared by the legitimacy clues (Step 5 safeguard)
        trace = ["no typosquat rule fires", f"words: {' '.join(words)}", f"string match {verdict}: {reason}"]
        if verdict == "unclear" and MODEL is not None:
            p = MODEL.score(string_model.features(x))
            verdict = "relevant" if p >= MODEL.threshold else "not relevant"
            trace.append(f"model: p={p:.2f} (threshold {MODEL.threshold}) -> {verdict}")
        if verdict != "relevant":
            return "none", trace                 # clear, no name, unclear without a model, or model says no
    legit = legitimacy.check(x, flagged_by)
    if legit:
        return "none", trace + [f"legitimate {legit[0]}: {legit[1]}"]
    park = parked.check(x["nameservers"], x["http"])
    cls, trace = (("domains_parked", trace + [f"parked: {park}"]) if park
                  else ("suspicious_domain_registration", trace + ["not parked"]))
    if flagged_by in alert_model.SENT and ALERT_MODEL is not None:
        keep, note = alert_model.check(ALERT_MODEL, string_model.features(x))
        trace.append(f"alert model: {note}")
        if not keep:
            return "none", trace
    return cls, trace


def train_model(rows, profiles):
    """Rows the rules call unclear: train on train companies, pick the threshold on val companies."""
    data = {"train": ([], []), "val": ([], [])}
    for r in rows:
        if r["split"] not in data or r["case"] in ("A1", "A2"):
            continue
        x = inputs(r["domain"], profiles[r["company_id"]])
        if ownership.check(x["domain"], x["profile"], x["rdap"]) or typosquat.check(x["domain"], x["profile"]):
            continue
        if string_match.check(x["domain"], x["profile"])[0] != "unclear":
            continue
        X, y = data[r["split"]]
        X.append(string_model.features(x))
        y.append(int(r["label"] == "relevant"))          # labels are read here, for train / val only
    model = string_model.Model(*data["train"])
    f1, t = model.pick_threshold(*data["val"])
    print(f"string model: train {len(data['train'][1])} rows ({sum(data['train'][1])} relevant), "
          f"val {len(data['val'][1])} rows ({sum(data['val'][1])} relevant); threshold {t} (val F1 on these {f1:.3f})")
    print("  top features:", ", ".join(f"{k} {v}" for k, v in model.importance()[:8]))
    return model


def train_alert_model(rows, profiles):
    """Alerts Steps 1-6 raise through 'brand + extra word': train on train companies, threshold on val companies."""
    data = {"train": [], "val": []}
    for r in rows:
        if r["split"] not in data or r["case"] in ("A1", "A2"):
            continue
        x = inputs(r["domain"], profiles[r["company_id"]])
        cls, trace = detect(x)                     # ALERT_MODEL is still None here: Steps 1-6 only
        if cls != "none" and any(s.startswith("typosquat: brand + extra word") for s in trace):
            data[r["split"]].append((string_model.features(x), int(r["label"] == "relevant")))
    return alert_model.train(data)


def main(out=None):
    global MODEL, ALERT_MODEL, OUT
    if out:
        OUT = Path(out)
    profiles = {p["id"]: p for p in A.profiles()}
    rows = json.loads((A.WORLD / "domains.json").read_text())["rows"]
    (OUT / "model").mkdir(parents=True, exist_ok=True)
    MODEL = train_model(rows, profiles)
    ALERT_MODEL = train_alert_model(rows, profiles)
    # the two trained models and their thresholds: copy this folder to inference/SuspiciousDomain/model/
    MODEL.save(OUT / "model" / "string_model.txt")
    ALERT_MODEL.save(OUT / "model" / "alert_model.txt")
    (OUT / "model" / "thresholds.json").write_text(json.dumps(
        {"features": string_model.FEATURES, "string_model_threshold": MODEL.threshold,
         "alert_model_threshold": ALERT_MODEL.threshold}, indent=1))
    with (OUT / "predictions.jsonl").open("w") as f:
        for r in rows:          # only the domain and its client are taken from the row
            cls, trace = detect(inputs(r["domain"], profiles[r["company_id"]]))
            f.write(json.dumps({"id": r["id"], "domain": r["domain"], "choice": cls, "trace": trace}) + "\n")
    evaluate.main(OUT)


if __name__ == "__main__":
    a = sys.argv
    main(a[a.index("--out") + 1] if "--out" in a else None)
