"""Step 09: build a local error viewer for the check 2 and check 3 models (one chosen epoch each).

    python code/09_build_error_viewer.py --check2-epoch 9 --check3-epoch 9

Writes outputs/09_build_error_viewer/:
  index.html             open in a browser (works offline from the folder)
  data/summary.js        score tables (zoomed out / zoomed in, val and test) and the list of misclassified cases
  data/cases/<n>.js      one file per case, loaded on demand: full prompt, OCR lines, extracted HTML fields, raw HTML
  images/<n>.jpg         the case's screenshot
Raw page HTML is shown as text only, never rendered. Notes are kept in the browser (localStorage) and can be
exported / imported as JSON.

Each case also carries the end-to-end view: page type (phishing / benign client page), check 4 (client's own domain,
or a form posting off-domain), the pipeline's Web Phish verdict from both models' predictions + check 4, the label
verdict, and whether this check-level error changes the final verdict. Check 4 runs before checks 2 and 3 in the
recipe's verdict rule, so errors on the client's own pages never reach the verdict.
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
OUT = OUTPUT / "09_build_error_viewer"
SPLITS = ("val", "test")
LABELS = {2: ["False", "True"], 3: ["3.1", "3.2", "none"]}
NAMES2 = {"False": "not branded", "True": "branded"}


def case_kind(check: int, label: str, pred: str, synthetic: bool) -> tuple[str, str]:
    """(tab, type) for one misclassified row."""
    if check == 2:
        if label == "True":
            return "fn", "FN: branded page called not branded"
        return "fp", "FP: wrong-client pair accepted" if synthetic else "FP: real unbranded page called branded"
    if pred == "none":
        return "fn", f"FN: {label} page called none"
    if label == "none":
        return "fp", f"FP: none page called {pred}"
    return "swap", f"Swap: {label} called {pred}"


def confusion(df: pd.DataFrame, labels: list) -> dict:
    rows = {}
    for a in labels:
        r = df[df["label"] == a]
        rows[a] = {p: int((r["pred"] == p).sum()) for p in labels}
    return {"labels": labels, "rows": rows, "n": int(len(df)),
            "accuracy": float((df["label"] == df["pred"]).mean()) if len(df) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check2-epoch", type=int, default=9)
    ap.add_argument("--check3-epoch", type=int, default=9)
    args = ap.parse_args()
    epochs = {2: args.check2_epoch, 3: args.check3_epoch}

    items = pd.read_csv(ROOT / "data" / "benchmark" / "items.csv", keep_default_na=False).set_index("item_id")
    labels = pd.read_csv(ROOT / "data" / "benchmark" / "labels.csv").set_index("item_id")
    html = pd.read_parquet(OUTPUT / "01_parse_html" / "b_html_fields.parquet").set_index("item_id")
    prompts = {c: pd.read_parquet(OUTPUT / "06_build_prompts" / f"check{c}.parquet") for c in (2, 3)}
    stage1 = pd.read_csv(ROOT / "data" / "benchmark" / "stage1_features.csv").set_index("item_id")
    # both models' predictions per item (real rows only for check 2), for the end-to-end verdict
    real_pred = {}
    for check in (2, 3):
        ep = OUTPUT / "07_train_models" / f"check{check}" / f"epoch_{epochs[check]}"
        for split in SPLITS:
            pr = pd.read_parquet(ep / f"predictions_{split}.parquet")
            pr = pr[~pr["synthetic"]] if check == 2 else pr
            real_pred.setdefault(check, {}).update(dict(zip(pr["item_id"], pr["pred"])))

    def end_to_end(item_id: str) -> dict:
        lab = labels.loc[item_id]
        own = not bool(lab["check_4_not_client_domain"])
        ext = bool(stage1.loc[item_id, "external_form_target"]) if item_id in stage1.index else False
        c4 = (not own) or ext
        pipe = real_pred[2][item_id] == "True" and real_pred[3][item_id] in ("3.1", "3.2") and c4
        return {"own_domain": own, "form_posts_off_domain": ext, "check4_pass": c4,
                "pipeline_web_phish": bool(pipe), "label_web_phish": lab["routed_to"] == "Web Phish",
                "label_routed_to": lab["routed_to"], "pred_check2": real_pred[2][item_id],
                "pred_check3": real_pred[3][item_id]}

    def write_case(head: dict, prompt_text: str):
        """Save one case: screenshot, and a lazily loaded JS file with every piece of evidence."""
        item_id, cid = head["item_id"], head["id"]
        it, lab, h = items.loc[item_id], labels.loc[item_id], html.loc[item_id]
        img = Image.open(ROOT / it["screenshot_path"]).convert("RGB")
        img.thumbnail((1600, 4000))
        img.save(OUT / "images" / f"{cid}.jpg", quality=82)
        ocr_path = OUTPUT / "02_ocr_read_screenshots" / f"{item_id}.json"
        ocr = json.loads(ocr_path.read_text())["lines"] if ocr_path.exists() else []
        raw = (ROOT / it["html_path"]).read_bytes()
        try:
            raw_text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raw_text = raw.decode("cp1252", errors="replace")
        summary["cases"].append(head)
        detail = head | {
            "url": it["url"], "final_url": it["final_url"],
            "labeller": {"source": lab["label_source"], "rationale": lab["rationale"],
                         "check_2_client_branded": bool(lab["check_2_client_branded"]),
                         "check_2_evidence": lab["check_2_evidence"], "matched_customer": lab["matched_customer"],
                         "check_3_case": lab["check_3_case"], "check_3_source": lab["check_3_source"]},
            "prompt": prompt_text, "prompt_chars": len(prompt_text),
            "ocr": [{"text": l["text"], "score": l["score"], "y": l["box"][0][1]} for l in ocr],
            "html_fields": {k: (list(h[k]) if hasattr(h[k], "tolist") else h[k]) for k in
                            ("html_lang", "title", "headings", "brand_hints", "forms", "inputs", "script_inputs",
                             "script_urls", "links", "iframes", "visible_text")},
            "raw_html": raw_text, "raw_html_bytes": len(raw), "image": f"images/{cid}.jpg"}
        (OUT / "data" / "cases" / f"{cid}.js").write_text(
            "window.CASE_DATA=window.CASE_DATA||{};window.CASE_DATA[" + json.dumps(cid) + "]="
            + json.dumps(detail, ensure_ascii=False, default=str) + ";")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "data" / "cases").mkdir(parents=True)
    (OUT / "images").mkdir()

    summary = {"epochs": epochs, "scores": {}, "cases": []}
    n = 0
    for check in (2, 3):
        ep = OUTPUT / "07_train_models" / f"check{check}" / f"epoch_{epochs[check]}"
        summary["scores"][check] = {}
        for split in SPLITS:
            pred = pd.read_parquet(ep / f"predictions_{split}.parquet")
            pred["class"] = pred["item_id"].map(items["class"])
            sc = {"all": confusion(pred, LABELS[check])}
            if check == 2:
                sc["real"] = confusion(pred[~pred["synthetic"]], LABELS[check])
                sc["pairs"] = confusion(pred[pred["synthetic"]], LABELS[check])
            else:
                sc["phishing"] = confusion(pred[pred["class"] == "positive"], LABELS[check])
                sc["benign"] = confusion(pred[pred["class"] == "negative"], LABELS[check])
            summary["scores"][check][split] = sc

            p = prompts[check][prompts[check]["split"] == split]
            for _, r in pred[pred["label"] != pred["pred"]].iterrows():
                tab, kind = case_kind(check, r["label"], r["pred"], bool(r["synthetic"]))
                match = p[(p["item_id"] == r["item_id"]) & (p["client_id"] == r["client_id"])]
                if check == 2:
                    match = match[match["synthetic"] == r["synthetic"]]
                prow = match.iloc[0]
                it = items.loc[r["item_id"]]
                cid = f"c{n:03d}"
                n += 1
                probs = {l: float(r[f"p_{l}"]) for l in LABELS[check]}
                e2e = end_to_end(r["item_id"])
                if bool(r["synthetic"]):
                    affects, why = False, "wrong-client pair: a test device, never run in the pipeline"
                elif not e2e["check4_pass"]:
                    affects, why = False, "client's own domain: check 4 stops it before checks 2/3 decide"
                elif check == 3 and r["label"] != "none" and r["pred"] != "none":
                    affects, why = False, "3.1 / 3.2 swap: both are Web Phish (only the case type differs)"
                elif e2e["pipeline_web_phish"] == e2e["label_web_phish"]:
                    affects, why = False, "the other check already decides the verdict the same way"
                else:
                    affects = True
                    why = ("missed Web Phish" if e2e["label_web_phish"] else "false Web Phish alarm")
                head = {"id": cid, "check": check, "split": split, "tab": tab, "type": kind,
                        "item_id": r["item_id"], "page_class": "phishing" if it["class"] == "positive" else "benign",
                        "prompt_client": prow["client_id"], "item_customer": it["customer_id"],
                        "synthetic": bool(r["synthetic"]), "label": r["label"], "pred": r["pred"],
                        "probs": probs, "n_tokens": int(r["n_tokens"]),
                        "affects_verdict": affects, "affects_reason": why, **e2e}
                write_case(head, prow["text"])

    # final verdict: Web Phish = check 1 (assumed live) + check 2 model + check 3 model + check 4 domain rule
    summary["scores"]["final"] = {}
    zerofox = items["web_phish_zerofox"].astype(str) == "True"
    p3 = prompts[3].set_index("item_id")
    for split in SPLITS:
        ids = labels.index[labels["split"] == split]
        rows = []
        for i in ids:
            e = end_to_end(i)
            rows.append({"item_id": i, "pipe": e["pipeline_web_phish"], "label": e["label_web_phish"],
                         "zerofox": bool(zerofox[i]), **e})
        df = pd.DataFrame(rows)
        sc = {}
        for name, truth in (("vs_labels", "label"),):
            y, pr = df[truth], df["pipe"]
            tp, fp = int((y & pr).sum()), int((~y & pr).sum())
            fn, tn = int((y & ~pr).sum()), int((~y & ~pr).sum())
            prec = tp / (tp + fp) if tp + fp else None
            rec = tp / (tp + fn) if tp + fn else None
            f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
            sc[name] = {"labels": ["not Web Phish", "Web Phish"], "n": len(df), "accuracy": (tp + tn) / len(df),
                        "rows": {"not Web Phish": {"not Web Phish": tn, "Web Phish": fp},
                                 "Web Phish": {"not Web Phish": fn, "Web Phish": tp}},
                        "precision": prec, "recall": rec, "f1": f1}
        tp_rows = df[df["pipe"] & df["label"]]
        sc["case_accuracy_on_tp"] = float((tp_rows["pred_check3"].values
                                           == labels.loc[tp_rows["item_id"], "check_3_case"].values).mean())
        sc["benign_flagged"] = int(sum(df["pipe"] & (items.loc[df["item_id"], "class"].values == "negative")))
        sc["benign_total"] = int(sum(items.loc[df["item_id"], "class"].values == "negative"))
        summary["scores"]["final"][split] = sc
        for _, r in df[df["pipe"] != df["label"]].iterrows():
            lab = labels.loc[r["item_id"]]
            if r["label"]:   # missed
                causes = [c for c, bad in (("check 2 said not branded", r["pred_check2"] != "True"),
                                           ("check 3 said none", r["pred_check3"] == "none"),
                                           ("check 4: client's own domain", not r["check4_pass"])) if bad]
                tab, kind = "fn", "Missed Web Phish (" + "; ".join(causes) + ")"
            else:
                why = [c for c, bad in (("label: not branded", not lab["check_2_client_branded"]),
                                        ("label: no login/payment (none)", lab["check_3_case"] == "none"),
                                        ("label: parked", bool(lab["parked"])),
                                        ("label: own domain, not compromised", not lab["check_4_not_client_domain"]))
                       if bad]
                tab, kind = "fp", "False Web Phish alarm (" + "; ".join(why or ["label routed elsewhere"]) + ")"
            it = items.loc[r["item_id"]]
            prow = p3.loc[r["item_id"]]
            c2p = pd.read_parquet(OUTPUT / "07_train_models" / "check2" / f"epoch_{epochs[2]}" / f"predictions_{split}.parquet")
            c3p = pd.read_parquet(OUTPUT / "07_train_models" / "check3" / f"epoch_{epochs[3]}" / f"predictions_{split}.parquet")
            c2r = c2p[(c2p["item_id"] == r["item_id"]) & (~c2p["synthetic"])].iloc[0]
            c3r = c3p[c3p["item_id"] == r["item_id"]].iloc[0]
            cid = f"c{n:03d}"
            n += 1
            head = {"id": cid, "check": "final", "split": split, "tab": tab, "type": kind,
                    "item_id": r["item_id"], "page_class": "phishing" if it["class"] == "positive" else "benign",
                    "prompt_client": prow["client_id"], "item_customer": it["customer_id"], "synthetic": False,
                    "label": "Web Phish" if r["label"] else "not Web Phish",
                    "pred": "Web Phish" if r["pipe"] else "not Web Phish",
                    "probs": {"branded (check 2)": float(c2r["p_True"]), "3.1 (check 3)": float(c3r["p_3.1"]),
                              "3.2 (check 3)": float(c3r["p_3.2"]), "none (check 3)": float(c3r["p_none"])},
                    "n_tokens": int(c3r["n_tokens"]), "affects_verdict": True,
                    "affects_reason": "this is a final-verdict error",
                    **{k: r[k] for k in ("own_domain", "form_posts_off_domain", "check4_pass", "pipeline_web_phish",
                                         "label_web_phish", "label_routed_to", "pred_check2", "pred_check3")}}
            head = {k: (bool(v) if isinstance(v, (bool,)) or type(v).__name__ == "bool_" else v) for k, v in head.items()}
            write_case(head, prow["text"])

    (OUT / "data" / "summary.js").write_text("window.SUMMARY=" + json.dumps(summary, ensure_ascii=False) + ";")
    shutil.copy(Path(__file__).with_name("error_viewer.html"), OUT / "index.html")
    print(f"{n} cases -> {OUT / 'index.html'}")


if __name__ == "__main__":
    main()
