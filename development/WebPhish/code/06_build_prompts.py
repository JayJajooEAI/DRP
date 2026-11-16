"""Step 06: build the model inputs for the check 2 (branding) and check 3 (data capture) models.

Every labelled item becomes one text with separately marked fields (recipe Step 10), built from the earlier tasks:
  [CLIENT] [TITLE] [HEADINGS] [BRAND_HINTS] [FORMS] [SCRIPT_URLS] [LINKS] [VISIBLE_TEXT] [OCR_TEXT] [SIGNALS]
Nothing is truncated here: every field is kept whole. Fitting the text to a model's context length is decided later,
with the model, so each field is also stored in its own column.

Check 2 also gets wrong-client pairs: a branded page paired with a client from another company, labelled false,
with that client's name and that client's logo similarity. Every split gets pairs, flagged `synthetic` = True: train
learns from them; val and test report them apart (false-positive rate on mismatched clients), while tuning and the
official scores use the real rows only. Pages are never paired within a company (Office365 / Outlook / OneDrive).

    python code/06_build_prompts.py

Reads  data/benchmark/{items,labels,p_suspicious_url}.csv, data/customer_assets/*/profile.json,
       outputs/01_parse_html/b_html_fields.parquet, outputs/02_ocr_read_screenshots/<item_id>.json,
       outputs/03_logo_similarity/{logo_scores,logo_brand_stats}.parquet
Writes outputs/06_build_prompts/check2.parquet, check3.parquet, examples.txt, run_info.json
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "data" / "benchmark"
ASSETS = ROOT / "data" / "customer_assets"
OUTPUT = ROOT / "outputs"
OUT = OUTPUT / "06_build_prompts"

SEED = 42
PAIR_SPLITS = ("train", "val", "test")   # val/test pairs are flagged synthetic and reported apart
SEP = " | "
FIELDS = ["client", "title", "headings", "brand_hints", "forms", "script_urls", "links", "visible_text", "ocr_text",
          "signals"]


def num(v) -> str:
    return "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def ocr_text(item_id: str) -> str:
    path = OUTPUT / "02_ocr_read_screenshots" / f"{item_id}.json"
    return SEP.join(l["text"] for l in json.loads(path.read_text())["lines"]) if path.exists() else ""


def page_fields(item_id: str, h) -> dict:
    """The client-independent fields of one page, from the HTML parse and the OCR."""
    forms = list(h.forms) + list(h.inputs) + list(h.script_inputs)
    return {"title": h.title, "headings": SEP.join(h.headings), "brand_hints": SEP.join(h.brand_hints),
            "forms": SEP.join(forms), "script_urls": SEP.join(h.script_urls), "links": SEP.join(h.links),
            "visible_text": h.visible_text, "ocr_text": ocr_text(item_id)}


def render(fields: dict) -> str:
    return "\n".join(f"[{name.upper()}] {fields[name]}" for name in FIELDS)


def signals(logo_sim, p_url, forwarded) -> str:
    return f"logo_similarity={num(logo_sim)} p_suspicious_url={'forwarded' if forwarded else num(p_url)}"


def main():
    profiles = {p.parent.name: json.loads(p.read_text()) for p in sorted(ASSETS.glob("*/profile.json"))}
    name = {c: p["display_name"] for c, p in profiles.items()}
    group = {c: p.get("corporate_group") or c for c, p in profiles.items()}

    items = pd.read_csv(BENCH / "items.csv", keep_default_na=False).set_index("item_id")
    labels = pd.read_csv(BENCH / "labels.csv").set_index("item_id")
    purl = pd.read_csv(BENCH / "p_suspicious_url.csv").set_index("item_id")
    html = pd.read_parquet(OUTPUT / "01_parse_html" / "b_html_fields.parquet").set_index("item_id")
    logo = pd.read_parquet(OUTPUT / "03_logo_similarity" / "logo_scores.parquet").set_index("item_id")
    brand = pd.read_parquet(OUTPUT / "03_logo_similarity" / "logo_brand_stats.parquet")
    brand_sim = brand.pivot(index="item_id", columns="brand", values="sim_max")

    rows2, rows3 = [], []
    for item_id in labels.index:          # labelled items only; the 685 unlabelled benign pages are left out
        it, lab, h = items.loc[item_id], labels.loc[item_id], html.loc[item_id]
        page = page_fields(item_id, h)
        cust = it["customer_id"]
        p_url, fwd = purl.loc[item_id, "p_suspicious_url"], bool(purl.loc[item_id, "url_forwarded"])
        base = {"item_id": item_id, "split": it["split"], "class": it["class"]}

        f = page | {"client": name[cust], "signals": signals(logo.loc[item_id, "logo_similarity"], p_url, fwd)}
        common = base | {"client_id": cust, "logo_similarity": logo.loc[item_id, "logo_similarity"],
                         "p_suspicious_url": p_url, **{f"f_{k}": v for k, v in f.items()}, "text": render(f)}
        rows2.append(common | {"synthetic": False, "pair_kind": "real", "label": bool(lab["check_2_client_branded"])})
        rows3.append(common | {"label": lab["check_3_case"]})

    check2 = pd.DataFrame(rows2)

    # wrong-client pairs: one per real branded page whose page shows its own client, in every split
    rng = random.Random(SEED)
    pairs = []
    src = check2[check2["label"] & check2["split"].isin(PAIR_SPLITS)]
    src = src[[group.get(labels.loc[i, "matched_customer"]) == group[c] for i, c in zip(src["item_id"], src["client_id"])]]
    for k, (_, r) in enumerate(src.iterrows()):
        others = [c for c in profiles if group[c] != group[r["client_id"]]]
        sims = brand_sim.loc[r["item_id"], others]
        if k % 2 == 0 and sims.notna().any():   # confusable: the other company whose logos look most like the page's
            wrong, kind = sims.idxmax(), "confusable"
        else:
            wrong, kind = rng.choice(others), "random"
        wsim = brand_sim.loc[r["item_id"], wrong]
        f = {k2[2:]: r[k2] for k2 in r.index if k2.startswith("f_")}
        f |= {"client": name[wrong], "signals": signals(wsim, r["p_suspicious_url"], r["f_signals"].endswith("forwarded"))}
        pairs.append(r.to_dict() | {"client_id": wrong, "logo_similarity": wsim, **{f"f_{k2}": v for k2, v in f.items()},
                                    "text": render(f), "synthetic": True, "pair_kind": kind, "label": False})
    check2 = pd.concat([check2, pd.DataFrame(pairs)], ignore_index=True)
    check3 = pd.DataFrame(rows3)
    for df in (check2, check3):
        df["n_chars"] = df["text"].str.len()

    OUT.mkdir(parents=True, exist_ok=True)
    check2.to_parquet(OUT / "check2.parquet", index=False)
    check3.to_parquet(OUT / "check3.parquet", index=False)

    ex = []
    for title, df, cond in [("CHECK 2, real, branded", check2, (check2["pair_kind"] == "real") & check2["label"]),
                            ("CHECK 2, real, not branded", check2, (check2["pair_kind"] == "real") & ~check2["label"]),
                            ("CHECK 2, wrong-client pair (confusable)", check2, check2["pair_kind"] == "confusable"),
                            ("CHECK 3, 3.2", check3, check3["label"] == "3.2")]:
        r = df[cond & (df["n_chars"] < 3000)].sample(1, random_state=SEED).iloc[0]
        ex += ["=" * 100, f"{title}: {r['item_id']}  split={r['split']}  label={r['label']}", "=" * 100, r["text"], ""]
    (OUT / "examples.txt").write_text("\n".join(ex))

    def counts(df, extra=None):
        keys = ["split", "label"] + ([extra] if extra else [])
        return {" / ".join(map(str, k)): int(v) for k, v in df.groupby(keys).size().items()}
    info = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seed": SEED,
            "check2_rows": len(check2), "check3_rows": len(check3),
            "check2_counts": counts(check2, "pair_kind"), "check3_counts": counts(check3),
            "n_chars_median": {"check2": int(check2["n_chars"].median()), "check3": int(check3["n_chars"].median())},
            "n_chars_p95": {"check2": int(check2["n_chars"].quantile(.95)), "check3": int(check3["n_chars"].quantile(.95))},
            "n_chars_max": int(check3["n_chars"].max())}
    (OUT / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
