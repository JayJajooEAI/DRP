"""Step 08: run a saved checkpoint on prompts (inference).

    python code/08_predict.py \
        --checkpoint outputs/07_train_models/check3/epoch_9 --split test
    python code/08_predict.py \
        --checkpoint outputs/07_train_models/check2/epoch_9 --prompts some_prompts.parquet --out preds.parquet

Loads the model and tokenizer from <checkpoint>/model and rebuilds each prompt exactly as in training, using
<checkpoint>/inference_config.json (field order, cut order, max length). The prompts file needs the f_* field
columns written by step 06 (check2.parquet / check3.parquet have them).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model_common import fit_to_length, predict_probs

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="an epoch_{k} folder")
    ap.add_argument("--prompts", help="parquet with f_* columns; default: the checkpoint's own prompt data")
    ap.add_argument("--split", help="only rows of this split (train / val / test)")
    ap.add_argument("--out", help="output parquet; default <checkpoint>/inference_<split or all>.parquet")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = Path(args.checkpoint)
    cfg = json.loads((ckpt / "inference_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(ckpt / "model")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt / "model", attn_implementation="sdpa").to(device)

    df = pd.read_parquet(args.prompts or ROOT / cfg["prompt_data"])
    if args.split:
        df = df[df["split"] == args.split].reset_index(drop=True)
    texts = [fit_to_length(r, tok, cfg["max_len"], cfg["fields"], cfg["cut_order"]) for _, r in df.iterrows()]
    ids = tok(texts, add_special_tokens=True, truncation=True, max_length=cfg["max_len"])["input_ids"]
    prob = predict_probs(model, ids, len(cfg["labels"]), tok.pad_token_id, device, cfg["eval_token_budget"],
                         cfg["eval_max_rows"])

    keep = [c for c in ("item_id", "split", "client_id", "synthetic", "label") if c in df]
    res = df[keep].copy()
    res["pred"] = [cfg["labels"][k] for k in prob.argmax(1)]
    for k, l in enumerate(cfg["labels"]):
        res[f"p_{l}"] = prob[:, k]
    res["n_tokens"] = [len(x) for x in ids]
    out = Path(args.out) if args.out else ckpt / f"inference_{args.split or 'all'}.parquet"
    res.to_parquet(out, index=False)
    print(f"{len(res)} rows -> {out}")
    if "label" in res:
        print("accuracy:", round(float((res["label"].astype(str) == res["pred"]).mean()), 4))


if __name__ == "__main__":
    main()
