"""Step 07: fine-tune mmBERT-base (Johns Hopkins, ModernBERT architecture, 8,192-token context) as the
check 2 (branded: true / false) and check 3 (data capture: 3.1 / 3.2 / none) classifiers. One run trains both models,
9 epochs each, on one learning-rate schedule (6% warm-up, then linear decay).

    python code/07_train_models.py                          # both models, 9 epochs each
    python code/07_train_models.py --check 3                # one model only
    python code/07_train_models.py --check 3 --resume-from outputs/07_train_models/check3/epoch_6   # continue to epoch 9
    python code/07_train_models.py --check 3 --limit 200 --epochs 1 --tag smoke    # quick timing check

Reads  outputs/06_build_prompts/check{2,3}.parquet; trains on the train split (for check 2 this includes the
wrong-client pairs). After every epoch the model is saved and scored on train, val and test (all rows); for check 2
each split is also broken down into real rows and wrong-client pairs. No epoch is chosen here: every checkpoint is
kept, with its metrics and predictions, so any epoch can be loaded for inference (08_predict.py).

Writes outputs/07_train_models/check{N}[_tag]/:
  epoch_{k}/model/                 weights, config and tokenizer (load with from_pretrained)
  epoch_{k}/training_state.pt      optimizer, LR schedule and scaler state, to resume from this epoch (about 2.3 GB)
  epoch_{k}/inference_config.json  label order, max length, prompt fields and cut order used to build the inputs
  epoch_{k}/metrics.json           every metric for train / val / test (and real / pairs for check 2)
  epoch_{k}/predictions_{split}.parquet  per-row probabilities and predicted label
  train_log.jsonl, run_info.json
"""
import argparse
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_common import (CUT_ORDER, FIELDS, LABELS, MODEL_ID, collate, fit_to_length, format_metrics, metrics,
                    predict_probs, token_batches)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "06_build_prompts"
OUT = ROOT / "outputs" / "07_train_models"
SPLITS = ("train", "val", "test")
CPU_THREADS = 2   # shared VM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, choices=[2, 3], help="train only this model (default: both, check 3 then check 2)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=9)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--token-budget", type=int, default=16384, help="max rows x longest row per batch")
    ap.add_argument("--max-rows", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--real-false-weight", type=float, default=3.0,
                    help="check 2: extra loss weight for the few real not-branded rows next to the many pairs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only N rows per split (timing / smoke test)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume-from", help="an epoch_{k} folder of one model: continue training from it up to --epochs in total "
                    "(use with --check, since a checkpoint belongs to one model)")
    args = ap.parse_args()
    if args.resume_from and not args.check:
        ap.error("--resume-from needs --check")
    for check in ([args.check] if args.check else [3, 2]):
        args.check = check
        train_one(args)


def train_one(args):
    """Fine-tune and score one model (args.check = 2 or 3)."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    torch.set_num_threads(CPU_THREADS)
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = "cuda"
    out = OUT / f"check{args.check}{'_' + args.tag if args.tag else ''}"
    out.mkdir(parents=True, exist_ok=True)
    labels = LABELS[args.check]

    df = pd.read_parquet(DATA / f"check{args.check}.parquet")
    if "synthetic" not in df:
        df["synthetic"] = False
    df["y"] = df["label"].map({l: k for k, l in enumerate(labels)})
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
    if args.limit:
        parts = {s: p.sample(min(args.limit, len(p)), random_state=args.seed).reset_index(drop=True)
                 for s, p in parts.items()}

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    t0 = time.time()
    enc = {s: tok([fit_to_length(r, tok, args.max_len) for _, r in p.iterrows()], add_special_tokens=True,
                  truncation=True, max_length=args.max_len)["input_ids"] for s, p in parts.items()}
    lengths = {s: np.array([len(x) for x in v]) for s, v in enc.items()}
    print("tokenised " + ", ".join(f"{s} {len(parts[s])} rows / {lengths[s].sum() / 1e6:.2f}M tokens" for s in SPLITS)
          + f" in {time.time() - t0:.0f}s", flush=True)

    train = parts["train"]
    counts = np.bincount(train["y"], minlength=len(labels))
    class_w = torch.tensor(np.sqrt(counts.sum() / np.maximum(counts, 1) / len(counts)), dtype=torch.float32,
                           device=device)
    row_w = np.ones(len(train), np.float32)
    if args.check == 2:
        row_w[(train["y"] == 0) & (~train["synthetic"])] = args.real_false_weight

    start = 0
    resume = Path(args.resume_from) if args.resume_from else None
    if resume:
        # continue from a saved epoch: weights from its model/ folder, optimizer + LR schedule + scaler from training_state.pt
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        start = state["epoch"]
        model = AutoModelForSequenceClassification.from_pretrained(resume / "model", attn_implementation="sdpa").to(device)
        print(f"resuming from {resume} (epoch {start})", flush=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(labels), attn_implementation="sdpa",
            id2label={k: str(l) for k, l in enumerate(labels)}, label2id={str(l): k for k, l in enumerate(labels)}
        ).to(device)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(token_batches(lengths["train"], args.token_budget, args.max_rows, False, 0))
                                / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup * total_steps)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler()
    if resume:
        opt.load_state_dict(state["optimizer"]); sched.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
    print(f"LR schedule over {total_steps} steps ({warmup_steps} warm-up)", flush=True)
    pad = tok.pad_token_id
    inference_config = {"model_id": MODEL_ID, "check": args.check, "labels": [str(l) for l in labels],
                        "max_len": args.max_len, "fields": FIELDS, "cut_order": CUT_ORDER,
                        "prompt_data": str((DATA / f"check{args.check}.parquet").relative_to(ROOT)),
                        "eval_token_budget": args.token_budget * 2, "eval_max_rows": args.max_rows * 2,
                        "lr_schedule_total_steps": total_steps}

    log = open(out / "train_log.jsonl", "a" if resume else "w")
    model.train()
    for epoch in range(start + 1, args.epochs + 1):
        batches = token_batches(lengths["train"], args.token_budget, args.max_rows, True, args.seed + epoch)
        te, seen, loss_sum = time.time(), 0, 0.0
        for b, rows in enumerate(batches):
            x, m = collate(enc["train"], rows, pad, device)
            y = torch.tensor(train["y"].values[rows], device=device)
            w = torch.tensor(row_w[rows], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids=x, attention_mask=m).logits
            loss = (torch.nn.functional.cross_entropy(logits.float(), y, weight=class_w, reduction="none") * w).mean()
            scaler.scale(loss / args.grad_accum).backward()
            if (b + 1) % args.grad_accum == 0 or b == len(batches) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            seen += len(rows); loss_sum += loss.item()
            if b % 50 == 0:
                rate = lengths["train"][np.concatenate(batches[:b + 1])].sum() / (time.time() - te)
                print(f"  epoch {epoch} batch {b + 1}/{len(batches)} rows {seen} loss {loss_sum / (b + 1):.4f} "
                      f"{rate:.0f} tok/s", flush=True)
        train_minutes = (time.time() - te) / 60

        # save this epoch's model, then score every split with it
        ep = out / f"epoch_{epoch}"
        model.save_pretrained(ep / "model"); tok.save_pretrained(ep / "model")
        # everything needed to continue training from this epoch (AdamW moments, LR scheduler, fp16 grad scaler)
        torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "total_steps": total_steps}, ep / "training_state.pt")
        (ep / "inference_config.json").write_text(json.dumps(inference_config | {"epoch": epoch}, indent=1))
        tv = time.time()
        rec = {"epoch": epoch, "train_loss": loss_sum / len(batches), "train_minutes": train_minutes,
               "lr_end": sched.get_last_lr()[0], "splits": {}}
        print(f"=== epoch {epoch}: train loss {rec['train_loss']:.4f}, {train_minutes:.1f} min", flush=True)
        for s in SPLITS:
            prob = predict_probs(model, enc[s], len(labels), pad, device, args.token_budget * 2, args.max_rows * 2)
            p = parts[s]
            y = p["y"].values
            rec["splits"][s] = {"all": metrics(args.check, y, prob)}
            print(format_metrics(f"{s} all", rec["splits"][s]["all"]), flush=True)
            if args.check == 2:
                for sub, mask in (("real", ~p["synthetic"].values), ("pairs", p["synthetic"].values)):
                    rec["splits"][s][sub] = metrics(args.check, y[mask], prob[mask])
                    print(format_metrics(f"{s} {sub}", rec["splits"][s][sub]), flush=True)
            pd.DataFrame({"item_id": p["item_id"], "client_id": p["client_id"], "synthetic": p["synthetic"],
                          "label": p["label"].astype(str), "pred": [str(labels[k]) for k in prob.argmax(1)],
                          "n_tokens": lengths[s], **{f"p_{l}": prob[:, k] for k, l in enumerate(labels)}}
                         ).to_parquet(ep / f"predictions_{s}.parquet", index=False)
        rec["eval_minutes"] = (time.time() - tv) / 60
        (ep / "metrics.json").write_text(json.dumps(rec, indent=1))
        log.write(json.dumps(rec) + "\n"); log.flush()

    info = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": MODEL_ID,
            "check": args.check, "args": vars(args), "rows": {s: len(p) for s, p in parts.items()},
            "tokens": {s: int(v.sum()) for s, v in lengths.items()}, "class_weights": class_w.tolist(),
            "minutes_total": (time.time() - t0) / 60,
            "checkpoints": [f"epoch_{k}" for k in range(start + 1, args.epochs + 1)]}
    (out / f"run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
