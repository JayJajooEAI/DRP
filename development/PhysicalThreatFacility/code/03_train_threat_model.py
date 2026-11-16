#!/usr/bin/env python3
"""
Training the Rung 2 threat classifier: supervised fine-tuning of distilbert-base-multilingual-cased (threat 1 / not 0).
Settings of the shipped model: 3 epochs, AdamW lr 2e-5, batch 32, max 128 tokens, fp16, seed 42.
Train = data/splits/train.csv (balanced 6,437 + 6,437); val and test are only scored.

    python code/03_train_threat_model.py [--out DIR]
    Writes outputs/03_train_threat_model/model_retrained/ (default), next to the shipped model/, which is never overwritten.
"""
import argparse, csv, json, time
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; DATA = ROOT / "data" / "splits"
csv.field_size_limit(10**9)
MODEL_ID = "distilbert-base-multilingual-cased"
MAXLEN, EPOCHS, BS, LR = 128, 3, 32, 2e-5

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "outputs" / "03_train_threat_model" / "model_retrained")); a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42); dev = "cuda" if torch.cuda.is_available() else "cpu"
    splits = {}
    for n in ("train", "val", "test"):
        rows = list(csv.DictReader(open(DATA / f"{n}.csv", encoding="utf-8"))); splits[n] = ([r["text"] for r in rows], [int(r["label"]) for r in rows])
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    def encode(X, y):
        e = tok(X, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
        return TensorDataset(e["input_ids"], e["attention_mask"], torch.tensor(y))
    ds = {n: encode(*v) for n, v in splits.items()}
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR); scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train(); tot = 0.0; dl = DataLoader(ds["train"], batch_size=BS, shuffle=True)
        for ids, mask, lab in dl:
            ids, mask, lab = ids.to(dev), mask.to(dev), lab.to(dev); opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                loss = model(input_ids=ids, attention_mask=mask, labels=lab).loss
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tot += loss.item()
        print(f"epoch {ep + 1}/{EPOCHS}  avg loss {tot / len(dl):.4f}  ({time.time() - t0:.0f}s)", flush=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)

    @torch.no_grad()
    def probs(d):
        model.eval(); out = []
        for ids, mask, _ in DataLoader(d, batch_size=64):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                out.append(torch.softmax(model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)
    for n in ("val", "test"):
        p, y = probs(ds[n]), np.array(splits[n][1]); pred = (p >= 0.5).astype(int)
        tp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
        print(n, json.dumps({"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
                             "unsure_0.3_0.6": int(((p >= 0.3) & (p < 0.6)).sum())}))
