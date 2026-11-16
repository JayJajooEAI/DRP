#!/usr/bin/env python3
"""
Step 2 of training the Rung 1 location model: supervised fine-tuning of MiniLMv2 to mark the same words GLiNER marked.

- Base model: nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large (6 layers, 384 hidden, multilingual)
- Task: word tagger, 3 tags per word piece: O (not a place), B (start of a place), I (inside a place)
- Labels: teacher_train.jsonl (GLiNER's places, from make_teacher_labels.py; the files used for the shipped model are included)
- Training: 8 epochs, AdamW lr 5e-5, weight decay 0.01, OneCycle (10% warm-up), batch 32, fp16, seed 0; loss = cross-entropy
- Cut-off (chosen on val): the tau with the fewest val threats lost, as long as no more are lost than GLiNER + place rules
  lose, and no more than 10% more val non-threats pass than with GLiNER (val: 17 lost vs 23, 487 passing vs 456 -> 0.4)

    python code/02_train_location_model.py [--epochs 8] [--teacher-dir DIR] [--out DIR]
    Reads the teacher labels (default data/teacher_labels/) and data/splits/val.csv. Writes outputs/02_train_location_model/model_retrained/
    (default), next to the shipped model/, which is never overwritten.
"""
import argparse, json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); import location as LOC
BASE = "nreimers/mMiniLMv2-L6-H384-distilled-from-XLMR-Large"; MAXLEN, BATCH, LR = 384, 32, 5e-5

TEACHER_DIR = ROOT / "data" / "teacher_labels"
def load(split): return [json.loads(l) for l in open(TEACHER_DIR / f"teacher_{split}.jsonl", encoding="utf-8")]

def encode(tok, chunks):
    enc = tok([c["text"] for c in chunks], truncation=True, max_length=MAXLEN, return_offsets_mapping=True); labs = []
    for c, offs in zip(chunks, enc["offset_mapping"]):
        y, prev = [], False
        for s, e in offs:
            if s == e: y.append(-100); prev = False; continue                     # special tokens: no loss
            inside = any(s < se and e > ss for ss, se, _ in c.get("spans", []))
            y.append((2 if prev else 1) if inside else 0); prev = inside           # 0 = O, 1 = B, 2 = I
        labs.append(y)
    return enc, labs

def batches(tok, enc, labs, order):
    for k in range(0, len(order), BATCH):
        idx = order[k:k + BATCH]; n = max(len(enc["input_ids"][i]) for i in idx)
        ids = torch.full((len(idx), n), tok.pad_token_id); att = torch.zeros((len(idx), n), dtype=torch.long); y = torch.full((len(idx), n), -100)
        for j, i in enumerate(idx):
            L = len(enc["input_ids"][i]); ids[j, :L] = torch.tensor(enc["input_ids"][i]); att[j, :L] = 1; y[j, :L] = torch.tensor(labs[i])
        yield ids.cuda(), att.cuda(), y.cuda()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--teacher-dir", default=str(TEACHER_DIR)); ap.add_argument("--out", default=str(ROOT / "outputs" / "02_train_location_model" / "model_retrained"))
    a = ap.parse_args(); TEACHER_DIR = Path(a.teacher_dir); torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    trc = [c for r in load("train") for c in r["chunks"]]; enc, labs = encode(tok, trc)
    model = AutoModelForTokenClassification.from_pretrained(BASE, num_labels=3, dtype=torch.float32).cuda()   # the checkpoint is stored in fp16
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = a.epochs * ((len(trc) + BATCH - 1) // BATCH); sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    for ep in range(a.epochs):
        model.train(); order = torch.randperm(len(trc), generator=torch.Generator().manual_seed(ep)).tolist(); tot = 0
        for ids, att, y in batches(tok, enc, labs, order):
            with torch.autocast("cuda", dtype=torch.float16): loss = model(input_ids=ids, attention_mask=att, labels=y).loss
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot / (len(trc) / BATCH):.4f}", flush=True)
    model.save_pretrained(a.out); tok.save_pretrained(a.out)

    # cut-off on val
    import csv; csv.field_size_limit(10**9)
    va = load("val"); texts = {r["id"]: r["text"] for r in csv.DictReader(open(ROOT / "data/splits/val.csv", encoding="utf-8"))}
    m = LOC.LocationModel(a.out); vac = [c["text"] for r in va for c in r["chunks"]]; _, vp = m._probs(vac)
    score, k = [], 0
    for r in va:
        s = 0.0
        for _ in r["chunks"]: s = max(s, max(vp[k], default=0.0)); k += 1
        score.append(s)
    rule = [bool(LOC.place_rule(texts[r["id"]])) for r in va]
    teacher = [any(c["spans"] for c in r["chunks"]) or rule[i] for i, r in enumerate(va)]
    g_lost = sum(1 for i, r in enumerate(va) if r["label"] == 1 and not teacher[i])
    g_pass = sum(1 for i, r in enumerate(va) if r["label"] == 0 and teacher[i]); pick = None
    for t in [0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
        lost = sum(1 for i, r in enumerate(va) if r["label"] == 1 and not (score[i] >= t or rule[i]))
        npass = sum(1 for i, r in enumerate(va) if r["label"] == 0 and (score[i] >= t or rule[i]))
        ok = lost <= g_lost and npass <= 1.1 * g_pass
        print(f"val tau {t}: threats lost {lost} (GLiNER {g_lost}), non-threats passing {npass} (GLiNER {g_pass})" + ("" if ok else "  x"))
        if ok and (pick is None or lost < pick[1]): pick = (t, lost)
    print("-> use tau", pick and pick[0], "(set TAU in code/location.py)")
