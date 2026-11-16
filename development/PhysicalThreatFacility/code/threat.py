#!/usr/bin/env python3
"""
RUNG 2 - DistilBERT threat classifier (distilbert-base-multilingual-cased, fine-tuned on the balanced train split).
Gives P(threat) for each message:  >= 0.6 -> threat  ·  < 0.3 -> not a threat  ·  0.3-0.6 -> unsure, send to Rung 3.

    from threat import ThreatModel, decide
    m = ThreatModel()                 # loads outputs/03_train_threat_model/model, GPU if available
    p = m.score(["Shots fired near 5th Ave, Seattle"])[0]     -> 0.99
    decide(p)                                                  -> "threat" | "not a threat" | "unsure"

    python threat.py "some message" ...
"""
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # the PhysicalThreatFacility folder
MODEL_DIR = ROOT / "outputs" / "03_train_threat_model" / "model"     # the trained threat model
LO, HI = 0.3, 0.6                # the unsure band, sent to Rung 3
MAXLEN, BATCH = 128, 64          # as in training

def decide(p):
    return "threat" if p >= HI else "not a threat" if p < LO else "unsure"

class ThreatModel:
    def __init__(self, model_dir=MODEL_DIR, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.dev).eval()

    @torch.no_grad()
    def score(self, texts):
        out = []
        for k in range(0, len(texts), BATCH):
            e = self.tok(texts[k:k + BATCH], truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt").to(self.dev)
            with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
                logits = self.model(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).logits.float()
            out += torch.softmax(logits, 1)[:, 1].tolist()
        return out

if __name__ == "__main__":
    m = ThreatModel()
    for t, p in zip(sys.argv[1:], m.score(sys.argv[1:])): print(f"{p:.3f} {decide(p):13s} | {t[:80]}")
