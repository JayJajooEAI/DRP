"""Step 6 of a request: the two fine-tuned mmBERT models read the prompt.

check 2 (branded):      does the page present itself as the client?          labels [False, True]
check 3 (data capture): is there a way to enter credentials or payment data? labels [3.1, 3.2, none]
                        3.1 = a text box for credentials / payment details; 3.2 = a login link or button; none = neither

Each model's folder holds its weights and `inference_config.json` (label order, maximum length, field order and the order in which fields
are shortened when the prompt is too long). The prompt is fitted to length exactly as in training, so a model sees what it was trained on.
"""

import json

import numpy as np

from WebPhish.settings import CHECK_DIRS, DEVICE


def fit_to_length(fields: dict, tok, max_len: int, order, cut_order) -> str:
    """Render the prompt; if it is longer than max_len tokens, shorten fields in cut_order until it fits."""
    values = {f: str(fields.get(f) or "") for f in order}
    render = lambda: "\n".join(f"[{f.upper()}] {values[f]}" for f in order)
    n = len(tok(render(), add_special_tokens=True)["input_ids"])
    for f in cut_order:
        if n <= max_len:
            break
        ids = tok(values[f], add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - (n - max_len) - 8)   # small margin for re-tokenisation drift
        values[f] = tok.decode(ids[:keep]) + (" …" if keep < len(ids) else "")
        n = len(tok(render(), add_special_tokens=True)["input_ids"])
    return render()


class Checker:
    """One fine-tuned model (check 2 or check 3)."""

    def __init__(self, check: int):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        d = CHECK_DIRS[check]
        self.torch = torch
        self.check = check
        self.cfg = json.loads((d / "inference_config.json").read_text())
        self.device = "cuda" if torch.cuda.is_available() and DEVICE in ("auto", "cuda") else "cpu"
        self.tok = AutoTokenizer.from_pretrained(d / "model")
        self.model = AutoModelForSequenceClassification.from_pretrained(d / "model", attn_implementation="sdpa").to(self.device).eval()

    def predict(self, fields: dict) -> dict:
        """fields = the prompt fields. Returns {label: probability} for this model's labels, and the chosen label."""
        torch, cfg = self.torch, self.cfg
        text = fit_to_length(fields, self.tok, cfg["max_len"], cfg["fields"], cfg["cut_order"])
        ids = self.tok(text, add_special_tokens=True, truncation=True, max_length=cfg["max_len"], return_tensors="pt")
        ids = {k: v.to(self.device) for k, v in ids.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
            logits = self.model(**ids).logits
        p = torch.softmax(logits.float(), -1)[0].cpu().numpy()
        labels = cfg["labels"]
        return {"probs": {l: round(float(v), 4) for l, v in zip(labels, p)}, "label": labels[int(np.argmax(p))],
                "n_tokens": int(ids["input_ids"].shape[1])}
