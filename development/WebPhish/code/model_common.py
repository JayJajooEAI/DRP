"""Shared helpers for steps 07 and 08: shared pieces for training and inference: labels, prompt fitting, batching, metrics.

predict.py and train.py both use fit_to_length with the settings saved in each checkpoint's inference_config.json,
so a loaded model always sees prompts prepared the same way as in training.
"""
import random

import numpy as np

MODEL_ID = "jhu-clsp/mmBERT-base"
LABELS = {2: [False, True], 3: ["3.1", "3.2", "none"]}
FIELDS = ["client", "title", "headings", "brand_hints", "forms", "script_urls", "links", "visible_text", "ocr_text",
          "signals"]
CUT_ORDER = ["visible_text", "ocr_text", "script_urls", "brand_hints", "headings", "links", "forms", "title"]


def fit_to_length(row, tok, max_len: int, fields=FIELDS, cut_order=CUT_ORDER) -> str:
    """Render the prompt from its f_* field columns; if it is longer than max_len tokens, shorten fields in
    cut_order (visible text first, then OCR, ...) until it fits, so title, client, forms, links and signals survive."""
    values = {f: str(row[f"f_{f}"] or "") for f in fields}
    render = lambda: "\n".join(f"[{f.upper()}] {values[f]}" for f in fields)
    n = len(tok(render(), add_special_tokens=True)["input_ids"])
    for f in cut_order:
        if n <= max_len:
            break
        ids = tok(values[f], add_special_tokens=False)["input_ids"]
        keep = max(0, len(ids) - (n - max_len) - 8)          # small margin for re-tokenisation drift
        values[f] = tok.decode(ids[:keep]) + (" …" if keep < len(ids) else "")
        n = len(tok(render(), add_special_tokens=True)["input_ids"])
    return render()


def token_batches(lengths: np.ndarray, budget: int, max_rows: int, shuffle: bool, seed: int) -> list:
    """Groups of row indices with similar length; rows x longest length <= budget."""
    order = np.argsort(lengths, kind="stable")
    if shuffle:   # shuffle inside coarse length buckets, so batches differ between epochs but stay length-grouped
        rng = np.random.default_rng(seed)
        order = np.concatenate([rng.permutation(c) for c in np.array_split(order, max(1, len(order) // 64))])
        order = order[np.argsort(lengths[order], kind="stable")]
    batches, cur, longest = [], [], 0
    for i in order:
        longest_new = max(longest, lengths[i])
        if cur and (longest_new * (len(cur) + 1) > budget or len(cur) >= max_rows):
            batches.append(cur)
            cur, longest_new = [], lengths[i]
        cur.append(int(i))
        longest = longest_new
    if cur:
        batches.append(cur)
    if shuffle:
        random.Random(seed).shuffle(batches)
    return batches


def collate(ids_list, rows, pad_id: int, device: str):
    import torch
    L = max(len(ids_list[i]) for i in rows)
    x = torch.full((len(rows), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(rows), L), dtype=torch.long)
    for j, i in enumerate(rows):
        x[j, :len(ids_list[i])] = torch.tensor(ids_list[i])
        m[j, :len(ids_list[i])] = 1
    return x.to(device), m.to(device)


def predict_probs(model, ids_list, n_labels: int, pad_id: int, device: str, budget: int, max_rows: int) -> np.ndarray:
    import torch
    was_training = model.training
    model.eval()
    lengths = np.array([len(x) for x in ids_list])
    prob = np.zeros((len(ids_list), n_labels), np.float32)
    with torch.no_grad():
        for rows in token_batches(lengths, budget, max_rows, False, 0):
            x, m = collate(ids_list, rows, pad_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(input_ids=x, attention_mask=m).logits
            prob[rows] = torch.softmax(logits.float(), -1).cpu().numpy()
    model.train(was_training)
    return prob


def metrics(check: int, y: np.ndarray, prob: np.ndarray) -> dict:
    """Every metric for one set of rows: accuracy, macro / weighted F1, per-class precision / recall / F1 / support,
    confusion matrix (rows = true, columns = predicted, label order LABELS[check]), log loss, and for check 2 ROC AUC,
    average precision and false-positive rate (not-branded rows predicted branded)."""
    from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, log_loss,
                                 precision_recall_fscore_support, roc_auc_score)
    if len(y) == 0:
        return {"n": 0}
    pred = prob.argmax(1)
    names = [str(l) for l in LABELS[check]]
    labels = list(range(len(names)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    out = {"n": int(len(y)), "accuracy": float((pred == y).mean()),
           "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
           "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
           "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1), labels=labels)),
           "per_class": {n: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
                         for k, n in enumerate(names)},
           "confusion": confusion_matrix(y, pred, labels=labels).tolist(), "confusion_labels": names}
    if check == 2:
        neg = y == 0
        out["false_positive_rate"] = float((pred[neg] == 1).mean()) if neg.any() else None
        out["false_negative_rate"] = float((pred[~neg] == 0).mean()) if (~neg).any() else None
        if len(set(y)) == 2:
            out["roc_auc"] = float(roc_auc_score(y, prob[:, 1]))
            out["average_precision_branded"] = float(average_precision_score(y, prob[:, 1]))
            out["average_precision_not_branded"] = float(average_precision_score(1 - y, prob[:, 0]))
    return out


def format_metrics(name: str, m: dict) -> str:
    """Readable block for the log."""
    if not m.get("n"):
        return f"  [{name}] no rows"
    head = (f"  [{name}] n={m['n']} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} "
            f"weighted_f1={m['weighted_f1']:.4f} log_loss={m['log_loss']:.4f}")
    for k in ("roc_auc", "average_precision_branded", "average_precision_not_branded", "false_positive_rate",
              "false_negative_rate"):
        if m.get(k) is not None:
            head += f" {k}={m[k]:.4f}"
    lines = [head]
    for c, v in m["per_class"].items():
        lines.append(f"      {c:>6}: P={v['precision']:.4f} R={v['recall']:.4f} F1={v['f1']:.4f} n={v['support']}")
    lines.append(f"      confusion (rows true, cols pred {m['confusion_labels']}): {m['confusion']}")
    return "\n".join(lines)
