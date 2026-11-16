"""Step 02: OCR every item's screenshot with docTR (Mindee, Apache-2.0, PyTorch on the GPU).

Replaces data/ocr/ (RapidOCR with Baidu's PP-OCR models) with a non-Chinese-origin engine; data/ocr/ is left as it
was. Writes one JSON per item in the same shape as data/ocr/ so later steps read either:
  {"engine", "image", "width", "height", "lines": [{"text", "score", "box": [[x,y] x 4 corners, pixels]}]}
`score` is the mean word confidence of the line. Lines are in docTR's reading order (top to bottom by block).

    python code/02_ocr_read_screenshots.py --limit 50     # quick check -> outputs/02_ocr_read_screenshots_limit50/
    python code/02_ocr_read_screenshots.py                # all items; resumes, skips finished items

Reads  data/benchmark/items.csv and each item's screenshot_path.
Writes outputs/02_ocr_read_screenshots/<item_id>.json, ocr_index.parquet (one row per item) and run_info.json.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "benchmark" / "items.csv"
OUT = ROOT / "outputs" / "02_ocr_read_screenshots"

DET_ARCH = "db_resnet50"     # text detection
RECO_ARCH = "parseq"         # text recognition
CPU_THREADS = 2              # shared VM: the models run on the GPU, keep CPU use small
BATCH = 8                    # screenshots per predictor call


def item_path(out: Path, item_id: str) -> Path:
    return out / f"{item_id}.json"


def to_lines(page, w: int, h: int) -> list:
    lines = []
    for block in page.blocks:
        for line in block.lines:
            words = [wd for wd in line.words if wd.value.strip()]
            if not words:
                continue
            (x0, y0), (x1, y1) = line.geometry   # relative to the page
            x0, x1, y0, y1 = round(x0 * w, 1), round(x1 * w, 1), round(y0 * h, 1), round(y1 * h, 1)
            lines.append({"text": " ".join(wd.value for wd in words),
                          "score": round(float(np.mean([wd.confidence for wd in words])), 4),
                          "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]})
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N items only, written to outputs/02_ocr_read_screenshots_limitN/")
    ap.add_argument("--ids", nargs="*", help="only these item_ids")
    args = ap.parse_args()
    out = OUT.parent / f"02_ocr_read_screenshots_limit{args.limit}" if args.limit else OUT

    import torch
    from doctr.models import ocr_predictor
    torch.set_num_threads(CPU_THREADS)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    items = pd.read_csv(ITEMS, keep_default_na=False)
    if args.ids:
        items = items[items["item_id"].isin(args.ids)]
    if args.limit:
        items = items.head(args.limit)
    todo = [r for r in items.to_dict("records") if not item_path(out, r["item_id"]).exists()]
    print(f"{len(items)} items, {len(items) - len(todo)} already done, {len(todo)} to run on {device}", flush=True)

    model = ocr_predictor(det_arch=DET_ARCH, reco_arch=RECO_ARCH, pretrained=True).to(device).eval()
    engine = f"doctr {__import__('doctr').__version__} ({DET_ARCH} + {RECO_ARCH})"

    t0, errors = time.time(), []
    for start in range(0, len(todo), BATCH):
        batch, images = [], []
        for r in todo[start:start + BATCH]:
            try:
                images.append(np.asarray(Image.open(ROOT / r["screenshot_path"]).convert("RGB")))
                batch.append(r)
            except Exception as exc:
                errors.append({"item_id": r["item_id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
        if not images:
            continue
        with torch.no_grad():
            result = model(images)
        for r, img, page in zip(batch, images, result.pages):
            h, w = img.shape[:2]
            rec = {"engine": engine, "image": r["screenshot_path"], "width": w, "height": h,
                   "lines": to_lines(page, w, h)}
            path = item_path(out, r["item_id"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rec, ensure_ascii=False))
        done = start + len(batch)
        if done % 200 < BATCH or done == len(todo):
            print(f"  {done}/{len(todo)} items, {done / (time.time() - t0):.2f} items/s", flush=True)

    # index of every item's OCR file (finished in this run or earlier)
    rows = []
    for r in items.to_dict("records"):
        path = item_path(out, r["item_id"])
        if path.exists():
            lines = json.loads(path.read_text())["lines"]
            rows.append({"item_id": r["item_id"], "ocr_path": str(path.relative_to(ROOT)), "n_lines": len(lines),
                         "n_chars": sum(len(l["text"]) for l in lines),
                         "mean_score": float(np.mean([l["score"] for l in lines])) if lines else np.nan})
    index = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    index.to_parquet(out / "ocr_index.parquet", index=False)
    info = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "engine": engine, "device": device,
            "items": len(index), "empty": int((index["n_lines"] == 0).sum()) if len(index) else 0,
            "image_errors": errors, "seconds": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "image_errors"}, indent=1), f"{len(errors)} image errors")


if __name__ == "__main__":
    main()
