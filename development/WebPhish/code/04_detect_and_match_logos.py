"""Step 04: find logo boxes in each screenshot (Phishpedia's Faster R-CNN) and score them against
the customer logo gallery (Phishpedia's siamese matcher, cosine similarity).

    python code/04_detect_and_match_logos.py --limit 50      # quick check
    python code/04_detect_and_match_logos.py --tag s50 --ids ...  # chosen items
    python code/04_detect_and_match_logos.py                 # all items, resumable
    python code/04_detect_and_match_logos.py --merge-only    # rebuild from shards

Needs logo_gallery.npz from build_gallery.py.

For each item the grid "boxes x logos" is reduced in two directions:
  per item  : each box's best score against a brand's logos, then max / mean / median / min over the boxes
  best box  : the best box's scores against every logo of that brand, then max / mean / median / min
`logo_similarity` (the value for the prompt) is the per-item max for the item's own customer. Items where no box
was found keep NaN (unresolved, never 0).

Writes to outputs/03_logo_similarity/:
  logo_scores.parquet       one row per item: the item's own customer stats, best brand, margin, boxes
  logo_brand_stats.parquet  one row per item x brand (all 25): the same 8 stats, for the wrong-client pairs
  logo_sims_raw.npz         every box x every gallery logo score (float16), to recompute any statistic later
"""
import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from PIL import Image

from logo_common import (CROP_PAD, DETECT_THRESHOLD, DETECTOR_ID, EMBEDDER_ID, GALLERY, ITEMS, LOGO_CLASS, MIN_BOX_PX,
                    OUT, ROOT, Detector, Embedder)

SHARDS = OUT / "shards"
CPU_THREADS = 2
STAT_NAMES = ["sim_max", "sim_mean", "sim_median", "sim_min",
              "best_box_logo_max", "best_box_logo_mean", "best_box_logo_median", "best_box_logo_min"]


def clean_boxes(boxes: np.ndarray, scores: np.ndarray, w: int, h: int):
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    ok = ((boxes[:, 2] - boxes[:, 0]) >= MIN_BOX_PX) & ((boxes[:, 3] - boxes[:, 1]) >= MIN_BOX_PX)
    order = np.argsort(-scores[ok])
    return boxes[ok][order], scores[ok][order]


def crop(img: Image.Image, box) -> Image.Image:
    x0, y0, x1, y1 = box
    px, py = (x1 - x0) * CROP_PAD, (y1 - y0) * CROP_PAD
    return img.crop((max(0, x0 - px), max(0, y0 - py), min(img.width, x1 + px), min(img.height, y1 + py)))


def stats_for(sims: np.ndarray, cols: np.ndarray) -> dict:
    """The 8 statistics of one item against one brand's logo columns (NaN when there are no boxes)."""
    if sims.shape[0] == 0 or len(cols) == 0:
        return {**{k: np.nan for k in STAT_NAMES}, "best_box": -1, "best_logo": -1}
    sub = sims[:, cols]
    per_box = sub.max(axis=1)
    b = int(per_box.argmax())
    row = sub[b]
    return {"sim_max": float(per_box.max()), "sim_mean": float(per_box.mean()),
            "sim_median": float(np.median(per_box)), "sim_min": float(per_box.min()),
            "best_box_logo_max": float(row.max()), "best_box_logo_mean": float(row.mean()),
            "best_box_logo_median": float(np.median(row)), "best_box_logo_min": float(row.min()),
            "best_box": b, "best_logo": int(cols[int(row.argmax())])}


def summarise(item, boxes, det, sims, gallery) -> tuple[dict, list]:
    customers, groups, paths = gallery["customer_id"], gallery["corporate_group"], gallery["path"]
    brand_rows, by_brand = [], {}
    for brand in sorted(set(customers)):
        st = stats_for(sims, np.flatnonzero(customers == brand))
        by_brand[brand] = st
        brand_rows.append({"item_id": item["item_id"], "brand": brand, **{k: st[k] for k in STAT_NAMES},
                           "best_box": st["best_box"]})
    own = by_brand.get(item["customer_id"], stats_for(sims, np.array([], int)))
    own_group = groups[customers == item["customer_id"]][0] if (customers == item["customer_id"]).any() else ""
    ranked = sorted(((v["sim_max"], k) for k, v in by_brand.items() if not np.isnan(v["sim_max"])), reverse=True)
    other = [(s, k) for s, k in ranked if groups[customers == k][0] != own_group]
    per_box_own = sims[:, customers == item["customer_id"]].max(axis=1) if len(sims) else np.array([])
    row = {
        "item_id": item["item_id"], "customer_id": item["customer_id"],
        "logo_similarity": own["sim_max"],
        **{k: own[k] for k in STAT_NAMES},
        "top3_box_sims": sorted(per_box_own.tolist(), reverse=True)[:3],
        "n_boxes": int(len(boxes)), "n_boxes_above_0.5": int((per_box_own > 0.5).sum()),
        "best_box_xyxy": [round(float(v), 1) for v in boxes[own["best_box"]]] if own["best_box"] >= 0 else [],
        "best_box_det_score": float(det[own["best_box"]]) if own["best_box"] >= 0 else np.nan,
        "best_logo_path": str(paths[own["best_logo"]]) if own["best_logo"] >= 0 else "",
        "best_brand": ranked[0][1] if ranked else "", "best_brand_sim": ranked[0][0] if ranked else np.nan,
        # best brand outside the item's company: Office365 / Outlook / OneDrive share Microsoft logos
        "best_other_brand": other[0][1] if other else "", "best_other_brand_sim": other[0][0] if other else np.nan,
        "margin": own["sim_max"] - other[0][0] if other and not np.isnan(own["sim_max"]) else np.nan,
        "boxes_xyxy": [[round(float(v), 1) for v in b] for b in boxes], "boxes_det_score": [float(s) for s in det],
    }
    return row, brand_rows


def write_shard(name, rows, brand_rows, raw, n_logos):
    SHARDS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(SHARDS / f"{name}_scores.parquet", index=False)
    pd.DataFrame(brand_rows).to_parquet(SHARDS / f"{name}_brands.parquet", index=False)
    np.savez_compressed(SHARDS / f"{name}_raw.npz",
                        item_id=np.array(raw["item_id"]), boxes=np.array(raw["boxes"], np.float32).reshape(-1, 4),
                        det_score=np.array(raw["det"], np.float32),
                        sims=np.array(raw["sims"], np.float16).reshape(len(raw["item_id"]), n_logos))


def merge(suffix=""):
    """Combine all shards into the three final files."""
    scores = pd.concat([pd.read_parquet(p) for p in sorted(SHARDS.glob("*_scores.parquet"))], ignore_index=True)
    brands = pd.concat([pd.read_parquet(p) for p in sorted(SHARDS.glob("*_brands.parquet"))], ignore_index=True)
    raws = [np.load(p) for p in sorted(SHARDS.glob("*_raw.npz"))]
    scores.drop_duplicates("item_id", keep="last").to_parquet(OUT / f"logo_scores{suffix}.parquet", index=False)
    brands.drop_duplicates(["item_id", "brand"], keep="last").to_parquet(OUT / f"logo_brand_stats{suffix}.parquet",
                                                                          index=False)
    g = np.load(GALLERY)
    np.savez_compressed(OUT / f"logo_sims_raw{suffix}.npz",
                        box_item_id=np.concatenate([r["item_id"] for r in raws]),
                        boxes=np.concatenate([r["boxes"] for r in raws]),
                        det_score=np.concatenate([r["det_score"] for r in raws]),
                        sims=np.concatenate([r["sims"] for r in raws]),
                        gallery_path=g["path"], gallery_customer_id=g["customer_id"])
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="first N items only (written with a _limitN suffix, not resumed)")
    ap.add_argument("--ids", nargs="*", help="only these item_ids")
    ap.add_argument("--tag", help="name for a test run's files (e.g. sample50), kept apart from the full run")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--merge-only", action="store_true")
    args = ap.parse_args()
    global SHARDS
    suffix = f"_{args.tag}" if args.tag else (f"_limit{args.limit}" if args.limit else "")
    if suffix:
        SHARDS = OUT / f"shards{suffix}"
    if args.merge_only:
        print(len(merge(suffix)), "items merged")
        return

    import torch
    torch.set_num_threads(CPU_THREADS)   # shared VM: keep CPU use small, the models run on the GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gallery = dict(np.load(GALLERY))
    gallery_emb = gallery["emb"]
    assert str(gallery["embedder"]) == EMBEDDER_ID, "gallery was built with another embedder: re-run build_gallery.py"

    items = pd.read_csv(ITEMS, keep_default_na=False)
    if args.ids:
        items = items[items["item_id"].isin(args.ids)]
    if args.limit:
        items = items.head(args.limit)
    done = set()
    for p in SHARDS.glob("*_scores.parquet"):
        done |= set(pd.read_parquet(p, columns=["item_id"])["item_id"])
    todo = items[~items["item_id"].isin(done)].to_dict("records")
    print(f"{len(items)} items, {len(done & set(items['item_id']))} already done, {len(todo)} to run on {device}")

    detector, embedder = Detector(device), Embedder(device)
    t0, n_shard = time.time(), len(list(SHARDS.glob("*_scores.parquet")))
    rows, brand_rows, raw, errors = [], [], {"item_id": [], "boxes": [], "det": [], "sims": []}, []
    for k, it in enumerate(todo):
        try:
            img = Image.open(ROOT / it["screenshot_path"]).convert("RGB")
        except Exception as exc:
            errors.append({"item_id": it["item_id"], "error": f"{type(exc).__name__}: {exc}"[:200]})
            img = None
        if img is not None:
            boxes, det = clean_boxes(*detector(img), img.width, img.height)
            crops = [crop(img, b) for b in boxes]
            sims = embedder(crops) @ gallery_emb.T if crops else np.zeros((0, gallery_emb.shape[0]), np.float32)
            row, brows = summarise(it, boxes, det, sims, gallery)
            rows.append(row); brand_rows += brows
            raw["item_id"] += [it["item_id"]] * len(boxes); raw["boxes"] += boxes.tolist()
            raw["det"] += det.tolist(); raw["sims"] += sims.tolist()
        if rows and (len(rows) >= args.shard_size or k == len(todo) - 1):
            write_shard(f"shard_{n_shard:04d}", rows, brand_rows, raw, gallery_emb.shape[0])
            n_shard += 1
            print(f"  {k + 1}/{len(todo)} items, {(k + 1) / (time.time() - t0):.2f} items/s", flush=True)
            rows, brand_rows, raw = [], [], {"item_id": [], "boxes": [], "det": [], "sims": []}

    scores = merge(suffix) if list(SHARDS.glob("*_scores.parquet")) else pd.DataFrame()
    info = {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "items": len(scores),
            "no_boxes": int((scores.get("n_boxes", pd.Series(dtype=int)) == 0).sum()), "image_errors": errors,
            "detector": DETECTOR_ID, "embedder": EMBEDDER_ID, "logo_class": LOGO_CLASS,
            "detect_threshold": DETECT_THRESHOLD, "min_box_px": MIN_BOX_PX, "crop_pad": CROP_PAD, "device": device,
            "seconds": round(time.time() - t0, 1)}
    (OUT / f"run_info{suffix}.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: v for k, v in info.items() if k != "image_errors"}, indent=1), f"{len(errors)} image errors")


if __name__ == "__main__":
    main()
