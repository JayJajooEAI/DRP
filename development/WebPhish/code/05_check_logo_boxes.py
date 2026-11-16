"""Step 05 (optional check): draw the detected boxes on screenshots so detection and matching can be checked by eye
(the recipe's "logo detection precision and recall on a hand-labelled sample").

    python code/05_check_logo_boxes.py --scores logo_scores_limit50.parquet --n 50

Each output image shows the screenshot with every detected logo box (grey) and the best box for the item's customer (red,
with its similarity), next to the gallery logo it matched best.
Writes outputs/03_logo_similarity/check/<item_id>.jpg
"""
import argparse

import pandas as pd
from PIL import Image, ImageDraw

from logo_common import ITEMS, OUT, ROOT, to_rgb

THUMB = 240


def draw(item, row) -> Image.Image:
    shot = Image.open(ROOT / item["screenshot_path"]).convert("RGB")
    d = ImageDraw.Draw(shot)
    for box, det in zip(row["boxes_xyxy"], row["boxes_det_score"]):
        d.rectangle(list(box), outline=(150, 150, 150), width=3)
        d.text((box[0] + 3, box[3] + 2), f"det {det:.2f}", fill=(120, 120, 120))
    if len(row["best_box_xyxy"]):
        d.rectangle(list(row["best_box_xyxy"]), outline=(230, 0, 0), width=5)
        d.text((row["best_box_xyxy"][0] + 3, row["best_box_xyxy"][1] - 14),
               f"{row['customer_id']} sim {row['logo_similarity']:.2f}", fill=(230, 0, 0))
    scale = 1280 / shot.width
    shot = shot.resize((1280, int(shot.height * scale)))
    canvas = Image.new("RGB", (1280 + THUMB + 20, max(shot.height, THUMB + 120)), "white")
    canvas.paste(shot, (0, 0))
    cd = ImageDraw.Draw(canvas)
    cd.text((1290, 10), f"{item['item_id']}\nbest logo of {row['customer_id']}:", fill="black")
    if row["best_logo_path"]:
        logo = to_rgb(Image.open(ROOT / row["best_logo_path"]))
        logo.thumbnail((THUMB, THUMB))
        canvas.paste(logo, (1290, 50))
    cd.text((1290, THUMB + 60), f"boxes: {row['n_boxes']}\nbest brand: {row['best_brand']} "
                                f"{row['best_brand_sim']:.2f}\nmargin: {row['margin']:.2f}", fill="black")
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default="logo_scores.parquet", help="file name inside outputs/03_logo_similarity")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    scores = pd.read_parquet(OUT / args.scores)
    items = pd.read_csv(ITEMS, keep_default_na=False).set_index("item_id", drop=False)
    sample = scores.sample(min(args.n, len(scores)), random_state=args.seed)
    out = OUT / "check"
    out.mkdir(parents=True, exist_ok=True)
    for _, row in sample.iterrows():
        draw(items.loc[row["item_id"]], row).save(out / f"{row['item_id'].replace('/', '_')}.jpg", quality=85)
    print(f"{len(sample)} images -> {out}")


if __name__ == "__main__":
    main()
