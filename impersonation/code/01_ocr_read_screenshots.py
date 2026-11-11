"""Step 1a: docTR OCR on every screenshot -> outputs/work/ocr_<split>.json (text lines with position, height, confidence).
Usage: python 01_ocr_read_screenshots.py train val test [--limit N]"""

# sys: read the command line; json: save the result; os/time/torch: timing + GPU support
import sys, json, os, time, torch

# config gives us: load_split (reads the split csv), SHOT (screenshot folder), WORK (output folder)
from utilities.config import *

# docTR = the OCR library; DocumentFile loads image files, ocr_predictor builds the OCR model
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Build the OCR model: db_resnet50 finds the text boxes, crnn_vgg16_bn reads the characters. Move it to the GPU and set eval mode (no training).
m = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True).cuda().eval()

# Optional "--limit N" flag: only read the first N pages of each split (handy for a quick test). None = read everything.
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

# Loop over every split name given on the command line (train / val / test)
for sp in [a for a in sys.argv[1:] if a in ("train", "val", "test")]:
    # rows = the pages of this split; out = results dict (page id -> OCR lines); t0 = start time; B = pages per OCR batch
    rows = load_split(sp)[:LIM]
    out = {}
    t0 = time.time()
    B = 16

    # Go through the pages 16 at a time (batching makes the GPU faster)
    for i in range(0, len(rows), B):
        # ch = this batch of rows; docs = their screenshots loaded as images
        ch = rows[i : i + B]
        docs = DocumentFile.from_images([f"{SHOT}/{r['id']}.png" for r in ch])

        # Run the OCR model on the whole batch
        res = m(docs)

        # Match each page of the result back to its row
        for r, pg in zip(ch, res.pages):
            # ls = the text lines found on this page
            ls = []

            # docTR groups text into blocks, and blocks into lines
            for b in pg.blocks:
                for l in b.lines:
                    # Line corners are fractions of the image (0..1); (x0,y0) = top-left, (x1,y1) = bottom-right
                    (x0, y0), (x1, y1) = l.geometry

                    # Store one line: x/y = top-left pixel, x1 = right edge, h = height, c = mean word confidence, t = the text. The screenshots are 900 px wide, so fractions * 900 = pixels.
                    ls.append(
                        dict(
                            x=int(x0 * 900),
                            y=int(y0 * 900),
                            x1=int(x1 * 900),
                            h=int((y1 - y0) * 900),
                            c=round(sum(w.confidence for w in l.words) / len(l.words), 3),
                            t=" ".join(w.value for w in l.words),
                        )
                    )

            # Save this page's lines sorted top-to-bottom, then left-to-right (reading order)
            out[r["id"]] = sorted(ls, key=lambda a: (a["y"], a["x"]))

        # Print progress every 20 batches
        if (i // B) % 20 == 0:
            print(sp, i, len(rows), round(time.time() - t0), "s", flush=True)

    # Write the whole split's OCR results to disk, so later steps never need to run OCR again
    json.dump(out, open(f"{WORK}/ocr_{sp}.json", "w"))
    print("saved", sp, len(out), flush=True)
