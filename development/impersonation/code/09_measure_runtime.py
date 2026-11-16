"""Measure model sizes and per-page runtime of each step on 96 validation pages (models already loaded, one warm-up batch) -> outputs/reports/runtime.json.
Usage: python 09_measure_runtime.py [--luna]    (--luna also times 5 real Luna calls)"""

import sys, time, json, torch, numpy as np, pandas as pd, joblib, warnings

warnings.filterwarnings("ignore")
from PIL import Image
from utilities.config import *
from utilities.layout_parser import parse
from utilities.signals import signals
from utilities.rules import rule_score
from utilities.ml_model import feats

# rows = first 96 validation pages; res = results dict; sync = wait until the GPU has really finished (needed for honest timings)
rows = load_split("val")[:96]
res = {}
sync = torch.cuda.synchronize

# npar(model) = number of parameters in the model
npar = lambda m: sum(p.numel() for p in m.parameters())
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# ---- OCR: load time, parameter counts (detector + recogniser)
t = time.time()
m = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True).cuda().eval()
sync()
res["ocr_load_s"] = time.time() - t
res["ocr_det_M"] = npar(m.det_predictor.model) / 1e6
res["ocr_reco_M"] = npar(m.reco_predictor.model) / 1e6
res["ocr_params_M"] = res["ocr_det_M"] + res["ocr_reco_M"]

# ids = screenshot paths. Run one batch first to warm up the GPU (not timed), then time the remaining 80 pages in batches of 16
ids = [f"{SHOT}/{r['id']}.png" for r in rows]
m(DocumentFile.from_images(ids[:16]))
sync()
t = time.time()
for i in range(16, 96, 16):
    m(DocumentFile.from_images(ids[i : i + 16]))

# seconds per page on GPU, and peak GPU memory used so far
sync()
res["ocr_s_per_page_gpu"] = (time.time() - t) / 80
res["gpu_mem_ocr_MB"] = torch.cuda.max_memory_allocated() / 2**20

# Same OCR on CPU (4 pages only, it is slow)
mc = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True).eval()
t = time.time()
mc(DocumentFile.from_images(ids[:4]))
res["ocr_s_per_page_cpu"] = (time.time() - t) / 4

# ---- Layout parser: milliseconds per page (O = saved OCR lines)
O = json.load(open(f"{WORK}/ocr_val.json"))
t = time.time()
for r in rows:
    parse(O[r["id"]])
res["parse_ms_per_page"] = (time.time() - t) / len(rows) * 1000

# ---- CLIP: load time, total parameters, and parameters of the image tower only (the part we really use)
from transformers import CLIPModel, CLIPProcessor

t = time.time()
M = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
P = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
sync()
res["clip_load_s"] = time.time() - t
res["clip_params_M"] = npar(M) / 1e6
res["clip_image_tower_M"] = (npar(M.vision_model) + npar(M.visual_projection)) / 1e6

# crops = the avatar region of each page (feed layout box)
crops = [Image.open(f"{SHOT}/{r['id']}.png").convert("RGB").crop((80, 172, 192, 284)) for r in rows]


def run(dev, mm):
    """Embed all crops with model mm on device dev (32 at a time) and return milliseconds per page."""
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(crops), 32):
            mm.get_image_features(**P(images=crops[i : i + 32], return_tensors="pt").to(dev))
    if dev == "cuda":
        sync()
    return (time.time() - t) / len(crops) * 1000


# First call = warm-up, second call = the measurement; then the same on CPU with 8 threads
run("cuda", M)
res["clip_ms_per_page_gpu"] = run("cuda", M)
Mc = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
torch.set_num_threads(8)
run("cpu", Mc)
res["clip_ms_per_page_cpu"] = run("cpu", Mc)

# ---- Signals: milliseconds per page (A = saved avatar scores; 48 pages)
A = json.load(open(f"{WORK}/avatar_val.json"))
t = time.time()
for r in rows[:48]:
    signals(r, O[r["id"]], A[r["id"]], Image.open(f"{SHOT}/{r['id']}.png"))
res["signals_ms_per_page"] = (time.time() - t) / 48 * 1000

# ---- Rules: milliseconds per page, on the whole saved val signals table
d = pd.read_csv(f"{WORK}/signals_val.csv")
t = time.time()
rule_score(d)
res["rules_ms_per_page"] = (time.time() - t) / len(d) * 1000

# ---- LightGBM: build the features, load the frozen model, one warm-up prediction, then time a second one
X = feats(d)
mdl = joblib.load(f"{WORK}/frozen.joblib")["model"]
mdl.predict_proba(X)
t = time.time()
mdl.predict_proba(X)
res["lgbm_ms_per_page"] = (time.time() - t) / len(d) * 1000

# ---- Luna (optional, costs a little): time 5 real calls
if "--luna" in sys.argv:
    import pandas as pd
    from utilities import luna as L

    # dv = first 5 val pages that were sent to Luna; meta = csv rows by id; L._key() loads the API key
    dv = pd.read_csv(f"{WORK}/dev_val_stage.csv")
    dv = dv[dv.stage == "llm"].head(5)
    meta = {r["id"]: r for r in load_split("val")}
    L._key()

    # c = API client with no automatic retries (so the timing is honest); ts = seconds per call
    from openai import OpenAI

    c = OpenAI(max_retries=0)
    ts = []
    for _, s in dv.iterrows():
        i = L.build_input(meta[s["id"]], s)
        t = time.time()
        c.decisions.create(model=L.MODEL, input=i, questions=L.QUESTIONS, timeout=60)
        ts.append(time.time() - t)
    res["luna_s_per_call"] = ts

# Save everything; the report page reads this file
json.dump(res, open(f"{REPORTS}/runtime.json", "w"), indent=1)
print(json.dumps(res, indent=1))
