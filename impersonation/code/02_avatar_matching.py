"""Step 1c: avatar-region matching (screenshot -> avatar crop -> similarity to client logo / client executive photos / generic default-avatar). No zero-shot text prompts.
Usage: python 02_avatar_matching.py train val test [--limit N]"""

# Standard libs, numpy for pixel maths, torch for the CLIP model
import sys, json, numpy as np, torch

# PIL = image loading / cropping / drawing
from PIL import Image, ImageDraw, ImageFilter
from utilities.config import *

# parse = layout parser (we only need the platform name here); LAYOUT = platform -> layout type (feed / grid / pro)
from utilities.layout_parser import parse, LAYOUT

# CLIP = a model that turns an image into a vector; similar-looking images get similar vectors
from transformers import CLIPModel, CLIPProcessor

# Where the avatar circle sits (left, top, right, bottom in pixels) for each layout type
BOX = {"feed": (80, 172, 192, 284), "grid": (80, 88, 206, 214), "pro": (80, 164, 208, 292)}

# Load CLIP onto the GPU (M = the model, P = the preprocessor that resizes / normalises images)
dev = "cuda"
M = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
P = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# no_grad = we are only running the model, not training it (saves memory)
@torch.no_grad()
def emb(imgs):
    """Images -> unit-length CLIP vectors (one row per image)."""
    # Preprocess and move to GPU; run the CLIP image tower; some versions return an object, so fall back to .pooler_output; then scale each vector to length 1 so a dot product = cosine similarity
    x = P(images=imgs, return_tensors="pt").to(dev)
    e = M.get_image_features(**x)
    e = e if torch.is_tensor(e) else e.pooler_output
    return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()


def logo_variants(path):
    """The client logo drawn on a white square at two sizes (full and 78%), so it can match avatars that show the logo smaller."""
    # Open the logo keeping transparency; out = list of variants
    im = Image.open(path).convert("RGBA")
    out = []

    # For each scale: white 200x200 canvas, shrink a copy of the logo to fit, paste it in the centre (the logo's own alpha is the paste mask)
    for s in (1.0, 0.78):
        c = Image.new("RGB", (200, 200), "white")
        w = int(200 * s)
        l = im.copy()
        l.thumbnail((w, w))
        c.paste(l, ((200 - l.width) // 2, (200 - l.height) // 2), l)
        out.append(c)
    return out


def silhouette():
    """Draw the generic 'default avatar' (grey background, white head + shoulders)."""
    # grey square; white circle = head; white half-disc = shoulders
    c = Image.new("RGB", (200, 200), "#cfd9de")
    d = ImageDraw.Draw(c)
    d.ellipse((60, 40, 140, 120), fill="white")
    d.pieslice((15, 120, 185, 280), 180, 360, fill="white")
    return c


def mask(n=64):
    """A filled circle on an n x n grid (True inside the circle). Used so corners outside the round avatar are ignored."""
    # yy, xx = pixel coordinates; keep pixels whose distance from the centre is <= radius
    yy, xx = np.mgrid[:n, :n]
    return ((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2) <= (n / 2) ** 2


# The circle mask, built once
MK = mask()


def gray(im, n=64):
    """Image -> n x n grayscale pixel array (floats)."""
    return np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)


def ncc(a, b):
    """Normalised cross-correlation between two gray images inside the circle: 1 = same pattern, 0 = unrelated, -1 = inverted."""
    # keep only the circle pixels, subtract the mean (so brightness does not matter), d = product of the two vector lengths; return correlation (0 if an image is flat)
    a = a[MK]
    b = b[MK]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def inners(crop):  # candidate avatar-disc crops (box is generous; try a few margins)
    """Cut 5 versions of the crop with different edge margins (3%..11%), because the box is a little bigger than the real avatar circle."""
    w, h = crop.size
    return [
        crop.crop((int(w * f), int(w * f), w - int(w * f), h - int(w * f)))
        for f in (0.03, 0.05, 0.07, 0.09, 0.11)
    ]


# ---- reference data per client: the things a screenshot avatar is compared against
# REF[client] = logo variants (lv), their CLIP vectors (le) and gray images (ln), plus the same three for the executive photos (ex, ee, en)
REF = {}
for c, v in CLIENT.items():
    # lv = logo images; ex = executive photos (only those that exist)
    lv = logo_variants(v["logo"])
    ex = [Image.open(e["photo"]).convert("RGB") for e in v["execs"] if e["photo"]]
    REF[c] = dict(
        lv=lv,
        le=emb(lv),
        ln=[gray(i) for i in lv],
        ex=ex,
        ee=emb(ex) if ex else None,
        en=[gray(i) for i in ex],
    )

# The default-avatar reference: SIL image, its CLIP vector SE and its gray array SN
SIL = silhouette()
SE = emb([SIL])[0]
SN = gray(SIL)

# Optional "--limit N" quick-test flag, same as in step 01
LIM = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

# Process each split named on the command line
for sp in [a for a in sys.argv[1:] if a in ("train", "val", "test")]:
    # rows = pages; O = the OCR output from step 01 (we need it to know which platform each page is); out = results
    rows = load_split(sp)[:LIM]
    O = json.load(open(f"{WORK}/ocr_{sp}.json"))
    out = {}

    # 64 pages per batch
    for i in range(0, len(rows), 64):
        ch = rows[i : i + 64]
        crops = []
        for r in ch:
            # Find the platform from the OCR, then the avatar box for that platform's layout (default: feed)
            pl = parse(O[r["id"]])["platform"]
            box = BOX[LAYOUT.get(pl, "feed")]

            # Cut the avatar region out of the screenshot
            crops.append(Image.open(f"{SHOT}/{r['id']}.png").convert("RGB").crop(box))

        # CLIP vectors of all the crops in the batch
        E = emb(crops)
        for r, cr, e in zip(ch, crops, E):
            # R = this page's client references; ics = 5 margin crops; gs = their gray arrays; ic/g = the middle (7% margin) crop used for colour stats and the default-avatar check
            R = REF[r["customer"]]
            ics = inners(cr)
            gs = [gray(i) for i in ics]
            ic = ics[2]
            g = gs[2]

            # Tiny 32x32 version of the avatar, scaled to 0..1, to measure colourfulness
            arr = np.asarray(ic.resize((32, 32)), dtype=np.float32) / 255

            # sat = average colour spread (max channel - min channel): ~0 for grey avatars, high for colourful ones
            sat = float(np.mean(arr.max(2) - arr.min(2)))

            # Save all avatar scores for this page:
            #  logo_clip / logo_ncc = best similarity to the client logo (CLIP vector / pixel correlation over all margins)
            #  exec_clip / exec_ncc = the same against the client's executive photos (0 if the client has none)
            #  sil_clip / sil_ncc   = similarity to the generic default avatar
            #  av_sat, av_std, av_mean = colourfulness, contrast and brightness of the avatar
            out[r["id"]] = dict(
                logo_clip=float(max(R["le"] @ e)),
                logo_ncc=max(ncc(gg, n) for gg in gs for n in R["ln"]),
                exec_clip=float(max(R["ee"] @ e)) if R["ee"] is not None else 0.0,
                exec_ncc=max([ncc(gg, n) for gg in gs for n in R["en"]] or [0.0]),
                sil_clip=float(SE @ e),
                sil_ncc=ncc(g, SN),
                av_sat=sat,
                av_std=float(g[MK].std()),
                av_mean=float(g[MK].mean()),
            )

        # Progress line per batch
        print(sp, i, len(rows), flush=True)

    # Save the avatar scores for this split (read by step 04)
    json.dump(out, open(f"{WORK}/avatar_{sp}.json", "w"))
    print("saved", sp)
