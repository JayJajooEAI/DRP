"""Step 1c: avatar matching. Crop the avatar from its fixed box, then compare it with the client's logo, the client's executive photos (their CLIP
vectors are stored in RDS) and a generic default avatar, using CLIP (meaning-level similarity) and masked pixel correlation (shape-level similarity). Same maths as the batch pipeline."""

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor

from pipeline.layout_parser import LAYOUT
from pipeline.settings import DEVICE

# Where the avatar circle sits (left, top, right, bottom in pixels) for each layout type
BOX = {"feed": (80, 172, 192, 284), "grid": (80, 88, 206, 214), "pro": (80, 164, 208, 292)}
CLIP_NAME = "openai/clip-vit-base-patch32"


def _logo_variants(image):
    """The logo on a white 200x200 square at two sizes (full and 78%)."""
    im = image.convert("RGBA")
    out = []
    for s in (1.0, 0.78):
        c = Image.new("RGB", (200, 200), "white")
        w = int(200 * s)
        l = im.copy()
        l.thumbnail((w, w))
        c.paste(l, ((200 - l.width) // 2, (200 - l.height) // 2), l)
        out.append(c)
    return out


def _silhouette():
    """The generic default avatar: grey background, white head and shoulders."""
    c = Image.new("RGB", (200, 200), "#cfd9de")
    d = ImageDraw.Draw(c)
    d.ellipse((60, 40, 140, 120), fill="white")
    d.pieslice((15, 120, 185, 280), 180, 360, fill="white")
    return c


def _circle_mask(n=64):
    """True inside a filled circle on an n x n grid, so the corners of the square crop are ignored."""
    yy, xx = np.mgrid[:n, :n]
    return ((xx - n / 2 + 0.5) ** 2 + (yy - n / 2 + 0.5) ** 2) <= (n / 2) ** 2


MK = _circle_mask()


def _gray(im, n=64):
    """Image -> n x n grayscale float array."""
    return np.asarray(im.convert("L").resize((n, n), Image.BILINEAR), dtype=np.float32)


def _ncc(a, b):
    """Normalised cross-correlation inside the circle: 1 = same pattern, 0 = unrelated, -1 = inverted."""
    a = a[MK]
    b = b[MK]
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


def _inners(crop):
    """5 versions of the crop with different edge margins (the box is a little bigger than the real avatar circle)."""
    w, h = crop.size
    return [crop.crop((int(w * f), int(w * f), w - int(w * f), h - int(w * f))) for f in (0.03, 0.05, 0.07, 0.09, 0.11)]


class AvatarMatcher:
    def __init__(self):
        # CLIP is loaded once, to embed the avatar crop of each incoming screenshot (the references come from RDS)
        use_cuda = torch.cuda.is_available() and DEVICE in ("auto", "cuda")
        self.dev = "cuda" if use_cuda else "cpu"
        self.model = CLIPModel.from_pretrained(CLIP_NAME).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_NAME)

        # The default-avatar reference
        sil = _silhouette()
        self.se = self._emb([sil])[0]
        self.sn = _gray(sil)

    @torch.no_grad()
    def _emb(self, imgs):
        """Images -> unit-length CLIP vectors (one row per image)."""
        x = self.proc(images=imgs, return_tensors="pt").to(self.dev)
        e = self.model.get_image_features(**x)
        e = e if torch.is_tensor(e) else e.pooler_output
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy()

    @staticmethod
    def _reference(client):
        """The client's comparison data: stored CLIP vectors (le, ee) plus gray images for the pixel check (ln, en). Built once per fetched client."""
        if "_ref" not in client:
            ph = client["photos"]
            client["_ref"] = dict(
                le=client["logo_vecs"],
                ln=[_gray(i) for i in _logo_variants(client["logo_image"])],
                ee=np.stack([e for _, e in ph]) if ph else None,
                en=[_gray(i) for i, _ in ph],
            )
        return client["_ref"]

    def match(self, img, platform, client):
        """img = the screenshot (PIL RGB, 900x900), client = the profile from RDS. Returns the avatar scores used by the signals step."""
        # Crop the avatar from the box that belongs to this platform's layout
        crop = img.crop(BOX[LAYOUT.get(platform, "feed")])
        e = self._emb([crop])[0]
        R = self._reference(client)

        # ics = 5 margin crops, gs = their gray arrays; the middle one (7% margin) is used for the colour stats and the default-avatar check
        ics = _inners(crop)
        gs = [_gray(i) for i in ics]
        ic = ics[2]
        g = gs[2]

        # sat = average colour spread of a 32x32 version of the avatar (about 0 for grey avatars)
        arr = np.asarray(ic.resize((32, 32)), dtype=np.float32) / 255
        sat = float(np.mean(arr.max(2) - arr.min(2)))
        return dict(
            logo_clip=float(max(R["le"] @ e)),
            logo_ncc=max(_ncc(gg, n) for gg in gs for n in R["ln"]),
            exec_clip=float(max(R["ee"] @ e)) if R["ee"] is not None else 0.0,
            exec_ncc=max([_ncc(gg, n) for gg in gs for n in R["en"]] or [0.0]),
            sil_clip=float(self.se @ e),
            sil_ncc=_ncc(g, self.sn),
            av_sat=sat,
            av_std=float(g[MK].std()),
            av_mean=float(g[MK].mean()),
        )
