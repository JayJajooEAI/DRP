"""Step 3 of a request: find the logos on the page screenshot and compare them with the client's own logos.

Detector: Phishpedia's Faster R-CNN layout detector (detectron2; class 1 is "logo"). Matcher: Phishpedia's siamese ResNetV2 (BiT-M-R50x1),
used for both the client's logos and the crops of the page, so their embeddings are comparable (cosine similarity). Phishpedia:
github.com/lindsey98/Phishpedia (CC0). Same code and settings as the batch pipeline (code/03 and 04). Weights are loaded with
weights_only=True (plain tensors, no pickled code).

`logo_similarity` = the best match between any detected logo box and any of the client's logos; NaN when no box was found (never 0).
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from WebPhish.phishpedia_bit import KNOWN_MODELS
from WebPhish.settings import DEVICE, PHISHPEDIA_DIR

DETECTOR_CFG = PHISHPEDIA_DIR / "faster_rcnn.yaml"
DETECTOR_WEIGHTS = PHISHPEDIA_DIR / "rcnn_bet365.pth"
EMBEDDER_WEIGHTS = PHISHPEDIA_DIR / "resnetv2_rgb_new.pth.tar"

LOGO_CLASS = 1             # Phishpedia keeps pred_classes == 1 as logos
DETECT_THRESHOLD = 0.05    # Phishpedia's default score threshold
MIN_BOX_PX = 8             # smaller boxes are noise
CROP_PAD = 0.10            # crop margin around a box, as a share of its size
EMBED_SIZE = 128           # the matcher's input: pad to square (white), resize to 128


def to_rgb(img: Image.Image) -> Image.Image:
    """Flatten transparency. A logo that is white on transparent would vanish on white, so it goes on dark grey."""
    if img.mode in ("P", "LA", "L"):
        img = img.convert("RGBA")
    if img.mode != "RGBA":
        return img.convert("RGB")
    alpha = np.asarray(img)[..., 3] > 0
    rgb = np.asarray(img)[..., :3]
    bright = rgb[alpha].mean() > 230 if alpha.any() else False
    bg = Image.new("RGB", img.size, (40, 40, 40) if bright else (255, 255, 255))
    bg.paste(img, mask=img.split()[3])
    return bg


def _load_weights(path: Path) -> dict:
    import torch
    weights = torch.load(path, map_location="cpu", weights_only=True)
    return weights["model"] if "model" in weights else weights


class Detector:
    """Phishpedia's Faster R-CNN; returns logo boxes (x0, y0, x1, y1) in original pixels and their scores."""

    def __init__(self, device: str):
        import torch
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        self.torch = torch
        cfg = get_cfg()
        cfg.merge_from_file(str(DETECTOR_CFG))
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECT_THRESHOLD
        cfg.MODEL.DEVICE = device
        self.cfg, self.device = cfg, device
        self.model = build_model(cfg).eval()
        result = self.model.load_state_dict(_load_weights(DETECTOR_WEIGHTS), strict=False)
        # newer detectron2 no longer stores pixel mean/std and anchor sizes as weights; nothing learned may be missing
        assert not result.missing_keys, result.missing_keys

    def __call__(self, img: Image.Image):
        torch = self.torch
        bgr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()   # detectron2 models take BGR
        h, w = bgr.shape[:2]
        scale = self.cfg.INPUT.MIN_SIZE_TEST / min(h, w)
        if max(h, w) * scale > self.cfg.INPUT.MAX_SIZE_TEST:
            scale = self.cfg.INPUT.MAX_SIZE_TEST / max(h, w)
        # moved to the GPU before resizing, so the resize does not load every CPU core on the shared VM
        t = torch.as_tensor(bgr.transpose(2, 0, 1)).to(self.device).float()[None]
        t = torch.nn.functional.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False)[0]
        with torch.no_grad():
            inst = self.model([{"image": t, "height": h, "width": w}])[0]["instances"].to("cpu")
        keep = inst.pred_classes == LOGO_CLASS
        return inst.pred_boxes.tensor[keep].numpy(), inst.scores[keep].numpy()


class Embedder:
    """Phishpedia's siamese logo embedder (2048-d), L2-normalised, so a dot product is the cosine similarity."""

    def __init__(self, device: str):
        import torch
        self.torch, self.device = torch, device
        weights = {k.split("module.")[-1]: v for k, v in _load_weights(EMBEDDER_WEIGHTS).items()}
        self.model = KNOWN_MODELS["BiT-M-R50x1"](head_size=weights["head.conv.weight"].shape[0], zero_head=True)
        self.model.load_state_dict(weights)
        self.model.to(device).eval()

    @staticmethod
    def _prepare(img: Image.Image):
        img = img.convert("RGB")
        side = max(img.size)
        dx, dy = (side - img.size[0]) // 2, (side - img.size[1]) // 2
        img = ImageOps.expand(img, (dx, dy, dx, dy), fill=(255, 255, 255)).resize((EMBED_SIZE, EMBED_SIZE))
        return (np.asarray(img, np.float32) / 255 - 0.5) / 0.5

    def __call__(self, images: list, batch_size: int = 64) -> np.ndarray:
        torch = self.torch
        out = []
        for i in range(0, len(images), batch_size):
            x = torch.as_tensor(np.stack([self._prepare(im) for im in images[i:i + batch_size]])).permute(0, 3, 1, 2)
            with torch.no_grad():
                feats = self.model.features(x.to(self.device))
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)


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


class LogoMatcher:
    """Loads the detector and the matcher once; then scores any page against any client's logos."""

    def __init__(self):
        import torch

        self.device = "cuda" if torch.cuda.is_available() and DEVICE in ("auto", "cuda") else "cpu"
        self.detector, self.embedder = Detector(self.device), Embedder(self.device)

    def similarity(self, img, client_logos):
        """img = the page screenshot (PIL), client_logos = the client's logo images (PIL).
        Returns (logo_similarity, number of boxes found, number of client logos used)."""
        img = img.convert("RGB")
        gallery = self.embedder([to_rgb(l) for l in client_logos])
        boxes, _ = clean_boxes(*self.detector(img), img.width, img.height)
        if len(boxes) == 0 or len(gallery) == 0:
            return float("nan"), int(len(boxes)), int(len(gallery))
        sims = self.embedder([crop(img, b) for b in boxes]) @ gallery.T
        return float(sims.max()), int(len(boxes)), int(len(gallery))
