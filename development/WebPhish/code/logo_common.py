"""Shared helpers for steps 03 to 05: paths, models and helpers for the logo similarity step.

Detector: Phishpedia's Faster R-CNN layout detector (detectron2, trained on phishing-page screenshots; class 1 is
"logo"). Matcher: Phishpedia's siamese ResNetV2 (BiT-M-R50x1) logo embedder, used for both the customer logo gallery
and the screenshot crops, so their embeddings are comparable. Phishpedia: github.com/lindsey98/Phishpedia (CC0).

Runs in the project .venv (detectron2 + torchvision); see setup_env.sh. Weights live in data/models/phishpedia/ and
are loaded with weights_only=True (plain tensors, no pickled code).
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "data" / "benchmark" / "items.csv"
ASSETS = ROOT / "data" / "customer_assets"
OUT = ROOT / "outputs" / "03_logo_similarity"
GALLERY = OUT / "logo_gallery.npz"
MODELS = ROOT / "data" / "models" / "phishpedia"

DETECTOR_ID = "phishpedia/rcnn_bet365.pth (Faster R-CNN R50-FPN, detectron2)"
EMBEDDER_ID = "phishpedia/resnetv2_rgb_new.pth.tar (siamese BiT-M-R50x1)"
DETECTOR_CFG = MODELS / "faster_rcnn.yaml"
DETECTOR_WEIGHTS = MODELS / "rcnn_bet365.pth"
EMBEDDER_WEIGHTS = MODELS / "resnetv2_rgb_new.pth.tar"

LOGO_CLASS = 1             # Phishpedia logo_recog.py keeps pred_classes == 1 as logos
DETECT_THRESHOLD = 0.05    # Phishpedia's default score threshold
MIN_BOX_PX = 8             # smaller boxes are noise
CROP_PAD = 0.10            # crop margin around a box, as a share of its size (the setting tested on the 50 samples)
EMBED_SIZE = 128           # Phishpedia's matcher input: pad to square (white), resize to 128


def load_profiles() -> dict:
    return {p.parent.name: json.loads(p.read_text()) for p in sorted(ASSETS.glob("*/profile.json"))}


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
        from phishpedia_bit import KNOWN_MODELS
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
