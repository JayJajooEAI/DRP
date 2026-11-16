"""Step 2 of a request: read the text on the page screenshot with docTR (Mindee, Apache-2.0). Same engine and settings as the batch
pipeline (code/02_ocr_read_screenshots.py): db_resnet50 text detection + parseq text recognition."""

import numpy as np

from WebPhish.settings import DEVICE

DET_ARCH = "db_resnet50"   # text detection
RECO_ARCH = "parseq"       # text recognition


class Ocr:
    def __init__(self):
        import torch
        from doctr.models import ocr_predictor

        torch.set_num_threads(2)
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() and DEVICE in ("auto", "cuda") else "cpu"
        self.model = ocr_predictor(det_arch=DET_ARCH, reco_arch=RECO_ARCH, pretrained=True).to(self.device).eval()

    def read(self, img):
        """img = the screenshot (PIL RGB). Returns the text lines, top to bottom: [{text, score, box}]."""
        arr = np.asarray(img.convert("RGB"))
        h, w = arr.shape[:2]
        with self.torch.no_grad():
            page = self.model([arr]).pages[0]
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
