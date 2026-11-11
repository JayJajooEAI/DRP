"""Step 1 (read): docTR OCR on one screenshot -> text lines with position, height and confidence (same output format as the batch pipeline)."""

import torch
from doctr.models import ocr_predictor

from pipeline.settings import DEVICE, SHOT_SIZE


class OCR:
    def __init__(self):
        # Pick the device: GPU if available (unless DEVICE=cpu), else CPU
        use_cuda = torch.cuda.is_available() and DEVICE in ("auto", "cuda")
        self.device = "cuda" if use_cuda else "cpu"

        # db_resnet50 finds the text boxes, crnn_vgg16_bn reads the characters
        m = ocr_predictor("db_resnet50", "crnn_vgg16_bn", pretrained=True)
        self.model = (m.cuda() if use_cuda else m).eval()

    def read(self, rgb):
        """rgb = 900x900 uint8 numpy array (H, W, 3). Returns the lines sorted top-to-bottom, then left-to-right."""
        page = self.model([rgb]).pages[0]
        lines = []
        for b in page.blocks:
            for l in b.lines:
                # Corners are fractions of the image (0..1); multiply by the image size to get pixels
                (x0, y0), (x1, y1) = l.geometry
                lines.append(
                    dict(
                        x=int(x0 * SHOT_SIZE),
                        y=int(y0 * SHOT_SIZE),
                        x1=int(x1 * SHOT_SIZE),
                        h=int((y1 - y0) * SHOT_SIZE),
                        c=round(sum(w.confidence for w in l.words) / len(l.words), 3),
                        t=" ".join(w.value for w in l.words),
                    )
                )
        return sorted(lines, key=lambda a: (a["y"], a["x"]))
