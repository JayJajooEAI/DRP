"""Step 5 of a request: build the model input text. One text, with separately marked fields, exactly as in the batch pipeline
(code/06_build_prompts.py):  [CLIENT] [TITLE] [HEADINGS] [BRAND_HINTS] [FORMS] [SCRIPT_URLS] [LINKS] [VISIBLE_TEXT] [OCR_TEXT] [SIGNALS]
Both models (branded / data capture) read this same text; each model shortens it to its own length limit (see classify.py)."""

import numpy as np

SEP = " | "
FIELDS = ["client", "title", "headings", "brand_hints", "forms", "script_urls", "links", "visible_text", "ocr_text", "signals"]


def num(v) -> str:
    return "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2f}"


def signals(logo_sim, p_url, forwarded) -> str:
    """The two numbers from the earlier steps: logo similarity and the suspicious-URL probability (or 'forwarded' for shortener links)."""
    return f"logo_similarity={num(logo_sim)} p_suspicious_url={'forwarded' if forwarded else num(p_url)}"


def build(client_name, html, ocr_lines, logo_sim, p_url, forwarded) -> dict:
    """html = the parsed fields from html_parse.parse(); ocr_lines = the OCR lines. Returns the fields (all kept whole)."""
    forms = list(html["forms"]) + list(html["inputs"]) + list(html["script_inputs"])
    return {
        "client": client_name,
        "title": html["title"],
        "headings": SEP.join(html["headings"]),
        "brand_hints": SEP.join(html["brand_hints"]),
        "forms": SEP.join(forms),
        "script_urls": SEP.join(html["script_urls"]),
        "links": SEP.join(html["links"]),
        "visible_text": html["visible_text"],
        "ocr_text": SEP.join(l["text"] for l in ocr_lines),
        "signals": signals(logo_sim, p_url, forwarded),
    }
