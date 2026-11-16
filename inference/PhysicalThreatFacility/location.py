#!/usr/bin/env python3
"""
RUNG 1 - rule-based filter: "does the message name a location?"  No location -> not a physical threat (stop here).

The check is done by one model, MiniLMv2 fine-tuned (mMiniLMv2-L6-H384, multilingual) as a word tagger with one tag,
"place" (B / I / O). A message has a location when any word piece scores P(B) + P(I) >= 0.4. When the model finds
nothing, simple place rules are the fallback (street-address shapes, "X County", ", TX", Japanese/Chinese place
suffixes, and a gazetteer of countries, US states and cities of 150k+ people).

    from location import LocationModel
    m = LocationModel()                       # loads model/rung1_location, GPU if available
    m.predict(["Shots fired near 5th Ave and Pine St, Seattle"])  ->  [{"has_location": True, "places": [...], "by": "model"}]

    python location.py "some message" ...     # quick check from the shell
"""
import re, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model" / "rung1_location"          # the trained location model (stored in this folder)
TAU = 0.4                        # place cut-off, chosen on the val split
MAXLEN, BATCH = 384, 32
CHUNK_CHARS, MAX_CHARS = 1000, 3000

# ---------------- text preparation (same as training) ----------------
def clean(t):
    """drop links and @handles; keep hashtag words, split camel case: "#ChicagoScanner" -> "Chicago Scanner" """
    t = re.sub(r"https?://\S+|www\.\S+", " ", t or "")
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"#(\w+)", lambda m: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", m.group(1)), t)
    return re.sub(r"\s+", " ", t).strip()

def chunks(t):
    """first 3,000 characters, in pieces of up to 1,000 cut at a space"""
    t = t[:MAX_CHARS]; out, i = [], 0
    while i < len(t):
        j = min(len(t), i + CHUNK_CHARS)
        if j < len(t):
            k = t.rfind(" ", i, j); j = k if k > i else j
        out.append(t[i:j]); i = j
    return out

# ---------------- place-rule fallback (runs on the raw message) ----------------
SHAPE = re.compile(r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?[A-Z][\w.]*(?:\s+[A-Z][\w.]*)*\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Dr|Drive|Ln|Lane|Way|Hwy|Highway|Pkwy|Ct|Pl)\b"
                   r"|\b(?:block of|near|at)\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Blvd|Dr)\b|\b[A-Z][a-z]+ (?:St|Ave|Rd|Blvd) (?:&|and|at) [A-Z]"
                   r"|\b[A-Z][a-z]+ (?:County|Parish|Province|District|Township)\b"
                   r"|,\s?(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b")
CJK = re.compile(r"[㐀-鿿]{1,6}(?:都|道|府|県|市|区|町|村|郡)")
_PLACES = None
def _gazetteer():
    global _PLACES
    if _PLACES is None:
        import geonamescache
        gc = geonamescache.GeonamesCache()
        p = {c["name"] for c in gc.get_countries().values()} | {s["name"] for s in gc.get_us_states().values()}
        p |= {c["name"] for c in gc.get_cities().values() if c["population"] >= 150000 and len(c["name"]) >= 4}
        _PLACES = {n for n in p if n[:1].isupper()}           # case-sensitive: "Reading" the city, not "reading"
    return _PLACES
RX_CAPWORDS = re.compile(r"[^\W\d_][\w'.-]*", re.U)
def has_place_name(t):
    places = _gazetteer(); maxw = max(len(n.split()) for n in places); words = RX_CAPWORDS.findall(t)
    for i in range(len(words)):
        if not words[i][:1].isupper(): continue
        for n in range(1, maxw + 1):
            if " ".join(words[i:i + n]) in places: return True
    return False
def place_rule(t):
    m = SHAPE.search(t) or CJK.search(t)
    return m.group(0).strip() + " · address / place rule" if m else "place name · gazetteer rule" if has_place_name(t) else ""

# ---------------- the model ----------------
class LocationModel:
    def __init__(self, model_dir=MODEL_DIR, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir, dtype=torch.float32).to(self.dev).eval()

    @torch.no_grad()
    def _probs(self, texts):
        """place probability (1 - P(O)) per word piece, for each chunk; shortest first for fast batches"""
        enc = self.tok(texts, truncation=True, max_length=MAXLEN, return_offsets_mapping=True)
        order = sorted(range(len(texts)), key=lambda i: len(enc["input_ids"][i])); probs = [None] * len(texts)
        for k in range(0, len(order), BATCH):
            idx = order[k:k + BATCH]; n = max(len(enc["input_ids"][i]) for i in idx)
            ids = torch.full((len(idx), n), self.tok.pad_token_id); att = torch.zeros((len(idx), n), dtype=torch.long)
            for j, i in enumerate(idx):
                L = len(enc["input_ids"][i]); ids[j, :L] = torch.tensor(enc["input_ids"][i]); att[j, :L] = 1
            with torch.autocast(self.dev, dtype=torch.float16, enabled=self.dev == "cuda"):
                logits = self.model(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).logits.float()
            p = 1 - torch.softmax(logits, -1)[..., 0]
            for j, i in enumerate(idx): probs[i] = p[j, :len(enc["input_ids"][i])].tolist()
        return enc["offset_mapping"], probs

    @staticmethod
    def _spans(text, offs, prob):
        """join neighbouring word pieces at or above the cut-off into place strings"""
        out = []
        for (s, e), p in zip(offs, prob):
            if s == e or p < TAU: continue
            if out and s <= out[-1][1] + 1: out[-1][1] = e
            else: out.append([s, e])
        return [text[s:e].strip() for s, e in out if text[s:e].strip()]

    def predict(self, texts):
        per = [chunks(clean(t)) for t in texts]; flat = [c for p in per for c in p]
        offs, probs = self._probs(flat) if flat else ([], [])
        out, k = [], 0
        for t, p in zip(texts, per):
            places = []
            for c in p: places += self._spans(c, offs[k], probs[k]); k += 1
            places = list(dict.fromkeys(places)); by = "model" if places else ""
            if not places:
                r = place_rule(t)
                if r: places, by = [r], "place rule"
            out.append({"has_location": bool(places), "places": places[:6], "by": by})
        return out

if __name__ == "__main__":
    m = LocationModel()
    for t, r in zip(sys.argv[1:], m.predict(sys.argv[1:])): print(r, "|", t[:80])
