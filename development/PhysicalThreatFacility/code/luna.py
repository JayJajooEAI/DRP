#!/usr/bin/env python3
"""
RUNG 3 - GPT-6 Luna decides the messages DistilBERT is unsure about (P(threat) 0.3-0.6).
Final prompt: v14 (prompt_v14.txt). The answer is strict JSON (response_schema.json):
    label 0/1 · subtype (one of 32 CTI subtypes, "other", or "none" when label is 0) · confidence · reason

Needs OPENAI_API_KEY in the environment, or in a .env file (the package root, or the file named by OPENAI_ENV_FILE). Never commit or share the key.

    from luna import Luna
    Luna().classify("Shots fired near 5th Ave, Seattle")
        -> {"label": 1, "subtype": "shots fired", "confidence": .., "reason": "..", "usage": {..}, "meta": {..}}

    python luna.py "some message"
Price (per 1M tokens): input $0.10, cached input $0.01, output $0.50. Without caching: about $0.30 per 1,000 messages.
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "gpt-6-luna"
MAX_CHARS = 4000                 # longer texts are cut before sending
SYSTEM = (HERE / "prompt_v14.txt").read_text(encoding="utf-8")
SCHEMA = json.load(open(HERE / "response_schema.json"))
PRICES = {"input": 0.10, "cached_input": 0.01, "output": 0.50}

def load_env(path=None):
    path = Path(path or os.environ.get("OPENAI_ENV_FILE", HERE.parent / ".env"))   # the key can also sit in a .env file (OPENAI_ENV_FILE names it)
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1); k, v = k.strip(), v.strip().strip('"').strip("'")
                if v and not os.environ.get(k): os.environ[k] = v
    if not os.environ.get("OPENAI_API_KEY"): sys.exit(f"OPENAI_API_KEY is not set (environment or {path})")

class Luna:
    def __init__(self):
        load_env()
        from openai import OpenAI
        self.client = OpenAI(max_retries=0); self.temperature = True

    def classify(self, text):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "TEXT:\n" + text[:MAX_CHARS]}]
        errors = []
        for attempt in range(8):
            try:
                kw = dict(model=MODEL, messages=msgs, response_format=SCHEMA)
                if self.temperature: kw["temperature"] = 0
                t0 = time.time(); r = self.client.chat.completions.create(**kw); latency = time.time() - t0
                out = json.loads(r.choices[0].message.content)
                label, sub = int(out["label"]), out["subtype"]
                if label == 0: sub = "none"
                elif sub == "none": sub = "other"
                u = r.usage.model_dump() if r.usage else {}
                tin, tout = u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                return {"label": label, "subtype": sub, "confidence": float(out["confidence"]), "reason": out["reason"],
                        "usage": u,                                            # every usage field the API returned
                        "cost_usd": (tin - cached) * PRICES["input"] / 1e6 + cached * PRICES["cached_input"] / 1e6 + tout * PRICES["output"] / 1e6,
                        "cost_usd_no_cache": tin * PRICES["input"] / 1e6 + tout * PRICES["output"] / 1e6,
                        "meta": {"request_id": r.id, "model_served": r.model, "service_tier": getattr(r, "service_tier", None),
                                 "system_fingerprint": getattr(r, "system_fingerprint", None), "created": r.created,
                                 "finish_reason": r.choices[0].finish_reason, "latency_s": round(latency, 3), "attempts": attempt + 1,
                                 "errors": errors, "temperature_0": self.temperature, "reasoning_effort": "default (not set)",
                                 "prompt": "v14", "max_chars": MAX_CHARS, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}}
            except Exception as e:
                msg = str(e); errors.append(msg[:200])
                if "temperature" in msg and self.temperature: self.temperature = False; continue   # some models only allow the default
                if attempt == 7 or "invalid_api_key" in msg or "model_not_found" in msg: raise
                time.sleep(min(60, 2 ** attempt))                                                  # rate limits / transient errors

if __name__ == "__main__":
    L = Luna()
    for t in sys.argv[1:]:
        r = L.classify(t); print(json.dumps({k: r[k] for k in ("label", "subtype", "confidence", "reason")}, ensure_ascii=False), "|", t[:80])
