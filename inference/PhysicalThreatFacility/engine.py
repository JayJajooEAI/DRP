"""The physical-threat detector: a message in, a verdict out. A three-rung ladder; each message stops at the first rung that decides it.

    RUNG 1  does the message name a location?   no -> not a threat
    RUNG 2  DistilBERT P(threat)                >= 0.6 threat | < 0.3 not a threat | 0.3 to 0.6 unsure
    RUNG 3  GPT-6 Luna (prompt v14) decides the unsure ones
Same code and thresholds as development/PhysicalThreatFacility. This incident type does not depend on a client.
"""

import threading
import time

from PhysicalThreatFacility.location import LocationModel
from PhysicalThreatFacility.settings import DEVICE, LOCATION_MODEL_DIR, LUNA_ENABLED, MAX_TEXT_CHARS, THREAT_MODEL_DIR
from PhysicalThreatFacility.threat import ThreatModel, decide


class BadInput(Exception):
    pass


class Engine:
    def __init__(self):
        # Loaded once at startup: both local models, and Rung 3's client when an API key is set
        device = "cpu" if DEVICE == "cpu" else None
        self.location = LocationModel(LOCATION_MODEL_DIR, device)
        self.threat = ThreatModel(THREAT_MODEL_DIR, device)
        self.luna, self.luna_reason = None, "turned off (LUNA_ENABLED=0)"
        if LUNA_ENABLED:
            try:
                from PhysicalThreatFacility.luna import Luna
                self.luna, self.luna_reason = Luna(), None
            except Exception as e:   # no key, or the openai package is missing: the other rungs still work
                self.luna_reason = f"{type(e).__name__}: {e}"[:200]
        self.lock = threading.Lock()   # the models handle one request at a time

    def info(self):
        return dict(device=self.location.dev, luna=dict(available=self.luna is not None, reason=self.luna_reason),
                    thresholds=dict(rung2_not_a_threat_below=0.3, rung2_threat_at_or_above=0.6, rung1_place_cutoff=0.4))

    def predict(self, text: str, use_luna: bool = True) -> dict:
        if not text or not text.strip():
            raise BadInput("give some text")
        if len(text) > MAX_TEXT_CHARS:
            raise BadInput(f"text is longer than {MAX_TEXT_CHARS} characters")
        t = {"start": time.time()}
        with self.lock:
            loc = self.location.predict([text])[0]                       # Rung 1
            t["rung1"] = time.time()
            p, g = None, None
            if not loc["has_location"]:
                decided_by, prediction = "rung1", 0
            else:
                p = self.threat.score([text])[0]                         # Rung 2
                t["rung2"] = time.time()
                d = decide(p)
                if d != "unsure":
                    decided_by, prediction = "rung2", int(d == "threat")
                elif use_luna and self.luna is not None:                 # Rung 3
                    g = self.luna.classify(text)
                    decided_by, prediction = "rung3", int(g["label"])
                    t["rung3"] = time.time()
                else:                                                    # Luna off or unavailable: Rung 2's best guess
                    decided_by, prediction = "rung2_fallback", int(p >= 0.5)

        last = max(t.values())
        return dict(
            prediction="threat" if prediction else "not_threat",
            decided_by=decided_by,
            location=dict(found=loc["has_location"], places=loc["places"], by=loc["by"] or None),
            p_threat=None if p is None else round(p, 4),
            luna=None if g is None else dict(label=g["label"], subtype=g["subtype"], confidence=g["confidence"], reason=g["reason"]),
            luna_skipped=None if (g is not None or decided_by != "rung2_fallback") else (self.luna_reason or "skipped by request"),
            timings_ms=dict(total=round((last - t["start"]) * 1000)),
        )
