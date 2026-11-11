"""The whole pipeline for ONE screenshot: OCR -> layout parser -> avatar matching -> signals -> rules -> ML model -> (Luna if still unsure)."""

import io
import threading
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from pipeline.avatar import AvatarMatcher
from pipeline.clients import RdsClients, RdsUnavailable
from pipeline.features import feats
from pipeline.layout_parser import parse
from pipeline.luna import Luna
from pipeline.ocr import OCR
from pipeline.rules import decide, rule_score
from pipeline.settings import MODEL_FILE, SHOT_SIZE
from pipeline.signals import signals


class BadImage(ValueError):
    """The upload could not be read as an image."""


class UnknownClient(KeyError):
    """The requested client_id is not in the registry."""


class Engine:
    def __init__(self):
        # Loaded once at startup: the frozen model + thresholds (local), OCR, CLIP, Luna client. Client data is fetched from RDS per request (cached).
        self.clients = RdsClients()
        F = joblib.load(MODEL_FILE)
        self.model, self.art = F["model"], F["art"]
        self.ocr = OCR()
        self.avatar = AvatarMatcher()
        self.luna = Luna()

        # The neural models are shared by all requests, so use them one request at a time
        self.lock = threading.Lock()

    def info(self):
        """Facts about the loaded model (for the /health endpoint)."""
        a = self.art
        try:
            n_clients = len(self.clients.ids())
        except RdsUnavailable as e:
            n_clients = f"unavailable: {e}"
        return dict(
            clients=n_clients,
            device=self.ocr.device,
            rule_thresholds=dict(not_impersonation_at_or_below=a["rule_lo"], impersonation_at_or_above=a["rule_hi"]),
            ml_thresholds=dict(not_impersonation_at_or_below=a["ml_lo"], impersonation_at_or_above=a["ml_hi"]),
            features=len(a["cols"]),
            luna=dict(available=self.luna.available, reason=self.luna.reason),
        )

    def _load_image(self, data):
        """Bytes -> RGB PIL image of exactly 900x900 (the size the parser's positions are calibrated for)."""
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
        except Exception:
            raise BadImage("the file is not a readable image")
        im = im.convert("RGB")
        resized = im.size != (SHOT_SIZE, SHOT_SIZE)
        if resized:
            im = im.resize((SHOT_SIZE, SHOT_SIZE), Image.LANCZOS)
        return im, resized

    def predict(self, data, client_id, use_luna=True):
        """data = the screenshot file's bytes, client_id = which protected client to check against. Returns a JSON-ready dict."""
        client = self.clients.get(client_id)
        if client is None:
            raise UnknownClient(client_id)
        img, resized = self._load_image(data)
        t = {}
        t0 = time.time()

        with self.lock:
            # Step 1a: read the text (OCR)
            ocr_lines = self.ocr.read(np.asarray(img))
            t["ocr"] = time.time()

            # Step 1b: turn the text lines into fields (platform, name, handle, bio, link, stats, posts)
            parsed = parse(ocr_lines)
            t["parse"] = time.time()

            # Step 1c: avatar matching against this client's logo and executive photos
            av = self.avatar.match(img, parsed["platform"], client)
            t["avatar"] = time.time()

        # Step 1d: about 100 signals, as a one-row table
        sig = signals(client, ocr_lines, av, img)
        row = pd.DataFrame([sig])
        t["signals"] = time.time()

        # Step 2: rules -> score 0..1 -> decision (1 = impersonation, 0 = not, -1 = unsure)
        rscore = float(rule_score(row).iloc[0])
        rd = int(decide(np.array([rscore]), self.art["rule_hi"], self.art["rule_lo"])[0])

        # Step 3: the ML model, only used when the rules were unsure (the probability is computed anyway, it costs ~0.01 ms)
        p_ml = float(self.model.predict_proba(feats(row)[self.art["cols"]])[:, 1][0])
        md = 1 if p_ml >= self.art["ml_hi"] else 0 if p_ml <= self.art["ml_lo"] else -1
        t["model"] = time.time()

        # Which stage answers: rules, else the model, else Luna
        luna = None
        if rd >= 0:
            stage, pred = "rules", rd
        elif md >= 0:
            stage, pred = "ml", md
        else:
            # Step 4: Luna; if it is off, unavailable or fails, keep the ML model's best guess
            stage, pred = "ml_fallback", int(p_ml >= 0.5)
            if use_luna and self.luna.available:
                luna = self.luna.decide(client, sig, img)
                if luna["choice"] is not None:
                    stage, pred = "luna", int(luna["choice"] == "impersonation")
            else:
                luna = dict(choice=None, p_imp=None, error=self.luna.reason if not self.luna.available else "skipped by request")
            t["luna"] = time.time()

        # Timings in milliseconds for each step that ran
        prev, ms = t0, {}
        for k, v in t.items():
            ms[k] = round((v - prev) * 1000, 1)
            prev = v
        ms["total"] = round((time.time() - t0) * 1000, 1)

        return dict(
            prediction="impersonation" if pred == 1 else "not_impersonation",
            is_impersonation=bool(pred == 1),
            decided_by=stage,
            client=client_id,
            rule_score=round(rscore, 4),
            ml_probability=round(p_ml, 4),
            luna=luna,
            parsed=dict(
                platform=parsed["platform"],
                display_name=parsed["name"],
                handle=parsed["handle"],
                bio=parsed["bio"],
                link=parsed["link"],
                followers=parsed["followers"] if parsed["followers"] is not None else parsed["connections"],
                posts_seen=len(parsed["posts"]),
            ),
            checks=dict(
                avatar_matches_client_logo=bool(sig["av_logo_strong"]),
                avatar_matches_executive_photo=bool(sig["av_exec_strong"]),
                verified_badge=bool(sig["verified"]),
                link_on_page=("official domain" if sig["link_official"] else "other domain" if sig["link_other"] else "none"),
            ),
            image_resized_to_900=resized,
            timings_ms=ms,
        )
