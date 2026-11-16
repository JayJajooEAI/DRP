"""The web-phish detector: a link and a client id in, a verdict out.

For a request the service
  1. reads the client (name, own domains, logos) and the stored page of the link (HTML + screenshot) from RDS,
  2. parses the HTML, reads the screenshot text (OCR), finds the logos on the screenshot and compares them with the client's logos,
  3. scores the URL with the suspicious-URL model,
  4. builds one prompt text from all of that and asks two fine-tuned models: check 2 "is the page branded as the client?" and
     check 3 "can the visitor enter credentials / payment data (3.1), or is there a login link (3.2), or neither (none)?",
  5. adds check 4 "is the page NOT on the client's own domain?" and combines them like the benchmark's routing rule:
        web phish = branded  AND  data-capture case is 3.1 or 3.2  AND  (not on the client's domain OR a form posts to another site).
"""

import threading
import time
from urllib.parse import urljoin, urlsplit

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import io

from WebPhish import prompt, url_features
from WebPhish.classify import Checker
from WebPhish.clients import RdsClients
from WebPhish.html_parse import EMPTY, _parse_doc, parse
from WebPhish.logo import LogoMatcher
from WebPhish.ocr import Ocr
from WebPhish.settings import BRANDED_THRESHOLD, MAX_HTML_BYTES, URL_MODEL


class UnknownClient(Exception):
    pass


class NoCapture(Exception):
    """No stored page for this link (and client)."""


class BadInput(Exception):
    pass


def is_client_domain(url: str, client: dict) -> bool:
    """True if the URL is on a domain the client owns. Hosts under the client's user-content domains (e.g. x.sharepoint.com) are not."""
    ext = url_features._extract(url)
    host = ext.fqdn.lower()
    if any(host == d or host.endswith("." + d) for d in client["user_content"]):
        return False
    return ext.top_domain_under_public_suffix in client["allowlist"] or ext.suffix.split(".")[-1] in client["owned_tlds"]


def external_form_target(raw: bytes, final_url: str) -> bool:
    """True if a form on the page posts to another registered domain than the page's own."""
    try:
        doc = _parse_doc(raw[:MAX_HTML_BYTES])
        page = url_features.registered_domain(final_url)
        for form in doc.iter("form"):
            action = (form.get("action") or "").strip()
            if action:
                target = urljoin(final_url, action)
                if urlsplit(target).scheme in ("http", "https") and url_features.registered_domain(target) != page:
                    return True
    except Exception:
        pass
    return False


class Engine:
    def __init__(self):
        # Loaded once at startup: OCR, logo detector + matcher, the URL model and the two fine-tuned models (all local)
        self.clients = RdsClients()
        self.ocr = Ocr()
        self.logos = LogoMatcher()
        self.url_model = joblib.load(URL_MODEL)
        self.check2, self.check3 = Checker(2), Checker(3)
        self.lock = threading.Lock()   # the GPU models handle one request at a time

    def info(self):
        return dict(url_model=self.url_model["variant"], device=self.check2.device,
                    check2_epoch=self.check2.cfg["epoch"], check3_epoch=self.check3.cfg["epoch"],
                    max_len=self.check2.cfg["max_len"])

    def url_score(self, url: str):
        """(p_suspicious_url, forwarded). Shortener / redirector links get no score (NaN) and forwarded = True."""
        if url_features.is_forwarded(url):
            return float("nan"), True
        m = self.url_model
        X = url_features.feature_frame([url], m["tld_risk"], m["vocab"])
        return float(m["model"].predict_proba(X[m["feature_columns"]])[:, 1][0]), False

    def predict(self, client_id: str, url: str) -> dict:
        """Check one link. A link can be stored more than once (several captures): every capture is run, and the answer lists them all."""
        if not url:
            raise BadInput("give a url")
        client = self.clients.get(client_id)
        if client is None:
            raise UnknownClient(client_id)
        caps = self.clients.captures(client_id, url)
        if not caps:
            raise NoCapture(url)
        aliases, owned = self.clients.brand_context()
        results = [self._run(client, cap, aliases, owned) for cap in caps]
        return dict(client=client_id, link=url, records=len(results),
                    web_phish_records=sum(r["prediction"] == "web_phish" for r in results), results=results)

    def _run(self, client: dict, cap: dict, aliases, owned) -> dict:
        """The full pipeline for one stored capture."""
        t = {"start": time.time()}
        t["rds"] = time.time()

        try:
            shot = Image.open(io.BytesIO(cap["screenshot"])).convert("RGB")
        except Exception:
            raise BadInput(f"the stored screenshot of content_id {cap['content_id']} is not a readable image")

        with self.lock:
            # Step 1: the HTML fields
            try:
                html = parse(cap["html"][:MAX_HTML_BYTES])
            except Exception:   # unparseable HTML: empty fields, like the batch pipeline
                html = EMPTY
            t["html"] = time.time()

            # Step 2: the screenshot text
            ocr_lines = self.ocr.read(shot)
            t["ocr"] = time.time()

            # Step 3: logos on the screenshot against the client's logos
            logo_sim, n_boxes, n_logos = self.logos.similarity(shot, client["logos"])
            t["logo"] = time.time()

            # Step 4: the suspicious-URL model (its brand-in-host feature uses all clients' names and domains from RDS)
            url_features.configure(aliases, owned)
            p_url, forwarded = self.url_score(cap["url"])
            t["url"] = time.time()

            # Steps 5 and 6: one prompt, two models
            fields = prompt.build(client["name"], html, ocr_lines, logo_sim, p_url, forwarded)
            c2, c3 = self.check2.predict(fields), self.check3.predict(fields)
            t["models"] = time.time()

        branded = c2["probs"]["True"] >= BRANDED_THRESHOLD
        case = c3["label"]
        on_client_domain = is_client_domain(cap["final_url"], client)
        ext_form = external_form_target(cap["html"], cap["final_url"])
        web_phish = branded and case != "none" and (not on_client_domain or ext_form)

        def ms(a, b):
            return round((t[b] - t[a]) * 1000)
        return dict(
            prediction="web_phish" if web_phish else "not_web_phish",
            checks=dict(
                branded=dict(value=bool(branded), p_branded=c2["probs"]["True"]),
                data_capture=dict(case=case, probs=c3["probs"]),
                not_client_domain=dict(value=not on_client_domain, final_url=cap["final_url"], form_posts_elsewhere=ext_form),
            ),
            signals=dict(logo_similarity=None if np.isnan(logo_sim) else round(logo_sim, 4), logo_boxes=n_boxes,
                         client_logos=n_logos, p_suspicious_url=None if np.isnan(p_url) else round(p_url, 4), url_forwarded=forwarded),
            capture=dict(content_id=cap["content_id"], url=cap["url"]),
            tokens=dict(check2=c2["n_tokens"], check3=c3["n_tokens"]),
            timings_ms=dict(html=ms("rds", "html"), ocr=ms("html", "ocr"), logo=ms("ocr", "logo"),
                            url=ms("logo", "url"), models=ms("url", "models"), total=ms("start", "models")),
        )
