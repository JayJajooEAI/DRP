"""The suspicious-domain detector: a domain and a client id in, a class out.

    1  ownership          owned list / WHOIS registrant is the client / listed partner          -> none
    2  typosquat rules    lookalike characters, other ending, plural, misspelling, brand + word  -> relevant
    3  string match       clear rules on names (brand parts, products, executives)
    4  string model       LightGBM on the names the rules call unclear
    5  legitimacy         a named other owner with a Tranco rank, or its own business name and contact page -> none
    6  parked check       parking nameservers or parking page -> domains_parked, else suspicious_domain_registration
    7  alert model        LightGBM on 'brand + extra word' alerts; below the threshold -> none

Same code and models as development/SuspiciousDomain. Client profile and domain facts are read from RDS on every request.
"""

import json
import re
import time

from SuspiciousDomain import alert_model, legitimacy, ownership, parked, string_match, string_model, typosquat
from SuspiciousDomain.clients import RdsClients
from SuspiciousDomain.settings import ALERT_MODEL, STRING_MODEL, THRESHOLDS


class UnknownClient(Exception):
    pass


class NoLookup(Exception):
    """RDS has no stored lookup (WHOIS, DNS, HTTP) for this domain."""


class BadInput(Exception):
    pass


def clean_domain(text: str) -> str:
    """'https://Harborbank-Login.com/x' -> 'harborbank-login.com'"""
    d = re.sub(r"^[a-z][a-z0-9+.-]*://", "", (text or "").strip().lower(), flags=re.I)
    return re.split(r"[/?#]", d, maxsplit=1)[0].rstrip(".")


class Engine:
    def __init__(self):
        self.clients = RdsClients()
        t = json.loads(THRESHOLDS.read_text())
        assert t["features"] == string_model.FEATURES, "the saved model was trained on a different feature list"
        self.string_model = string_model.Model.load(STRING_MODEL, t["string_model_threshold"])
        self.alert_model = string_model.Model.load(ALERT_MODEL, t["alert_model_threshold"])

    def info(self):
        return dict(string_model_threshold=self.string_model.threshold, alert_model_threshold=self.alert_model.threshold,
                    features=len(string_model.FEATURES), source="RDS")

    def detect(self, x):
        """-> (class, decided_by, trace). The same steps and trace text as the development pipeline."""
        own = ownership.check(x["domain"], x["profile"], x["rdap"])
        if own:
            return "none", "step 1 ownership", [f"ownership {own[0]}: {own[1]}"]
        hit = typosquat.check(x["domain"], x["profile"])
        if hit:
            trace, flagged_by = [f"typosquat: {hit[0]} ({hit[1]})"], hit[0]
            decided = "step 2 typosquat rules"
        else:
            flagged_by = "string match"
            verdict, reason, words = string_match.check(x["domain"], x["profile"])
            if reason.startswith("executive full name"):
                flagged_by = "executive full name"
            trace = ["no typosquat rule fires", f"words: {' '.join(words)}", f"string match {verdict}: {reason}"]
            decided = "step 3 string match"
            if verdict == "unclear":
                p = self.string_model.score(string_model.features(x))
                verdict = "relevant" if p >= self.string_model.threshold else "not relevant"
                trace.append(f"model: p={p:.2f} (threshold {self.string_model.threshold}) -> {verdict}")
                decided = "step 4 string model"
            if verdict != "relevant":
                return "none", decided, trace
        legit = legitimacy.check(x, flagged_by)
        if legit:
            return "none", "step 5 legitimacy", trace + [f"legitimate {legit[0]}: {legit[1]}"]
        park = parked.check(x["nameservers"], x["http"])
        cls, trace = (("domains_parked", trace + [f"parked: {park}"]) if park
                      else ("suspicious_domain_registration", trace + ["not parked"]))
        decided = "step 6 parked check"
        if flagged_by in alert_model.SENT:
            keep, note = alert_model.check(self.alert_model, string_model.features(x))
            trace.append(f"alert model: {note}")
            decided = "step 7 alert model"
            if not keep:
                return "none", decided, trace
        return cls, decided, trace

    def predict(self, client_id: str, domain: str) -> dict:
        t0 = time.time()
        domain = clean_domain(domain)
        if not domain or "." not in domain:
            raise BadInput("give a domain such as harborbank-login.com")
        profile = self.clients.profile(client_id)
        if profile is None:
            raise UnknownClient(client_id)
        facts = self.clients.domain(domain)
        if facts is None:
            raise NoLookup(domain)
        t1 = time.time()
        x = {"domain": facts["domain"], "profile": profile, "rdap": facts["rdap"], "nameservers": facts["nameservers"],
             "http": facts["http"], "page": facts["page"], "tranco_rank": facts["tranco_rank"]}
        cls, decided_by, trace = self.detect(x)
        t2 = time.time()
        return dict(
            client=client_id, domain=facts["domain"], prediction=cls, decided_by=decided_by, trace=trace,
            facts=dict(registrant_org=facts["rdap"]["registrant_org"], nameservers=facts["nameservers"],
                       http_status=facts["http"]["status"], parked_page=facts["http"]["body_is_parking_template"],
                       page_answered=facts["page"] is not None, tranco_rank=facts["tranco_rank"]),
            timings_ms=dict(rds=round((t1 - t0) * 1000), detect=round((t2 - t1) * 1000), total=round((t2 - t0) * 1000)),
        )
