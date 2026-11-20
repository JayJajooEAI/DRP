"""Offline stand-ins for the data sources a detector would call in production. LOCAL TESTING ONLY.

Each function answers from data/ (the synthetic dataset) and returns only what the real source would return:
no case, label or expected path ever reaches the caller. Nothing here touches the network.

    get_new_domains(day)    the day's newly registered domains (WhoisDS-style feed)
    get_rdap(domain)        registrant, registrar, dates (RDAP / WHOIS)
    get_nameservers(domain) NS records
    get_http(domain)        status, redirect, and whether the response body is a known parking template
    render(domain)          page facts from a sandboxed headless browser (not the HTML)
    get_cert(domain)        TLS certificate issuer and dates
    is_known_phish(domain)  phishing-feed lookup
    tranco_rank(domain)     rank in the Tranco top-sites list, or None
"""
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "data"
CALLS = {}                       # source -> number of calls, for the cost report


def _count(name):
    CALLS[name] = CALLS.get(name, 0) + 1


@lru_cache(None)
def _intake():
    p = WORLD / "intake.json"
    if not p.exists():
        raise SystemExit("data/intake.json missing: run python code/01_build_intake.py first")
    return {r["domain"]: r for r in json.loads(p.read_text())["domains"]}


@lru_cache(None)
def profiles():
    return json.loads((WORLD / "customers.json").read_text())["customers"]


@lru_cache(None)
def _tranco():
    return json.loads((WORLD / "tranco.json").read_text())


def get_new_domains(day=None):
    _count("feed")
    return list(_intake())


def get_rdap(domain):
    _count("rdap")
    return dict(_intake()[domain]["rdap"])


def get_nameservers(domain):
    _count("dns")
    return list(_intake()[domain]["dns"]["nameservers"])


def get_http(domain):
    _count("http")
    r = _intake()[domain]
    return {**r["http"], "body_is_parking_template": r["page"]["kind"] in ("parked", "parked_brand_ads")}


# what a sandboxed browser would report for each kind of page
_TEXT = {"blank": 0, "hosting_default": 120, "coming_soon": 60, "generic_landing": 400, "captcha": 40, "login": 150,
         "parked": 300, "parked_brand_ads": 300, "business": 1500, "personal": 1200, "franchisee": 1400,
         "fan_review": 1800, "client_site": 2000, "clone_full": 2000, "clone_partial": 150, "clone_logo": 150}


def render(domain):
    """Page facts, or None when nothing answers. Costs a headless-browser load in production."""
    _count("render")
    r = _intake()[domain]
    kind = r["page"]["kind"]
    if r["http"]["status"] is None or kind == "none":
        return None
    return {"status": r["http"]["status"], "final_url": r["http"]["final_url"],
            "has_password_input": kind == "login" or bool(r["page"]["has_login_form"]),
            "captcha_widget": kind == "captcha",
            "visible_text_chars": _TEXT[kind],
            "parking_template": kind in ("parked", "parked_brand_ads"),
            **{k: r["page"].get(k) for k in ("links_to_client_site", "shows_own_business_name", "has_contact_page",
                                             "has_privacy_page", "shows_client_logo")}}


def get_cert(domain):
    _count("cert")
    return _intake()[domain]["cert"]


def is_known_phish(domain):
    _count("phish_feed")
    return bool(_intake()[domain]["known_phish"])


def tranco_rank(domain):
    _count("tranco")
    return _tranco().get(domain)
