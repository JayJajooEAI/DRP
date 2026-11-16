"""Lexical URL features for the suspicious-URL model (same code as the batch pipeline's pipeline/url_features.py, trimmed to what
the saved model needs). The saved model file pickles `TldRisk` from this module path (`pipeline.url_features`), so the class must stay here.

Two features need the clients' brand names and owned domains (`brand_in_host`). The service reads them from RDS for every request
and hands them in with `configure(...)`; nothing is cached between requests.
"""
import ipaddress
import math
import re
from collections import Counter
from urllib.parse import parse_qsl, urlsplit

import numpy as np
import pandas as pd

import tldextract

# Offline suffix list (the snapshot bundled with tldextract), private suffixes on, so tenants of shared platforms
# (x.azurewebsites.net, x.000webhostapp.com) are their own registered domains.
_extract = tldextract.TLDExtract(include_psl_private_domains=True, suffix_list_urls=())


def registered_domain(url_or_host: str) -> str:
    return _extract(url_or_host).top_domain_under_public_suffix


# The brand tokens and owned domains of all the clients; set per request by configure()
_BRANDS: tuple = ()
_OWNED: frozenset = frozenset()


def configure(aliases, owned_domains) -> None:
    """aliases = every client's brand names and aliases; owned_domains = every client's own domains (both from RDS)."""
    global _BRANDS, _OWNED
    toks = set()
    for a in aliases:
        t = re.sub(r"[^a-z0-9]", "", a.lower())
        if len(t) >= 4 and t not in ("office", "microsoft365"):
            toks.add(t)
    _BRANDS, _OWNED = tuple(sorted(toks)), frozenset(owned_domains)

KEYWORDS = ("login", "logon", "signin", "sign-in", "verify", "verification", "secure", "account", "update", "confirm",
            "wallet", "bank", "password", "auth", "billing", "payment", "invoice", "recover", "unlock", "suspend",
            "support", "webscr")
HOST_WORDS = ("login", "secure", "verify", "account", "update", "support")
SHARED_HOSTS = (
    "firebaseapp.com", "web.app", "weeblysite.com", "weebly.com", "wixsite.com", "000webhostapp.com",
    "github.io", "blogspot.com", "repl.co", "replit.app", "glitch.me", "netlify.app", "vercel.app",
    "herokuapp.com", "pages.dev", "workers.dev", "r2.dev", "azurewebsites.net", "web.core.windows.net",
    "appspot.com", "square.site", "godaddysites.com", "webflow.io", "gitbook.io", "ngrok.io",
    "ipfs.io", "cloudflare-ipfs.com", "dweb.link", "nftstorage.link", "w3s.link", "fleek.co",
)
# Link shorteners and redirect / tracking wrappers: the destination can't be judged from the URL
FORWARD_HOSTS = (
    "bit.ly", "bitly.com", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "rebrand.ly", "cutt.ly",
    "shorturl.at", "rb.gy", "tiny.cc", "bit.do", "t.ly", "s.id", "v.gd", "shorte.st", "adf.ly", "lnkd.in", "db.tt",
    "we.tl", "1drv.ms", "aka.ms", "fb.me", "amzn.to", "youtu.be", "qrco.de", "urlz.fr", "x.co",
    "safelinks.protection.outlook.com", "urldefense.com", "urldefense.proofpoint.com", "l.facebook.com",
    "lm.facebook.com", "click.mailchimp.com", "list-manage.com", "sendgrid.net", "mandrillapp.com",
)
REDIRECT_PARAMS = {"url", "next", "redirect", "redirect_uri", "redirect_url", "return", "returnurl", "return_to",
                   "continue", "dest", "destination", "goto", "target", "u", "r"}
SPECIAL = "@~%=&?_."
VOWELS = set("aeiou")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.I)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_EXT = re.compile(r"\.([a-z0-9]{1,5})$")
COMMON_EXT = ("php", "html", "htm", "asp", "aspx", "jsp", "cgi", "js", "exe", "zip", "pdf", "doc", "docx", "xls")


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return bool(re.fullmatch(r"0x[0-9a-f]+|\d+", host))  # integer / hex hosts


def _scripts(s: str) -> set:
    """Rough script classes of the letters in s (latin, cyrillic, greek, other)."""
    out = set()
    for c in s:
        if not c.isalpha():
            continue
        o = ord(c)
        out.add("latin" if o < 0x250 else "greek" if 0x370 <= o < 0x400 else "cyrillic" if 0x400 <= o < 0x530
                else "other")
    return out


def brand_tokens() -> tuple:
    """Lower-case brand tokens of the clients, e.g. 'paypal', 'wellsfargo', 'bankofamerica'."""
    return _BRANDS


def owned_domains() -> frozenset:
    return _OWNED


def domain_only(url: str) -> str:
    """The URL reduced to its registered domain (or IP host): the input of the domain-only variant."""
    return registered_domain(url) or _parts(url)[2]


def strip_scheme(url: str) -> str:
    return _SCHEME.sub("", url.strip())


def _parts(url: str):
    """Scheme-free URL, its parts and host. A leading `www.` and a bare trailing `/` are dropped first: they say
    nothing about legitimacy, and PhiUSIIL's legitimate URLs make them shortcuts."""
    raw = re.sub(r"^www\.", "", strip_scheme(url), flags=re.I)
    if re.fullmatch(r"[^/?#]+/", raw):
        raw = raw[:-1]
    try:
        parts = urlsplit("//" + raw)
    except ValueError:  # unbalanced IPv6 bracket
        parts = urlsplit("//" + raw.replace("[", "").replace("]", ""))
    return raw, parts, (parts.hostname or "").rstrip(".")


def is_forwarded(url: str) -> bool:
    host = _parts(url)[2].lower()
    return any(host == h or host.endswith("." + h) for h in FORWARD_HOSTS)


def url_features(url: str) -> dict:
    raw, parts, host = _parts(url)
    low = raw.lower()
    try:
        has_port = parts.port is not None
    except ValueError:
        has_port = True
    path, query, fragment = parts.path, parts.query, parts.fragment

    is_ip = _is_ip(host)
    labels = host.split(".") if host else []
    tld = "" if is_ip or not labels else labels[-1]
    tokens = [t for t in _TOKEN_SPLIT.split(low) if t]
    n = max(len(raw), 1)
    path_low = path.lower()
    host_letters = [c for c in host.lower() if c.isalpha()]
    consonant_runs = re.findall(r"[b-df-hj-np-tv-z]+", host.lower())
    params = parse_qsl(query, keep_blank_values=True)
    ext = _EXT.search(path_low)
    ext = ext.group(1) if ext else ""
    host_compact = re.sub(r"[^a-z0-9]", "", host.lower())
    # a brand name in the host is suspicious only off the brands' own domains (paypal.com is not, paypal-x.com is)
    brand_host = any(b in host_compact for b in brand_tokens()) and registered_domain(host) not in owned_domains()
    path_compact = re.sub(r"[^a-z0-9]", "", (path + "?" + query).lower())
    brands = brand_tokens()

    f = {
        # lengths
        "url_len": len(raw),
        "host_len": len(host),
        "domain_len": len(labels[-2]) if len(labels) >= 2 and not is_ip else len(host),
        "path_len": len(path),
        "query_len": len(query),
        "fragment_len": len(fragment),
        "path_depth": sum(1 for seg in path.split("/") if seg),
        "n_tokens": len(tokens),
        "longest_token": max((len(t) for t in tokens), default=0),
        # host
        "host_dots": host.count("."),
        "subdomain_levels": 0 if is_ip else max(len(labels) - 2, 0),
        "host_hyphens": host.count("-"),
        "host_digits": sum(c.isdigit() for c in host),
        "is_ip": int(is_ip),
        "has_port": int(has_port),
        "punycode": int("xn--" in host),
        "non_ascii": int(any(ord(c) > 127 for c in raw)),
        "mixed_scripts": int(len(_scripts(host)) > 1),
        "host_entropy": _entropy(host),
        "tld_len": len(tld),
        "consonant_ratio": (sum(c not in VOWELS for c in host_letters) / len(host_letters)) if host_letters else 0.0,
        "longest_consonant_run": max((len(r) for r in consonant_runs), default=0),
        # whole-URL characters
        "url_entropy": _entropy(raw),
        "digit_ratio": sum(c.isdigit() for c in raw) / n,
        "letter_ratio": sum(c.isalpha() for c in raw) / n,
        "n_hyphens": raw.count("-"),
        "n_slashes": raw.count("/"),
        **{f"n_{name}": raw.count(ch) for name, ch in zip(
            ["at", "tilde", "percent", "eq", "amp", "qmark", "underscore", "dot"], SPECIAL)},
        "n_pct_encoded": len(re.findall(r"%[0-9a-fA-F]{2}", raw)),
        "n_other_special": sum(1 for c in raw if not c.isalnum() and c not in SPECIAL + ":/-"),
        "double_slash_in_path": int("//" in path),
        # query
        "has_query": int(bool(query)),
        "n_params": len(params),
        "max_param_len": max((len(v) for _, v in params), default=0),
        "redirect_param": int(any(k.lower() in REDIRECT_PARAMS for k, _ in params)
                              or bool(re.search(r"=(https?%3a|https?:)", query, re.I))),
        # path
        "file_ext": ext if ext in COMMON_EXT else ("other" if ext else "none"),
        # hosting
        "shared_host": int(any(host == s or host.endswith("." + s) for s in SHARED_HOSTS)),
        "ipfs_path": int(path_low.startswith(("/ipfs/", "/ipns/"))),
        # brands (any of the 25 customers)
        "brand_in_host": int(brand_host),
        "brand_in_path_not_host": int(any(b in path_compact for b in brands) and not brand_host),
        "tld": tld,
    }
    for kw in HOST_WORDS:
        f["host_kw_" + kw] = int(kw in host.lower())
    for kw in KEYWORDS:
        f["path_kw_" + kw.replace("-", "_")] = int(kw in path_low + "?" + query.lower())
    f["host_kw_count"] = sum(kw in host.lower() for kw in KEYWORDS)
    f["path_kw_count"] = sum(kw in path_low for kw in KEYWORDS)
    return f


CATEGORICAL = ("tld", "file_ext")
# Features of the host alone. PhiUSIIL's legitimate URLs have no path or query, so anything computed from the path
# can only learn "has a path = phishing"; the host-only variant avoids that.
HOST_FEATURES = ["host_len", "domain_len", "host_dots", "subdomain_levels", "host_hyphens", "host_digits", "is_ip",
                 "has_port", "punycode", "mixed_scripts", "host_entropy", "tld_len", "consonant_ratio",
                 "longest_consonant_run", "shared_host", "brand_in_host", "tld", "tld_risk",
                 *[f"host_kw_{k}" for k in HOST_WORDS], "host_kw_count"]


class TldRisk:
    """Smoothed phishing rate per TLD, fitted on training URLs only (unseen TLDs get the prior)."""

    def __init__(self, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, tlds: pd.Series, y: np.ndarray):
        self.prior = float(np.mean(y))
        g = pd.DataFrame({"tld": tlds.values, "y": y}).groupby("tld")["y"].agg(["sum", "count"])
        self.table = ((g["sum"] + self.smoothing * self.prior) / (g["count"] + self.smoothing)).to_dict()
        return self

    def transform(self, tlds: pd.Series) -> np.ndarray:
        return tlds.map(self.table).fillna(self.prior).to_numpy()


def encode(raw: pd.DataFrame, tld_risk: TldRisk, vocab: dict) -> pd.DataFrame:
    """Model matrix from url_features() rows: adds TLD risk, fixes the categorical vocabularies."""
    X = raw.copy()
    X["tld_risk"] = tld_risk.transform(X["tld"])
    for col in CATEGORICAL:
        cats = [c for c in vocab[col] if c != "other"]
        X[col] = pd.Categorical(X[col].where(X[col].isin(cats), "other"), categories=[*cats, "other"])
    return X


def feature_frame(urls, tld_risk: TldRisk, vocab: dict) -> pd.DataFrame:
    return encode(pd.DataFrame([url_features(u) for u in urls]), tld_risk, vocab)
