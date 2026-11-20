"""Step 1: quick ownership check (PS doc, "Leverage or confusion"; plan Step 2). Lists and records only, no model.

A domain that belongs to the client or a known legitimate third party is never an incident:
    1a  the domain is on the client's owned list (main, regional, defensive domains)
    1b  the WHOIS registrant is the client's company (names compared after normalising, see norm_org)
    1c  the domain is on the client's partner / reseller / franchisee list

What does NOT count as proof (doc): a hidden registrant, a redirect to the client, being live or not live.
"""
import re
import unicodedata

import tldextract

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
REDACTED = re.compile(r"redacted|privacy|private|withheld|not disclosed|data protected|proxy|whoisguard", re.I)
LEGAL_FORMS = {"na", "inc", "llc", "ltd", "limited", "gmbh", "ag", "sa", "sl", "srl", "bv", "plc", "corp",
               "corporation", "co", "company", "kk", "sas", "sarl", "spa", "pty", "lp", "llp", "de", "cv"}


def norm_org(name):
    """'Harbor Bank, N.A.' -> 'harbor bank'; '&' reads as 'and'; legal forms dropped."""
    s = unicodedata.normalize("NFKD", (name or "").lower().replace("&", " and "))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))          # Móvil -> movil
    s = re.sub(r"\b([a-z])\.(?=[a-z]\.)", r"\1", s)           # n.a. -> na.
    words = re.sub(r"[^a-z0-9]+", " ", s).split()
    while words and words[-1] in LEGAL_FORMS:
        words.pop()
    return " ".join(words)


def registrable(domain):
    ex = _EXTRACT(domain.lower())
    return f"{ex.domain}.{ex.suffix}"


def check(domain, profile, rdap):
    """-> (sub-check, reason) when the domain is the client's or a listed partner's, else None."""
    d = domain.lower()
    forms = {d, registrable(d)}
    od = profile["owned_domains"]
    owned = {x.lower() for x in od["main"] + od["regional"] + od["defensive_listed"]}
    if forms & owned:
        return "1a owned list", "on the client's owned-domain list"
    org = (rdap or {}).get("registrant_org") or ""
    if org and not REDACTED.search(org) and norm_org(org) == norm_org(profile["registrant_org"]):
        return "1b registrant", f"WHOIS registrant is the client ({org})"
    partners = {t["domain"].lower(): t for t in profile["known_third_parties"]}
    hit = next((partners[f] for f in forms if f in partners), None)
    if hit:
        return "1c partner list", f"listed {hit['kind']}: {hit['org']}"
    return None
