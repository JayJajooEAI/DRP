"""Step 3: string match, clear rules only (PS doc, "Meaningful string match"; plan Steps 4a and 4b).

Runs on domains the typosquat rules (Step 2) did not flag. It finds the client's names in the domain and decides only the
clear cases; everything else is "unclear" and, for now, stays None.

    3a  split the name into words (hyphens and dots first, then wordsegment)
        find client names: brand parts (harbor, summit...), product names (SaverPlus...), executive full names and
        surnames
    3b  relevant      an executive's full name appears as whole words              janemorrow.com
        relevant      a brand part or product name sits next to a login-type word  saverplus-login.com
        clear         every client name found only sits inside a longer real word  harboroughcycles.co.uk
        unclear       anything else                                                harborviewmarina.com

The doc's test for the unclear middle is whether the domain "can stand on its own feet" without making you think of
the client; that judgement is left to Step 4.
"""
import re
from importlib import import_module

T = import_module("SuspiciousDomain.typosquat")

# words that make a brand or product name read as an account / service page of the client (doc: login, support,
# official, account, careers; plus the usual phishing vocabulary)
KEYWORDS = {"login", "logon", "signin", "sign", "support", "official", "account", "accounts", "careers", "verify",
            "secure", "security", "help", "helpdesk", "service", "services", "billing", "update", "auth", "password",
            "portal", "online", "app", "customer", "pay", "payment", "access", "id", "my", "web", "mobile", "center",
            "team", "recovery", "reset", "unlock", "wallet", "refund", "claim", "claims"}


def client_terms(profile):
    """[(term, kind)] with kind brand_part / product / exec_full / exec_surname (letters only, lower case)."""
    terms = [(T.unglyph(t["term"]), "brand_part") for t in profile["brand_terms"] if t["kind"] == "brand_part"]
    full_brands = {T.unglyph(t["term"]) for t in profile["brand_terms"] if t["kind"] == "full_brand"}
    terms += [(T.unglyph(p), "product") for p in profile["products"] if T.unglyph(p) not in full_brands]
    for e in profile["executives"]:
        parts = [T.unglyph(w) for w in e["name"].split()]
        terms += [("".join(parts), "exec_full"), (parts[-1], "exec_surname")]
    return [(t, k) for t, k in terms if len(t) >= 3]


def find(name, profile):
    """Every client term in the name, with where it sits: whole words, inside a real word, or cut through letters."""
    s, bounds = T.unglyph(name), T.word_bounds(name)
    edges = set(bounds)
    hits = []
    for term, kind in client_terms(profile):
        for m in re.finditer(re.escape(term), s):
            a, b = m.start(), m.end()
            if a in edges and b in edges:
                where = "whole words"
            elif T.cuts_real_word(s, a, b, bounds):
                where = "inside a real word"
            elif kind.startswith("exec"):
                continue                     # a person's name only counts as whole words ('cho' in 'chose')
            else:
                where = "whole words"        # cut through non-words: treat as a real use of the name
            hits.append({"term": term, "kind": kind, "where": where, "start": a, "end": b})
    # a term inside a longer whole-word hit is part of that hit ('harbor' inside 'harborcard')
    return [h for h in hits if not any(o is not h and o["where"] == "whole words" and o["start"] <= h["start"]
                                       and o["end"] >= h["end"] and o["end"] - o["start"] > h["end"] - h["start"]
                                       for o in hits)]


def other_words(name, hits):
    """Words of the name not covered by a whole-word client term."""
    s, bounds = T.unglyph(name), T.word_bounds(name)
    covered = set()
    for h in hits:
        if h["where"] == "whole words":
            covered |= set(range(h["start"], h["end"]))
    return [s[x:y] for x, y in zip(bounds, bounds[1:]) if not set(range(x, y)) <= covered]


def check(domain, profile):
    """-> (verdict, reason, words). verdict: relevant / clear / unclear / no_name."""
    name, _ = T.split_domain(domain)
    hits = find(name, profile)
    words = [T.unglyph(name)[x:y] for x, y in zip(T.word_bounds(name), T.word_bounds(name)[1:])]
    if not hits:
        return "no_name", "no brand part, product or executive name", words
    whole = [h for h in hits if h["where"] == "whole words"]
    full = next((h for h in whole if h["kind"] == "exec_full"), None)
    if full:
        return "relevant", f"executive full name '{full['term']}'", words
    others = other_words(name, whole)
    joined = {a + b for a, b in zip(others, others[1:])}
    for h in whole:
        if h["kind"] in ("brand_part", "product"):
            kw = sorted((set(others) | joined) & KEYWORDS)
            if kw:
                return "relevant", f"{h['kind'].replace('_', ' ')} '{h['term']}' next to '{kw[0]}'", words
    if not whole:
        return "clear", f"'{hits[0]['term']}' only inside a longer real word", words
    names = ", ".join(sorted({f"{h['kind'].replace('_', ' ')} '{h['term']}'" for h in whole}))
    return "unclear", f"{names}; no rule decides", words
