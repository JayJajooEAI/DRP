"""Step 2: typosquat rules (PS doc, "Typosquatting"; plan Step 3). Pure rules, no model.

A domain is a typosquat of the client when its name, after cleaning, is one of the doc's lookalike kinds of a client
name. Client names are the names of its main and regional domains plus its full brand terms. Checks run in a fixed
order; the first that matches wins:

    1 lookalike characters   harb0rbank.com, hàrborbank.com
    2 different ending       harborbank.info
    3 country-code abuse     harborbank.co / .cm / .om (client uses .com)
    4 plural                 harborbanks.com, harborbankes.com
    5 missing / extra dot    wwwharborbank.com, harborbankcom.net
    6 misspelling            harborbnak.com (edit limit by name length, see EDIT_LIMIT)
    7 brand + extra word     harborbank-login.com (the brand must not cut through a real word: harborbankruptcy)

Names of 3 letters or fewer (nbx) only get checks 2-5 and 7 on the short name; letter edits (1, 4, 6) are checked
against the full brand (nbxlogistics), since one changed letter in a 3-letter name is just another short name.
"""
import re
import unicodedata
from functools import lru_cache

import tldextract
import wordsegment
from wordfreq import zipf_frequency

_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)     # bundled suffix list, no download
wordsegment.load()

EDIT_LIMIT = ((6, 2), (9, 2), (10 ** 6, 3))     # (max name length, edits allowed): <=6: 2, 7-9: 2, 10+: 3
SHORT_NAME = 3
CCTLD_ABUSE = {"co", "cm", "om"}
REAL_WORD = 2.0                                  # wordfreq Zipf score for "a real word"
GLYPHS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b", "@": "a", "$": "s",
          "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j", "ԁ": "d",
          "ı": "i", "ł": "l", "ø": "o", "đ": "d"}


def split_domain(domain):
    """'xn--hrborbank-l2a.com' -> ('hàrborbank', 'com'); subdomains are kept in the name."""
    ex = _EXTRACT(domain.lower())
    name = ".".join(x for x in (ex.subdomain, ex.domain) if x)
    try:
        name = ".".join(p.encode().decode("idna") if p.startswith("xn--") else p for p in name.split("."))
    except UnicodeError:
        pass
    return name, ex.suffix


def unglyph(s):
    """Lookalike characters back to letters, accents removed, letters only."""
    s = "".join(GLYPHS.get(ch, ch) for ch in s.lower())
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z]", "", s)


def damerau(a, b, limit):
    """Optimal-string-alignment distance, or limit + 1 once it is clearly larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        prev2, prev = prev, cur
    return prev[-1]


def edit_limit(name):
    return next(k for n, k in EDIT_LIMIT if len(name) <= n)


@lru_cache(None)
def _words(piece):
    return tuple(wordsegment.segment(piece))


def word_bounds(name):
    """Word edges in the cleaned name: hyphens and dots first, then wordsegment inside each piece."""
    bounds, pos = [0], 0
    for piece in re.split(r"[-_.]+", name):
        piece = unglyph(piece)
        for w in (_words(piece) if piece else ()):
            pos += len(w)
            bounds.append(pos)
    return bounds


def cuts_real_word(s, start, end, bounds):
    """The brand at s[start:end] runs into a longer real word ('harbor|bank|ruptcy') -> that word, else None."""
    for x, y in zip(bounds, bounds[1:]):
        inside = min(y, end) - max(x, start)
        if inside > 0 and (x < start or y > end) and (y - x) - inside >= 3 and zipf_frequency(s[x:y], "en") >= REAL_WORD:
            return s[x:y]
    return None


def client_names(profile):
    """[(name, owned endings)] for the main / regional domains, plus the full brand terms."""
    od = profile["owned_domains"]
    names = {}
    for d in od["main"] + od["regional"]:
        n, suf = split_domain(d)
        names.setdefault(unglyph(n), set()).add(suf)
    for t in profile["brand_terms"]:
        if t["kind"] == "full_brand":
            names.setdefault(unglyph(t["term"]), set())
    return names


def check(domain, profile):
    """-> (kind, reason) when the domain is a typosquat of the client, else None."""
    name, suffix = split_domain(domain)
    s = unglyph(name)
    raw = re.sub(r"[^a-z0-9]", "", name.lower())
    names = client_names(profile)
    main_suffixes = {split_domain(d)[1] for d in profile["owned_domains"]["main"]}
    long_names = [b for b in names if len(b) > SHORT_NAME]
    if suffix in names.get(s, ()) and raw == s:
        return None                         # the client's own main or regional domain, not a lookalike of it

    for b in long_names:                                                        # 1 lookalike characters
        if s == b and raw != b:
            return "lookalike characters", f"'{name}' reads as '{b}'"
    for b, owned in names.items():                                              # 2, 3 same name, other ending
        if s == b and raw == b and owned and suffix not in owned:
            if suffix in CCTLD_ABUSE and "com" in main_suffixes:
                return "country-code abuse", f"{b}.{suffix} instead of {b}.com"
            return "different ending", f"{b}.{suffix}"
    for b in long_names:                                                        # 4 plural
        if s in (b + "s", b + "es"):
            return "plural", f"{b} + '{s[len(b):]}'"
    for b in names:                                                             # 5 missing / extra dot
        endings = "|".join(sorted({"com", "net", "org"} | main_suffixes))
        if re.fullmatch(rf"w{{2,4}}{b}|{b}({endings})", s):
            return "missing or extra dot", f"'{name}' is '{b}' with a dot dropped"
    for b in long_names:                                                        # 6 misspelling
        k = edit_limit(b)
        dist = damerau(s, b, k)
        if 0 < dist <= k:
            return "misspelling", f"{dist} edit(s) from '{b}' (limit {k} for {len(b)} letters)"
    for b in sorted(names, key=len, reverse=True):                              # 7 brand + extra word
        i = s.find(b)
        if i < 0 or len(s) == len(b):
            continue
        word = cuts_real_word(s, i, i + len(b), word_bounds(name))
        if word:
            continue                                                            # 'harbor bankruptcy'
        rest = (s[:i] + " " + s[i + len(b):]).strip()
        return "brand + extra word", f"'{b}' + '{rest}'"
    return None
