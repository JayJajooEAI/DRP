"""Stage 1a: numeric signals per screenshot from parsed OCR fields + avatar matching + client profile. Used by rules and ML."""

import re, json, numpy as np, cv2

# rapidfuzz.fuzz = fuzzy text similarity (0..100)
from rapidfuzz import fuzz
from PIL import Image
from utilities.config import *
from utilities.layout_parser import parse

# Accent colour of each platform (its verified-badge colour); used to find the badge
ACC = {
    "X": "#0f1419",
    "Facebook": "#1877f2",
    "Instagram": "#c13584",
    "LinkedIn": "#0a66c2",
    "Telegram": "#229ed9",
    "TikTok": "#fe2c55",
}


def rx(p):
    return re.compile(p, re.I)  # compile a case-insensitive regular expression


# LEX = word lists (as regexes). Each is counted in the name+bio ("h_...") and in the posts ("p_...")
LEX = dict(
    # claims to be the official/support/care presence of the brand
    claim=rx(
        r"\b(official|support|help ?desk|help|customer (care|service|support)|care|service (center|centre|desk)|billing|account (recovery|access)|we are|our team|agents?|assistance|contact us)\b"
    ),
    # says it is a fan / parody / review / news / unofficial page
    disc=rx(
        r"\b(fan|fans|parody|satire|satirical|unofficial|not affiliated|no affiliation|not (the )?(real|official)|tribute|review|reviews|reviewer|news|critic|complaints?|watchdog|independent|just joking|jokes?|spoof|memes?|community (page|group)|discussion)\b"
    ),
    # "my own opinions" / personal account wording
    own=rx(
        r"\b(my own|views? (are )?my own|opinions? (are )?my own|personal account|personal page|my personal|all opinions)\b"
    ),
    # scam / phishing / giveaway / crypto wording
    scam=rx(
        r"\b(giveaway|airdrop|crypto|bitcoin|wallet|claim your|prize|winner|free gift|invest\w*|double your|limited offer|verify your|click (the )?link|urgent|suspended|refund|lottery|cash|payment link|moons|stock tips?|changing our name|new account|wallet address|send \w+ to)\b"
    ),
    # hiring / careers wording
    hire=rx(r"\b(hiring|jobs?|careers?|recruit\w*|apply|applications?|talent|interview|vacanc\w*)\b"),
    # "DM us" / WhatsApp style contact requests
    dm=rx(r"\b(dm|dms|direct message|message us|inbox|whatsapp|telegram me|text us)\b"),
    # partner / reseller / agency wording
    partner=rx(
        r"\b(partners?|resellers?|certified|authori[sz]ed|agency|consultant|consulting|integrat\w+|vendor)\b"
    ),
    # executive job titles
    role=rx(
        r"\b(chief|ceo|cfo|cto|coo|ciso|cmo|vp|vice president|president|director|head of|board member|general counsel|managing)\b"
    ),
    # personal-life wording (hobbies, family, other jobs)
    personal=rx(
        r"\b(hobby|hiking|coffee|dog|cat|travel|photography|cooking|books|cycling|gardener?|garden|marathon|climbing|woodworker|plant|family|dad|mom|designer|agent in|real estate|teacher|student)\b"
    ),
    # "works at ..." / employee wording
    employee=rx(
        r"\b(works? at|working at|employee|engineer|manager|analyst|account manager|ex-|former|intern|specialist|@ )\w*"
    ),
    # promotional / discount wording
    promo=rx(r"\b(offer|deal|deals|discount|sale|cheap|% off|half price|insider|special|limited)\b"),
)

# Handle suffixes that make an account look official (brand_support, brandhq, ...)
SUF = rx(
    r"(official|real|support|help|care|service|team|hq|billing|security|secure|login|verify|assist|desk|center|centre|online|app|admin)"
)

# Platform's own domains: links to these say nothing about the account, so they are ignored
GEN_DOM = {
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "telegram.org",
    "t.me",
    "tiktok.com",
}


def sl(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())  # slug: lowercase letters+digits only


def fz(a, b):
    """Similarity 0..100 between two texts. 'partial' = best matching piece, so 'Acme Support' still matches 'Acme'. Very short texts must match exactly."""
    # Compare slugs; empty = no similarity
    a, b = sl(a), sl(b)
    if not a or not b:
        return 0.0

    # partial match if both are >= 4 characters, otherwise only an exact match counts
    return float(fuzz.partial_ratio(a, b) if min(len(a), len(b)) >= 4 else (100.0 if a == b else 0.0))


def cnt(rg, t):
    return len(rg.findall(t)) if t else 0  # how many times a word-list regex matches a text


def months(j):
    """Account age in months from a 'YYYY-MM' join date (relative to Nov 2026, the date of the synthetic data); -1 if unknown."""
    m = re.match(r"(\d{4})-(\d{2})", j or "")
    return (2026 * 12 + 11) - (int(m.group(1)) * 12 + int(m.group(2))) if m else -1


def sld(d):
    return d.split("/")[0].lower().strip(".")  # a link -> just its domain (drop the path)


def is_official(d, off):
    d = sld(d)
    return d == off or d.endswith(
        "." + off
    )  # is the link the client's official domain (or a subdomain of it)?


def badge(img, plat, name_box, lay):
    """Detect a verified-badge disc (filled ~17px circle in the platform accent colour) in the header region outside the avatar."""
    # col = this platform's accent colour; no known colour -> no badge
    col = ACC.get(plat)
    if not col:
        return 0

    # c = the colour as an (R,G,B) array; a = the header region of the screenshot (rows 84-340, columns 200-800) as numbers
    c = np.array([int(col[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.int32)
    a = np.asarray(img.convert("RGB"), dtype=np.int32)[84:340, 200:800]

    # m = 1 for pixels close to the accent colour; then group touching pixels into blobs (connected components)
    m = (np.abs(a - c).sum(2) < 60).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)

    # For each blob: w, h = width, height; ar = pixel area. A badge is ~14-20 px square and 50-88% filled (a disc fills ~78% of its box)
    for i in range(1, n):
        w, h, ar = st[i, 2], st[i, 3], st[i, 4]
        if 14 <= w <= 20 and 14 <= h <= 20 and 0.5 <= ar / (w * h) <= 0.88:
            return 1
    return 0


def signals(r, ocr_lines, av, img):
    """All signals of ONE page -> dict. r = csv row (only 'customer' is used), ocr_lines = step 01 output, av = step 02 scores, img = the screenshot."""
    # C = this client's profile; B = brand name; bs = brand slug; off = the client's official domain
    C = CLIENT[r["customer"]]
    B = C["brand"]
    bs = sl(B)
    off = C["domain"]

    # Parse the OCR into fields; ptxt = all post text joined; head = name + bio joined
    p = parse(ocr_lines)
    name, handle, bio = p["name"], p["handle"], p["bio"]
    ptxt = " ".join(x["text"] for x in p["posts"])
    head = f"{name} {bio}"

    # s = the signals dict being built
    s = {}
    s["platform"] = p["platform"]
    s["layout"] = p["layout"]

    # --- name / handle vs brand: fuzzy score, "contains brand", "equals brand"
    s["name_brand"] = fz(name, B)
    s["name_has_brand"] = int(bs in sl(name))
    s["handle_brand"] = fz(handle, B)
    s["handle_has_brand"] = int(bs in sl(handle))
    s["handle_ratio"] = float(fuzz.ratio(sl(handle), bs))
    s["name_ratio"] = float(fuzz.ratio(sl(name), bs))
    s["name_equals_brand"] = int(sl(name) == bs)
    s["handle_equals_brand"] = int(sl(handle) == bs)

    # --- handle shape: has an official-looking suffix, number of digits, length, number of "_" and "."
    s["handle_claim"] = int(bool(SUF.search(handle)))
    s["handle_digits"] = sum(ch.isdigit() for ch in handle)
    s["handle_len"] = len(handle)
    s["handle_us"] = handle.count("_") + handle.count(".")

    # How many characters the handle has beyond the brand name (-1 if the brand is not in the handle)
    s["handle_extra"] = max(0, len(sl(handle)) - len(bs)) if bs in sl(handle) else -1

    # --- bio and posts: brand mentioned? length, empty bio, number of posts seen, no posts
    s["bio_brand"] = int(bs in sl(bio))
    s["post_brand"] = sum(bs in sl(x["text"]) for x in p["posts"])
    s["bio_len"] = len(bio)
    s["no_bio"] = int(p["no_bio"] or len(bio) < 3)
    s["n_posts_seen"] = len(p["posts"])
    s["no_posts"] = int(len(p["posts"]) == 0)

    # Share of posts whose author name equals the profile name (-1 if no posts)
    s["post_name_match"] = (
        float(np.mean([fz(x["name"], name) >= 85 for x in p["posts"]])) if p["posts"] else -1
    )

    # Banner wordmark: is there banner text, and does it match the brand?
    s["banner_brand"] = int(bool(p["banner_text"]) and fz(p["banner_text"], B) >= 70)
    s["banner_text"] = int(bool(p["banner_text"]))

    # links
    # ds = external domains found (platform domains removed); offd = the official ones; oth = all others
    ds = [
        d
        for d in p["domains"]
        if sld(d) not in GEN_DOM and not sld(d).endswith(("facebook.com", "linkedin.com"))
    ]
    offd = [d for d in ds if is_official(d, off)]
    oth = [d for d in ds if not is_official(d, off)]

    # Link flags: official link present / another domain present / no link at all
    s["link_official"] = int(bool(offd))
    s["link_other"] = int(bool(oth))
    s["link_none"] = int(not ds)

    # Look-alike link: another domain that contains the brand or is >= 70% similar to the official domain
    s["link_lookalike"] = int(any(bs in sl(sld(d)) or fuzz.ratio(sl(sld(d)), sl(off)) >= 70 for d in oth))

    # Brand inside another domain; link only in the bio (no link field); link FIELD is official / is another domain
    s["link_brand_sub"] = int(any(bs in sl(sld(d)) for d in oth))
    s["link_bio_only"] = int(bool(p["link"]) is False and bool(ds))
    s["link_field_official"] = int(bool(p["link"]) and is_official(p["link"], off))
    s["link_field_other"] = int(bool(p["link"]) and not is_official(p["link"], off))

    # Highest similarity between any other domain and the official domain
    s["link_sim"] = max([fuzz.ratio(sl(sld(d)), sl(off)) for d in oth] or [0.0])

    # lexicon counts: head (name+bio) and posts
    for k, rg in LEX.items():
        s[f"h_{k}"] = cnt(rg, head)
        s[f"p_{k}"] = cnt(rg, ptxt)

    # "Any disclaimer" and "any scam wording" flags (head or posts)
    s["disc_any"] = int(s["h_disc"] + s["p_disc"] > 0)
    s["scam_any"] = int(s["h_scam"] + s["p_scam"] > 0)

    # executives
    # ex = the client's protected executives; er = name similarity to each (also tried with the words reversed); k = the best-matching executive
    ex = C["execs"]
    er = [
        max(
            fuzz.ratio(sl(name), sl(e["name"])),
            fuzz.ratio(sl(name), sl("".join(reversed(e["name"].split())))),
        )
        for e in ex
    ]

    # Best name score and the "exact" flag (>= 92)
    k = int(np.argmax(er))
    s["exec_name"] = float(er[k])
    s["exec_name_exact"] = int(er[k] >= 92)

    # Does the bio contain that executive's role? Any executive-type title? Is the brand named in the bio?
    rt = sl(ex[k]["role"])
    s["exec_role_bio"] = int(fz(bio, ex[k]["role"]) >= 85) if len(rt) >= 4 else 0
    s["exec_role_any"] = int(s["h_role"] > 0)
    s["exec_brand_bio"] = int(bs in sl(bio))

    # avatar
    # Copy every avatar score from step 02, prefixed with "av_"
    for kk, v in av.items():
        s[f"av_{kk}"] = v

    # Strong logo match (pixel correlation > 0.5 or CLIP > 0.9); strong executive-photo match (> 0.70 or > 0.80)
    s["av_logo_strong"] = int(av["logo_ncc"] > 0.5 or av["logo_clip"] > 0.9)
    s["av_exec_strong"] = int(av["exec_ncc"] > 0.70 or av["exec_clip"] > 0.80)

    # Default avatar: the picture correlates > 0.7 with the generic silhouette
    s["av_default"] = int(av["sil_ncc"] > 0.7)

    # account stats
    # Followers (LinkedIn: connections; -1 if not shown), following, number of posts
    f = (
        p["followers"]
        if p["followers"] is not None
        else (p["connections"] if p["connections"] is not None else -1)
    )
    s["followers"] = f
    s["following"] = p["following"] if p["following"] is not None else -1
    s["posts_n"] = p["posts_n"] if p["posts_n"] is not None else -1

    # Account age, log of followers (tames huge values), followers-to-following ratio
    s["age_months"] = months(p["joined"])
    s["log_followers"] = float(np.log1p(max(f, 0)))
    s["foll_ratio"] = (f / (s["following"] + 1)) if f >= 0 and s["following"] >= 0 else -1

    # Verified badge found in the picture?
    s["verified"] = badge(img, p["platform"], p["name_box"], p["layout"])

    # Private columns (start with "_") keep the readable text for the report; the ML model drops them
    s["_name"], s["_handle"], s["_bio"], s["_posts"], s["_link"], s["_exec_k"] = (
        name,
        handle,
        bio,
        ptxt,
        p["link"],
        ex[k]["name"],
    )
    return s
