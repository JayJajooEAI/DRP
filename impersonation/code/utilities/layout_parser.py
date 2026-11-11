"""Stage 0b: turn OCR lines of a profile screenshot into structured fields (screenshot-only, no CSV text used)."""

import re

# Text found in the URL bar -> platform name
PLAT = {
    "x": "X",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "telegram": "Telegram",
    "tiktok": "TikTok",
}

# Platform -> page layout type (feed = text feed, grid = photo grid, pro = professional profile)
LAYOUT = {
    "X": "feed",
    "Facebook": "feed",
    "Telegram": "feed",
    "Instagram": "grid",
    "TikTok": "grid",
    "LinkedIn": "pro",
}

# A post header line: "Name @handle · 3h" (name, @handle, then an age like 5m / 3h / 2d / 1w)
POSTHDR = re.compile(r"^(.*?)\s*@\s*[\w.\-]+(?:\s[\w.\-]+)?\s*[-·•–.:]*\s*\d+\s*[smhdw]\s*$", re.I)

# A stats line: a number (optionally with K / M) followed by posts / followers / following / connections
STATS = re.compile(r"(\d[\d,.]*\s*[KkMm]?)\s*(posts|followers|following|connections)", re.I)

# A web domain such as shop.example.com/path (a list of common endings, then an optional /path)
DOMAIN = re.compile(
    r"(?<![@\w.\-])((?:[a-z0-9][a-z0-9\-]*\.)+(?:com|org|net|co|io|info|biz|app|xyz|top|online|site|support|help|link|me|us|uk|de|fr|br|pt|es|cc|ru|cn|tk|ly|shop|store|live|club|pro|cloud|page|click|vip|gov|edu|mil|ai|dev|tech|in|ca|au|eu|it|nl|se|no|ch|at|mx|ar|cl|ph|ng|za|id|ir|kz|tv|fun|xyz|work|today|world|space|website|net)\b(?:/[^\s]*)?)",
    re.I,
)


def num(s):
    """'12.5K' -> 12500.0, '3,400' -> 3400.0, '1.2M' -> 1200000.0; None if it is not a number."""
    # Remove commas and spaces, lowercase; m = multiplier (1 unless K or M)
    s = s.replace(",", "").replace(" ", "").lower()
    m = 1
    if s.endswith("k"):
        m = 1000
        s = s[:-1]
    elif s.endswith("m"):
        m = 1_000_000
        s = s[:-1]
    try:
        return float(s) * m
    except:
        return None


# The same list of domain endings as plain text (kept for reference)
TLD = r"(?:com|org|net|co|io|info|biz|app|xyz|top|online|site|support|help|link|me|us|uk|de|fr|br|pt|es|cc|ru|cn|tk|ly|shop|store|live|club|pro|cloud|page|click|vip|gov|edu|mil|ai|dev|tech|in|ca|au|eu|it|nl|se|no|ch|at|mx|ar|cl|ph|ng|za|id|ir|kz|tv|fun|xyz|work|today|world|space|website|net)"


def norm_dom(t):
    return re.sub(
        r"\s*([./])\s*", r"\1", t.strip()
    )  # remove OCR spaces around "." and "/" so "shop . acme . com" becomes "shop.acme.com"


def clean_url(t):
    return re.sub(r"\s+", "", t)  # remove all spaces from a URL read by OCR


def parse(lines):
    """OCR lines (each with x, y, x1, h, t) -> dict of fields: platform, handle, name, bio, link, domains, posts, follower counts, join date, banner text..."""
    # o = the result, every field starts empty
    o = dict(
        platform="",
        handle_url="",
        name="",
        handle="",
        bio="",
        link="",
        domains=[],
        posts=[],
        followers=None,
        following=None,
        posts_n=None,
        connections=None,
        joined="",
        no_bio=False,
        no_posts=False,
        banner_text="",
        name_box=None,
        layout="",
    )
    hdr_lines, post_start = [], None

    # Find where the posts begin: the first line that looks like a post header ("Name @handle 3h") below the top of the page
    for i, l in enumerate(lines):
        if POSTHDR.match(l["t"].strip()) and l["y"] > 90 and l["h"] >= 10:
            post_start = i
            break

    # head = everything above the posts (the profile header); if no posts, the whole page
    head = lines[:post_start] if post_start is not None else lines

    # url bar
    # The browser URL bar is at the very top (y < 32): the first line there tells us the platform and the handle
    for l in head:
        if l["y"] < 32:
            # u = URL text without spaces, lowercase; seg = last part after "/" (unused)
            u = clean_url(l["t"]).lower()
            seg = u.split("/")[-1] if "/" in u.replace("https://", "") else ""

            # Platform = the first platform word found in the URL
            for k in PLAT:
                if re.search(r"(?:www\.)" + re.escape(k) + r"|" + re.escape(k) + r"\.?c", u):
                    o["platform"] = PLAT[k]
                    break

            # Handle from the URL: the last segment, when the URL has at least 3 slashes
            o["handle_url"] = u.split("/")[-1] if u.count("/") >= 3 else ""
            break

    # Fallback: the platform name printed as a line near the top
    if not o["platform"]:
        for l in head:
            if l["y"] < 70 and l["t"].strip().lower() in PLAT:
                o["platform"] = PLAT[l["t"].strip().lower()]

    # Layout type follows from the platform
    o["layout"] = LAYOUT.get(o["platform"], "feed")

    # cand = header lines that might be name / handle / bio / link
    cand = []
    for l in head:
        # t = line text, tl = lowercase
        t = l["t"].strip()
        tl = t.lower()

        # Skip the URL bar, the platform name line, tab labels ("pdf", "activity") and 1-character noise
        if l["y"] < 32 or (l["y"] < 70 and tl in PLAT) or tl in ("pdf", "activity") or len(t) < 2:
            continue

        # In feed / pro layouts the banner sits at y 74-224: text to the right of the avatar there is the banner wordmark
        if o["layout"] != "grid" and 74 <= l["y"] <= 224:
            if l["x"] > 200 and not o["banner_text"]:
                o["banner_text"] = t
            continue

        # Skip any text that sits inside the avatar picture
        if (o["layout"] != "grid" and l["x"] < 200 and 165 < l["y"] < 280) or (
            o["layout"] == "grid" and l["x"] < 212 and 88 < l["y"] < 216
        ):
            continue  # text inside avatar

        # Stats line ("120 posts · 3.4K followers · 80 following")
        m = STATS.findall(t)
        if m:
            # Store each number under its label
            for v, k in m:
                k = k.lower()
                n = num(v)
                if k == "posts":
                    o["posts_n"] = n
                elif k == "followers":
                    o["followers"] = n
                elif k == "following":
                    o["following"] = n
                elif k == "connections":
                    o["connections"] = n

            # The same line may hold a join date ("Joined 2021-04")
            mj = re.search(r"(?:joined|since)\s*(\d{4})\s*[-–]\s*(\d{1,2})", t, re.I)
            if mj:
                o["joined"] = f"{mj.group(1)}-{int(mj.group(2)):02d}"

            # ... or a link before the first separator ("acme.com · 3K followers")
            first = re.split(r"\s[·\-–•]\s", t)[0]
            if first != t and not STATS.search(first):
                for dm in DOMAIN.findall(norm_dom(first)):
                    o["link"] = o["link"] or dm.lower()
                    break
            continue

        # "No posts yet" message
        if "no posts yet" in tl:
            o["no_posts"] = True
            continue

        # Join-date line: "<link> · Joined 2021-04"; the part before it may be the profile link
        mj = re.search(r"(?:joined|since)\s*(\d{4})\s*[-–]\s*(\d{1,2})", t, re.I)
        if mj:
            o["joined"] = f"{mj.group(1)}-{int(mj.group(2)):02d}"
            pre = t[: mj.start()].replace("Joined", "").replace("Member", "").strip(" ·-")
            for dm in DOMAIN.findall(norm_dom(pre)):
                o["link"] = dm.lower()
                break
            continue

        # "No bio" placeholder
        if tl == "no bio":
            o["no_bio"] = True
            continue

        # Large text (height >= 30) in the banner band is the brand wordmark
        if l["y"] > 74 and l["y"] < 232 and l["h"] >= 30 and not o["banner_text"]:
            o["banner_text"] = t
            continue  # wordmark in banner

        # Anything else could be the name, handle, bio or link
        cand.append(l)

    # candidates: name / @handle / bio / link lines
    # texts = candidate lines below the top bar
    texts = [c for c in cand if c["y"] > 80]

    # The first line starting with "@" is the handle
    for c in texts:
        t = c["t"].strip()
        if t.startswith("@") and not o["handle"]:
            o["handle"] = t.lstrip("@").replace(" ", "_")

    # rest = the lines that are not the handle; hu = the handle from the URL, simplified for comparison
    rest = [c for c in texts if not c["t"].strip().startswith("@")]
    hu = o["handle_url"].replace("_", "").replace(" ", "").lower()

    # rest2 = rest minus handle echoes and link lines
    rest2 = []
    for c in rest:
        t = c["t"].strip()

        # Grid layouts print the handle again near the top: drop it when it equals the URL handle
        if (
            hu
            and o["layout"] == "grid"
            and c["y"] < 150
            and re.sub(r"[^a-z0-9.]", "", t.lower().replace("_", "")) == re.sub(r"[^a-z0-9.]", "", hu)
        ):
            continue  # grid layout handle line

        # A short line that is just a domain is the profile link field
        if DOMAIN.fullmatch(norm_dom(t)) and len(t) < 45 and " " not in norm_dom(t):
            o["link"] = o["link"] or norm_dom(t).lower()
            continue
        rest2.append(c)

    # Name = first remaining line; bio = the lines after it
    if o["layout"] == "grid":
        # order: handle, stats, name, bio
        o["name"] = rest2[0]["t"].strip() if rest2 else ""
        bio = rest2[1:]
    else:
        o["name"] = rest2[0]["t"].strip() if rest2 else ""
        bio = rest2[1:]

        # Remember where the name line ends (x1, y, height): the verified badge is drawn next to it
        if rest2:
            o["name_box"] = (rest2[0]["x1"], rest2[0]["y"], rest2[0]["h"])

    # Bio = the remaining lines joined into one text
    o["bio"] = " ".join(c["t"].strip() for c in bio)

    # posts
    # Walk through the lines from where the posts start: each post header starts a new post, and the lines until the next header are its text
    i = post_start
    while i is not None and i < len(lines):
        t = lines[i]["t"].strip()
        m = POSTHDR.match(t)
        if m and lines[i]["h"] >= 10:
            # pt = text lines of this post; j = scanning index
            pt = []
            j = i + 1
            while j < len(lines) and not (POSTHDR.match(lines[j]["t"].strip()) and lines[j]["h"] >= 10):
                tt = lines[j]["t"].strip()

                # Keep real sentences; skip tiny noise such as "12 3 likes" button labels
                if (
                    len(re.findall(r"[A-Za-z]{3,}", tt)) >= 1
                    and not re.fullmatch(r"[\d\sA-Za-z:()|]{0,14}", tt)
                    or len(tt) > 14
                ):
                    pt.append(tt)
                j += 1

            # Save the post (author name from the header + its text) and jump to the next header
            o["posts"].append(dict(name=m.group(1).strip(), text=" ".join(pt)))
            i = j
        else:
            i += 1

    # Fallbacks: no name -> use the first post's author; no @handle -> use the URL handle
    if not o["name"] and o["posts"]:
        o["name"] = o["posts"][0]["name"]
    if o["layout"] == "grid" and not o["handle"]:
        o["handle"] = o["handle_url"]
    if not o["handle"]:
        o["handle"] = o["handle_url"]

    # If the name box was not set, find the name line again to get its position
    if not o["name_box"] and o["name"]:
        for l in lines:
            if l["t"].strip() == o["name"] and l["y"] > 80:
                o["name_box"] = (l["x1"], l["y"], l["h"])
                break

    # Collect every domain mentioned in the bio and posts (OCR spaces around dots removed), plus the link field
    alltext = " ".join([o["bio"]] + [p["text"] for p in o["posts"]])
    for dm in DOMAIN.findall(re.sub(r"\s+(?=\.)|(?<=\.)\s+", "", alltext)):
        o["domains"].append(dm.lower())
    if o["link"]:
        o["domains"].append(o["link"])

    # Unique, sorted, trailing punctuation removed
    o["domains"] = sorted(set(d.rstrip(".,") for d in o["domains"]))
    return o
