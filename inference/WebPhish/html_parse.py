"""Step 1 of a request: pull the fields the prompt needs (title, headings, forms, inputs, links, script URLs, visible text) out of the
page's HTML. Same parser as the batch pipeline (code/01_parse_html.py); it never fails on malformed HTML (whatever parsed is kept).
Scripts are never run: script inputs and URLs are read from the script text by pattern matching."""
import re

import lxml.html

MAX_HTML_BYTES = 5_000_000
# No caps on how many inputs, links, brand hints, headings or iframes are kept: everything found in the HTML is
# stored, and the prompt builder (step 06) decides what fits. Only single strings are bounded.
MAX_TITLE = 200
MAX_CLICKABLE_TEXT = 80      # longer text on a clickable element is a container, not a button label
MAX_BARE_TEXT = 30           # a plain span/div counts as a sign-in control only if its whole text is this short
INPUT_ATTRS = ("name", "id", "placeholder", "aria-label", "autocomplete", "title")
TEXT_INPUT_TYPES = {"text", "email", "tel", "number", "search", "url"}

# Elements that start a new visual block or are separate controls: a space is inserted around them so that
# neighbouring text does not run together ("Log inSign up"). Inline tags (span, b, font, ...) are left alone so
# "<b>Pay</b>Pal" stays one word.
SPACED_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "fieldset", "figcaption", "figure",
    "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav", "ol", "p", "pre",
    "section", "table", "tbody", "thead", "tfoot", "tr", "td", "th", "ul", "a", "button", "input", "label",
    "option", "select", "textarea", "img", "iframe", "center", "caption", "legend", "summary", "details",
}
INVISIBLE_CHARS = dict.fromkeys(map(ord, "﻿​‌‍⁠­"), None)

# Sign-in wording decides 3.2 (labelling guide: by the link's wording, not where it leads). STRONG are sign-in /
# log-in phrases; AMBIGUOUS words ("access", "enter") count as sign-in only as a short button label (Itaú's
# "Acessar" button, not "Acessar a página Cartões"); WEAK are account / profile words that also appear in help
# links; LOGOUT is kept so the model can tell "Log out"-only pages apart.
STRONG_LINK = re.compile(
    r"\b(log[ -]?ins?|log[ -]?on|sign[ -]?ins?|sign[ -]?on|already a customer|"
    r"connexion|se connecter|vous connecter|s.identifier|identifiez[ -]vous|"
    r"(?<!neu )anmelden|einloggen|login|inloggen|iniciar sesi[oó]n|"
    r"giri[sş] yap|oturum a[cç]|σύνδεση|zaloguj|войти|đăng nhập)\b|ログイン|登录|로그인", re.I)
AMBIGUOUS_LINK = re.compile(r"^\W*(entrar|acessar|acceder|accedi|accesso|access)\W*$", re.I)
WEAK_LINK = re.compile(
    r"\b(my account|your account|accounts?|profiles?|mon compte|espace client|mein konto|mi cuenta|minha conta|"
    r"il mio account|hesab[iı]m)\b", re.I)
LOGOUT_LINK = re.compile(r"\b(log[ -]?out|sign[ -]?out|log[ -]?off|sign[ -]?off|signing out|logging out|"
                         r"d[ée]connexion|abmelden)\b", re.I)
# account/profile labels that read as help-article titles, not controls ("Managing Your Account", "How to ...").
# Action buttons ("Create Account", "Secure My Account", "Use another account") are kept: the labels count some of
# them as 3.2, so the model, not the parser, decides.
NOT_ACCOUNT_ENTRY = re.compile(
    r"^\W*(creating|managing|deleting|deactivating|updating|closing|opening|changing|recovering|adding|"
    r"how|what|why|when|about|learn|understanding)\b", re.I)
HOSTNAME = re.compile(r"^[\w-]+(\.[\w-]+)+(/\S*)?$")   # "account.jagex.com" in a list of official domains
NOT_A_SIGNIN = re.compile(r"\b(dismiss|close|cancel|schlie(ss|ß)en|fermer)\b", re.I)
MAX_WEAK_LABEL = 40        # longer account/profile phrases are help-article titles
SHORT_LABEL = 24           # sign-in controls are short; long sign-in phrases are usually help-article titles
MAX_BARE_LABEL = 20        # a plain span/div is a sign-in control only with a label this short ("Hello, sign in")

LOGO_TOKEN = re.compile(r"^(logo(?!n|ut|ff)\w*|\w*logo|brand|brandname|brandmark|brand-?logo|wordmark)$")
HOME_WORD = re.compile(r"\bhome(page)?\b", re.I)    # "eBay Home", "AT&T Home": the header logo link
ICON_WORDS = re.compile(r"\b(icon|search|menu|close|clear|loading|filter|back|arrow|chevron|expand|collapse|"
                        r"play|pause|next|previous|share|cart|caret|phone|spinner|hamburger|dismiss|toggle|"
                        r"switch|dropdown|drawer|ellipsis|options|click here)\b", re.I)
NOT_BRAND_AREA = re.compile(r"cookie|consent|onetrust|\bot-|cybot|didomi|usercentrics|truste|gdpr|000webhost|"
                            r"hostinger|webhost", re.I)
NAV_WORDS = {"home", "personal", "business", "store", "shop", "help", "support", "careers", "blog", "news", "about",
             "contact", "sports", "deals", "menu", "sign up", "register", "join", "join now"}
MAX_SELECT_OPTIONS = 5     # longer option lists (countries, dates, security questions) are dropped from visible text
HEADER_AREA = re.compile(r"header|masthead|navbar|top-?nav|gh-|uhf", re.I)
HIDDEN_CLASS = re.compile(r"(^|[\s_-])(hidden|hide|sr-only|visually-?hidden|d-none|invisible)($|[\s_-])|[a-z]Hidden\b",
                          re.I)
NOT_A_CONTROL = re.compile(r"greet|title|heading|headline|callout|message|instruction", re.I)
ICON_LIGATURE = re.compile(r"\b[a-z]+(?:_[a-z]+)+\b")   # icon-font names used as text: seta_direita, arrow_forward
MAX_FAVICONS = 2
MAX_ANCESTORS = 12
SCRIPT_INPUT = re.compile(r"<input\b[^>]{0,300}", re.I)
SCRIPT_INPUT_TYPE = re.compile(r"type\s*=\s*\\?[\"']?(password|email|tel|text|number)", re.I)
# URLs a script sends the victim to (redirect) or sends data to (request). Only quoted literals are matched.
_Q = r"""(?P<q>["'`])(?P<url>(?:(?!(?P=q)).){1,300}?)(?P=q)"""
REDIRECT_PATTERNS = [
    re.compile(r"(?:window\.|document\.|top\.|self\.|parent\.)?location(?:\.href)?\s*=\s*" + _Q),
    re.compile(r"location\.(?:replace|assign)\(\s*" + _Q),
    re.compile(r"window\.open\(\s*" + _Q),
]
REQUEST_PATTERNS = [   # (pattern, method if fixed); "call" is the text after the match, searched for method / data
    (re.compile(r"\$\.(?:ajax)\(\s*\{(?P<call>.{0,800}?)\}\s*\)", re.S), None),
    (re.compile(r"\$\.(?P<m>post|get|getJSON)\(\s*" + _Q + r"(?P<call>.{0,300})", re.S), "m"),
    (re.compile(r"\bfetch\(\s*" + _Q + r"(?P<call>.{0,300})", re.S), None),
    (re.compile(r"\baxios\.(?P<m>post|get|put)\(\s*" + _Q + r"(?P<call>.{0,300})", re.S), "m"),
    (re.compile(r"\.open\(\s*[\"'](?P<m>GET|POST|PUT)[\"']\s*,\s*" + _Q, re.I), "m"),
    (re.compile(r"\.action\s*=\s*" + _Q), "form action"),
]
CALL_URL = re.compile(r"\burl\s*:\s*" + _Q)
CALL_METHOD = re.compile(r"\b(?:method|type)\s*:\s*[\"'](\w+)[\"']", re.I)
CALL_DATA = re.compile(r"\b(?:data|body)\s*:\s*(\{[^{}]{0,400}\}|[\w$.()#'\"-]{1,80})", re.S)
POSITIONAL_DATA = re.compile(r"^\s*,\s*(\{[^{}]{0,400}\}|[\w$.()#'\"-]{1,80})", re.S)   # $.post(url, {...})
DATA_KEY = re.compile(r"[\"']?([A-Za-z_][\w-]{0,30})[\"']?\s*:")
META_REFRESH_URL = re.compile(r"url\s*=\s*['\"]?([^'\";]+)", re.I)
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")
META_CHARSET = re.compile(rb"<meta[^>]+charset=[\"']?([a-zA-Z0-9_-]+)", re.I)


def _clean(text) -> str:
    return " ".join((text or "").translate(INVISIBLE_CHARS).split())


def _text(el) -> str:
    return _clean(el.text_content())


def _decode(raw: bytes) -> str:
    """UTF-8 if it is valid, else the page's declared charset, else cp1252 (old kits are often latin-1)."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    m = META_CHARSET.search(raw[:4096])
    for enc in ([m.group(1).decode()] if m else []) + ["cp1252"]:
        try:
            return raw.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _parse_doc(raw: bytes):
    # lxml rejects str input that carries an XML encoding declaration, so drop it
    text = re.sub(r"^\s*<\?xml[^>]*>", "", _decode(raw))
    text = CONTROL_CHARS.sub("", text)   # lxml refuses to set text containing NUL / control characters
    return lxml.html.document_fromstring(text if text.strip() else "<html></html>")


def _dedupe(seq):
    return list(dict.fromkeys(s for s in seq if s))


def _lower_attr(el, name: str) -> str:
    return (el.get(name) or "").strip().lower()


def _script_urls(doc) -> list:
    """Redirect and request targets written in inline scripts, inline event handlers and <meta refresh>:
    'request get sendmail.php (sends: email, password)', 'redirect https://...'."""
    texts = [s.text or "" for s in doc.iter("script") if not s.get("src")]
    texts += [el.get(a) for a in ("onclick", "onsubmit", "onload") for el in doc.xpath(f"//*[@{a}]")]
    found = []
    for text in texts:
        for pat in REDIRECT_PATTERNS:
            found += [f"redirect {m.group('url').strip()}" for m in pat.finditer(text)]
        for pat, method in REQUEST_PATTERNS:
            for m in pat.finditer(text):
                call = m.groupdict().get("call") or ""
                call = call.split(");")[0]   # stay inside this call, not the next statement
                url = m.groupdict().get("url") or (CALL_URL.search(call).group("url") if CALL_URL.search(call) else "")
                if not url.strip():
                    continue
                verb = (m.group("m") if method == "m" else method) or \
                    (CALL_METHOD.search(call).group(1) if CALL_METHOD.search(call) else "")
                entry = f"request {verb.lower() + ' ' if verb else ''}{url.strip()}"
                data = CALL_DATA.search(call) or (POSITIONAL_DATA.search(call) if method == "m" else None)
                if data:
                    keys = DATA_KEY.findall(data.group(1)) if data.group(1).startswith("{") else [data.group(1)]
                    entry += f" (sends: {', '.join(_dedupe(keys))})"
                found.append(entry)
    for meta in doc.iter("meta"):
        if _lower_attr(meta, "http-equiv") == "refresh":
            m = META_REFRESH_URL.search(meta.get("content") or "")
            if m:
                found.append(f"redirect (meta refresh) {m.group(1).strip()}")
    return _dedupe(found)


def _script_inputs(doc) -> list:
    """Credential-type <input> tags written inside scripts (innerHTML / template strings), not in the DOM."""
    found = []
    for s in doc.iter("script"):
        for m in SCRIPT_INPUT.finditer(s.text or ""):
            t = SCRIPT_INPUT_TYPE.search(m.group(0))
            if t:
                name = re.search(r"(?:name|id|placeholder)\s*=\s*\\?[\"']?([\w@. -]{1,40})", m.group(0), re.I)
                found.append(f"script input type={t.group(1).lower()}" + (f" {name.group(1)!r}" if name else ""))
    return _dedupe(found)


def _ancestors(el):
    for i, anc in enumerate(el.iterancestors()):
        if i >= MAX_ANCESTORS:
            break
        yield anc


def _ident(el) -> str:
    return " ".join(filter(None, (el.get("class"), el.get("id"), el.get("itemprop"), el.get("type"))))


def _in_area(el, pattern) -> bool:
    return any(pattern.search(_ident(x)) or (isinstance(x.tag, str) and pattern.search(x.tag))
               for x in [el, *_ancestors(el)])


def _is_hidden(el) -> bool:
    for x in [el, *_ancestors(el)]:
        style = _lower_attr(x, "style").replace(" ", "")
        if (x.get("hidden") is not None or _lower_attr(x, "aria-hidden") == "true" or "display:none" in style
                or "visibility:hidden" in style or HIDDEN_CLASS.search(x.get("class") or "")):
            return True
    return False


def _words(*parts) -> str:
    """Join alt / title / aria-label without repeating the same phrase ("Facebook Facebook")."""
    return " ".join(_dedupe(_clean(p) for p in parts))


def _has_logo_token(text: str) -> bool:
    """'logo' / 'brand' as a whole token of a class, id or file name: nav-logo, LogoLink, dhl-logo.svg,
    brand-name; not logon / logout / logoff."""
    split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "").lower()
    return any(LOGO_TOKEN.match(t) for t in re.split(r"[^a-z0-9]+", split) if t)


def _is_icon(words: str) -> bool:
    return bool(ICON_WORDS.search(re.sub(r"([a-z])([A-Z])", r"\1 \2", words)))   # also "ChevronRightIcon"


def _site_host(doc) -> str:
    """The page's own host, from its canonical / og:url link (HTML only)."""
    for el in doc.xpath("//link[@rel='canonical'] | //meta[@property='og:url']"):
        m = re.match(r"https?://([^/]+)", (el.get("href") or el.get("content") or "").strip(), re.I)
        if m:
            return m.group(1).lower().removeprefix("www.")
    return ""


def _brand_hints(doc) -> list:
    """Logo / brand evidence from the markup. Order: page metadata, favicons, logos in the header, other logos,
    then other image text. Cookie banners, hosting banners and tracking pixels are skipped."""
    meta, header, logos, other = [], [], [], []
    for m in doc.iter("meta"):
        key = _lower_attr(m, "property") or _lower_attr(m, "name")
        if key in ("og:site_name", "application-name", "apple-mobile-web-app-title") and m.get("content"):
            meta.append(f"meta {key}={_clean(m.get('content'))[:60]!r}")
    favicons = _dedupe((l.get("href") or l.get("src") or "").strip()[:80] for l in doc.iter("link")
                       if "icon" in _lower_attr(l, "rel"))
    meta += [f"favicon {f}" for f in favicons[:MAX_FAVICONS]]

    def add(el, hint, logo_like, *extra):
        if (_in_area(el, NOT_BRAND_AREA) or any(NOT_BRAND_AREA.search(x or "") for x in extra)
                or any(x.tag == "noscript" for x in _ancestors(el))):
            return
        if logo_like:
            (header if _in_area(el, HEADER_AREA) else logos).append(hint)
        else:
            other.append(hint)

    for svg in doc.iter("{http://www.w3.org/2000/svg}svg", "svg"):
        title = svg.find(".//title")
        parent = svg.getparent()
        words = _words(svg.get("aria-label"), title.text_content() if title is not None else "",
                       parent.get("aria-label") if parent is not None else "")
        ident = _ident(svg) + " " + (_ident(parent) if parent is not None else "")
        if _is_icon(ident):
            continue   # spinner / chevron icons, even inside a "logo" wrapper
        logo_like = _has_logo_token(ident) or bool(HOME_WORD.search(words))
        if not words and not _ident(svg):
            continue   # nothing to show: an unnamed svg inside a logo wrapper
        if logo_like or (words and not _is_icon(words)):
            add(svg, f"svg label={words[:60]!r} class={_ident(svg)[:60]!r}", logo_like)
    for img in doc.iter("img"):
        if img.get("width") in ("0", "1") or img.get("height") in ("0", "1"):
            continue   # tracking pixel
        src = (img.get("src") or img.get("data-src") or img.get("data-lazy-src") or img.get("data-image-path")
               or (img.get("srcset") or "").strip().split(" ")[0]).strip()
        file_name = "" if src.startswith("data:") else src.split("?")[0].rstrip("/").rsplit("/", 1)[-1]
        shown = "data:image (inline)" if src.startswith("data:") else src[:80]
        parent = img.getparent()
        words = _words(img.get("alt"), img.get("title"), parent.get("aria-label") if parent is not None else "")
        if re.fullmatch(r"[\w.-]+\.(png|jpe?g|gif|svg|webp|ico)", words, re.I):
            words = ""   # alt text that is just the file name ("1left.svg")
        ident = _ident(img) + " " + (_ident(parent) if parent is not None else "")
        logo_like = _has_logo_token(file_name) or _has_logo_token(ident) or bool(re.search(r"\blogo\b", words, re.I))
        if logo_like or (words and not _is_icon(words)):
            add(img, f"img alt={words[:60]!r} src={shown}", logo_like, src, words,
                parent.get("href") if parent is not None else "")
    # CSS-sprite and text logos: an element with a logo-like class/id that names the brand in aria-label, title,
    # alt or its own short text (<a class="nav-logo" aria-label="Amazon">, <p class="paypal-logo">PayPal</p>)
    for el in doc.xpath("//*[@class or @id or @type]"):
        if (not isinstance(el.tag, str) or el.tag in ("img", "svg", "html", "body", "nav", "ul", "ol", "header",
                                                      "footer", "form", "section", "label", "input")
                or not _has_logo_token(_ident(el))):
            continue
        if el.find(".//img") is not None or _is_icon(_ident(el)):
            continue
        # own short text only when the element is a small leaf (not a container like "eBay HomeShop by category")
        text = _clean(" ".join(el.itertext())) if len(el) <= 2 else ""
        label = _clean(el.get("aria-label") or el.get("title") or el.get("alt") or
                       (text if len(text) <= MAX_BARE_TEXT else ""))
        if re.search(r"[^\W\d_]{3}", label) and not _is_icon(label):
            add(el, f"{el.tag} logo label={label[:60]!r}", True, el.get("href"), label)
    # home links: a link to the root of this site, named by aria-label or short text (<a href="/">eBay</a>)
    host = _site_host(doc)
    for a in doc.iter("a"):
        href = (a.get("href") or "").strip()
        m = re.fullmatch(r"https?://([^/]+)/?", href, re.I)
        label = _clean(a.get("aria-label") or a.get("title") or _text(a) or
                       " ".join(i.get("alt", "") for i in a.iter("img")))
        link_host = m.group(1).lower().removeprefix("www.") if m else ""
        # an absolute root link counts when it is this page's own host, or when the label is exactly the site
        # name of that host (<a href="https://www.ebay.de">eBay</a>), not a section ("Business" -> business.att.com)
        site_name = link_host.split(".")[-2] if link_host.count(".") >= 1 else ""
        names_site = bool(site_name) and re.sub(r"\W", "", label.lower()) == site_name
        own_root = href in ("/", "./", "/home", "/index.html") or (link_host and (link_host == host or names_site))
        if not own_root or not label or len(label) > MAX_BARE_TEXT or label.lower() in NAV_WORDS:
            continue
        if not _is_icon(label) and not _link_kind(label) and re.search(r"[^\W\d_]{2}", label):
            add(a, f"home link label={_clean(ICON_LIGATURE.sub(' ', label))[:60]!r}", True, href)
    return _dedupe(meta + header + logos + other)   # only identical lines are merged


def _link_kind(label: str, target: str = "") -> str:
    if NOT_A_SIGNIN.search(label) or HOSTNAME.match(label):
        return ""
    if "," in label and " and " in label:
        return ""   # a help-centre category ("Login, Recovery and Security"), not a control
    if STRONG_LINK.search(label):
        return "strong" if len(label) <= SHORT_LABEL else "strong_long"
    if AMBIGUOUS_LINK.search(label):
        return "strong"
    if LOGOUT_LINK.search(label):
        return "logout"
    if (WEAK_LINK.search(label) and len(label) <= MAX_WEAK_LABEL and not (label.endswith("?") and len(label) > 20)
            and not NOT_ACCOUNT_ENTRY.search(label) and not (target.startswith("#") and len(target) > 1)):
        return "weak"
    return ""


def _links(doc) -> list:
    """Sign-in / account / log-out controls in page order, grouped: short sign-in labels first, then account
    links, then long sign-in phrases (usually help titles), then log-out."""
    ids = {el.get("id"): el for el in doc.xpath("//*[@id]")}
    groups = {"strong": [], "weak": [], "strong_long": [], "logout": []}
    # one union query keeps document order, so a header span "Sign in" is listed before body links
    for el in doc.xpath("//a | //button | //input[@type] | //*[@role='button' or @role='link' or @role='menuitem']"
                        " | //*[@onclick] | //span | //div | //li"):
        tag = el.tag if isinstance(el.tag, str) else ""
        clickable = tag in ("a", "button", "input") or el.get("role") in ("button", "link", "menuitem") \
            or el.get("onclick") is not None
        if tag == "input" and _lower_attr(el, "type") not in ("submit", "button", "image"):
            continue
        if not clickable:
            # a plain span/div/li counts only when its whole text is a short sign-in label, it is not inside a
            # heading or a link, and it is not hidden (popup status text, "Sign in to {0}" templates)
            own = _clean(el.text)
            if not own or len(own) > MAX_BARE_LABEL or not STRONG_LINK.search(own) or _text(el) != own:
                continue
            if "{" in own or own.endswith((".", ":")) or NOT_A_CONTROL.search(_ident(el)) or any(x.tag in ("h1", "h2", "h3", "h4", "h5", "h6", "a", "button") or
                                 x.get("role") == "heading" for x in _ancestors(el)) or _is_hidden(el):
                continue
        # parts joined with spaces: <span>Sign in</span><span>Opens overlay</span> -> "Sign in Opens overlay"
        text = _clean(ICON_LIGATURE.sub(" ", " ".join(el.itertext())))
        if len(text) > MAX_CLICKABLE_TEXT:
            text = ""   # a container with a click handler: rely on its aria-label / title instead
        labelled_by = " ".join(_text(ids[i]) for i in (el.get("aria-labelledby") or "").split() if i in ids)
        alts = " ".join(i.get("alt", "") for i in el.iter("img"))
        names = [n for n in (_clean(s)[:60] for s in (text, el.get("value"), el.get("aria-label"), el.get("title"),
                                                      labelled_by, alts)) if n]
        if not names:
            continue
        # the visible text is the label; an aria-label/title is added only when the text alone is not sign-in
        # wording (icon-only buttons), so footer links don't become "X / Acessar a página X"
        target = (el.get("href") or el.get("onclick") or el.get("formaction") or "").strip()
        label = names[0]
        if not _link_kind(label, target):
            extra = next((n for n in names[1:] if _link_kind(n, target)), "")
            label = f"{label} / {extra}" if extra and label != extra else (extra or label)
        kind = _link_kind(label, target)
        if not kind:
            continue
        groups[kind].append(f"{tag}: {label} -> {target[:80]}")
    return groups["strong"] + groups["weak"] + groups["strong_long"] + groups["logout"]


def _drop_noise(doc):
    """Remove text that is never shown or is not the page: inline display:none / visibility:hidden elements
    (random strings kits hide between letters), cookie-consent banners and long <option> lists. Anything that
    contains a form field is kept, because a credential box hidden behind a modal still counts as 3.1."""
    def has_field(el):
        return el.tag in ("input", "select", "textarea", "form") or bool(el.xpath(".//input|.//select|.//textarea"))

    for el in doc.xpath("//*[@style or @hidden]"):
        style = _lower_attr(el, "style").replace(" ", "")
        hidden = el.get("hidden") is not None or "display:none" in style or "visibility:hidden" in style
        if hidden and el.getparent() is not None and el.tag not in ("html", "body") and not has_field(el):
            el.drop_tree()
    body_chars = len(_text(doc.find("body"))) if doc.find("body") is not None else 0
    for el in doc.xpath("//*[@id or @class]"):
        if el.getparent() is None or el.tag in ("html", "body", "main", "form") or not NOT_BRAND_AREA.search(
                (el.get("id") or "") + " " + (el.get("class") or "")):
            continue
        if not has_field(el) or not el.xpath(".//input[@type='password']"):
            if len(_text(el)) < 0.5 * max(body_chars, 1):   # never the page itself ("has-cookie-banner" on a wrapper)
                el.drop_tree()
    for sel in doc.iter("select"):
        options = sel.findall(".//option")
        if len(options) > MAX_SELECT_OPTIONS:
            for opt in options[1:]:
                opt.drop_tree()


def parse(raw: bytes) -> dict:
    doc = _parse_doc(raw)
    html_lang = (doc.get("lang") or doc.get("xml:lang") or "").strip().lower()

    script_inputs = _script_inputs(doc)
    script_urls = _script_urls(doc)
    brand_hints = _brand_hints(doc)   # before svg removal: svg <title>/aria-label can name the brand
    for el in doc.xpath("//script | //style | //noscript | //svg | //template[not(@shadowrootmode) and "
                        "not(@shadowroot)] | //comment() | //processing-instruction()"):
        if el.getparent() is not None:   # comments before <html> have no parent and cannot be dropped
            el.drop_tree()
    _drop_noise(doc)
    for el in doc.iter(*SPACED_TAGS):
        el.tail = " " + (el.tail or "")
        el.text = " " + (el.text or "")

    title = _clean(doc.findtext(".//title", ""))[:MAX_TITLE]
    # h1 first, so navigation h2/h3 menus do not push the page's main heading out of the cap
    heads = doc.xpath("//h1") + doc.xpath("//*[@role='heading']") + doc.xpath("//h2") + doc.xpath("//h3")
    headings = _dedupe(t[:80] for t in (_text(h) for h in heads))

    forms = [f"action={f.get('action', '')!r} method={(f.get('method') or 'get').strip().lower()}"
             for f in doc.iter("form")]

    labels_by_id = {}
    for l in doc.iter("label"):
        if l.get("for") and _text(l):
            labels_by_id.setdefault(l.get("for"), _text(l)[:60])   # first label wins on duplicate ids
    inputs = []
    for el in doc.xpath("//input|//select|//textarea"):
        kind = _lower_attr(el, "type") or ("text" if el.tag == "input" else el.tag)
        if kind == "hidden":
            continue
        desc = [f"{a}={el.get(a)[:40]!r}" for a in INPUT_ATTRS if el.get(a)]
        if labels_by_id.get(el.get("id")):
            desc.append(f"label={labels_by_id[el.get('id')]!r}")
        if kind in TEXT_INPUT_TYPES and el.get("value"):
            desc.append("prefilled=yes")   # e.g. the victim's email filled in by the kit; value itself not kept
        inputs.append(f"{el.tag} type={kind} " + " ".join(desc))

    links = _links(doc)
    iframes = _dedupe((f.get("src") or f.get("data-src") or "")[:120] for f in doc.xpath("//iframe|//frame"))
    n_iframes = len(doc.xpath("//iframe|//frame"))
    body = doc.find("body")
    visible = _clean(ICON_LIGATURE.sub(" ", (body if body is not None else doc).text_content()))

    return {
        "html_lang": html_lang,
        "title": title,
        "headings": headings,
        "forms": forms,
        "inputs": inputs,
        "script_inputs": script_inputs,
        "script_urls": script_urls,
        "links": links,
        "brand_hints": brand_hints,
        "iframes": iframes,
        "visible_text": visible,
        "n_forms": len(forms), "n_inputs": len(inputs), "n_links": len(links), "n_iframes": n_iframes,
        "visible_chars": len(visible),
    }


EMPTY = parse(b"<html></html>")
