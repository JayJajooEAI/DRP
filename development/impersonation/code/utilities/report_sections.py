"""Extra HTML sections for the pipeline page: (1) page-by-page enrichment of Step 1 with OCR boxes, parsed fields, avatar matching and the rule points it leads to;
(2) runtime and model size. Returns (css, enrich_html, runtime_html)."""

import base64, io, json, re, html as H
import numpy as np, pandas as pd
from PIL import Image, ImageDraw
from utilities.config import *
from utilities.layout_parser import parse, LAYOUT, STATS, DOMAIN, norm_dom
from utilities.rules import rule_scores, to01

# Where the avatar sits on each layout (same boxes as step 02)
BOX = {"feed": (80, 172, 192, 284), "grid": (80, 88, 206, 214), "pro": (80, 164, 208, 292)}

# Colour used to outline OCR text by the field the parser gave it
COL = {
    "url bar": "#8a8f98",
    "banner": "#d9822b",
    "name": "#c0392b",
    "handle": "#7a3fa0",
    "bio": "#1f7a3f",
    "link": "#0d6b63",
    "stats": "#2c5fa8",
    "post": "#b5a000",
    "other": "#aaaaaa",
}


# Image -> base64 jpeg string (so the image can be embedded in the HTML)
def b64(im, q=72):
    o = io.BytesIO()
    im.convert("RGB").save(o, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(o.getvalue()).decode()


# Escape text for safe use inside HTML
def e(s):
    return H.escape(str(s))


# Lowercase letters+digits only (for comparing texts)
def nz(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


# Choose the 6 validation pages shown in Step 00, one per platform (X, Instagram, Facebook, Telegram, TikTok, LinkedIn)
def pick_pages():
    st = pd.read_csv(f"{WORK}/dev_val_stage.csv")
    want = [
        ("X", "P1_brand_support_clone"),
        ("Instagram", "P3_lookalike_handle"),
        ("Facebook", "N1_official_client"),
        ("Telegram", "P2_fake_official_page"),
        ("TikTok", "HN3_critic_boycott"),
        ("LinkedIn", "P4_exec_full"),
    ]
    out = []
    for pl, c in want:
        d = st[(st.platform == pl) & (st.concept == c)]
        if not len(d):
            d = st[st.platform == pl]
        if len(d):
            r = d.iloc[0]
            out.append(
                (r.id, f"{pl}: " + ("impersonation" if int(r.label) == 1 else "not impersonation"), r.concept)
            )
    return out


# classify() = which field did the parser give this OCR line to? (url bar / post / handle / stats / name / link / banner / bio / other). Used to colour the boxes.
def classify(l, p, post_y):
    t = l["t"].strip()
    tl = t.lower()
    if l["y"] < 32:
        return "url bar"
    if post_y is not None and l["y"] >= post_y:
        return "post"
    if t.startswith("@"):
        return "handle"
    if STATS.search(t) or re.search(r"(joined|since)\s*\d{4}", tl):
        return "stats"
    if p["name"] and nz(t) == nz(p["name"]):
        return "name"
    if p["link"] and nz(p["link"]) in nz(t) and len(t) < 50:
        return "link"
    if p["layout"] != "grid" and 74 <= l["y"] <= 224 and l["x"] > 200:
        return "banner"
    if p["bio"] and nz(t) and nz(t) in nz(p["bio"]):
        return "bio"
    return "other"


# Human-readable layout names
LAYNAME = {
    "feed": "feed style (X, Facebook, Telegram)",
    "grid": "grid style (Instagram, TikTok)",
    "pro": "professional style (LinkedIn)",
}


# page_block() = the HTML for ONE page: screenshot with coloured OCR boxes, parsed fields, avatar match, signals and rule points
def page_block(pid, title, concept, S, R, A, O, meta):
    r = meta[pid]
    C = CLIENT[r["customer"]]
    lines = O[pid]
    p = parse(lines)
    lay = p["layout"]
    box = BOX[LAYOUT.get(p["platform"], "feed")]
    img = Image.open(f"{SHOT}/{pid}.png").convert("RGB")
    d = ImageDraw.Draw(img)
    post_y = next(
        (
            l["y"]
            for l in lines
            if p["posts"]
            and l["y"] > 90
            and re.match(r"^.*?@\s*[\w.\-]+", l["t"])
            and re.search(r"\d+\s*[smhdw]\s*$", l["t"].strip())
        ),
        None,
    )
    legend = {}
    for l in lines:
        k = classify(l, p, post_y)
        legend[k] = legend.get(k, 0) + 1
        d.rectangle((l["x"] - 1, l["y"] - 1, l["x1"] + 1, l["y"] + l["h"] + 1), outline=COL[k], width=2)
    d.rectangle(box, outline="#e0115f", width=3)
    ocr_img = b64(img.resize((560, 560)), 70)
    crop = Image.open(f"{SHOT}/{pid}.png").convert("RGB").crop(box).resize((112, 112))
    logo = Image.open(C["logo"]).convert("RGBA")
    bg = Image.new("RGB", logo.size, "white")
    bg.paste(logo, mask=logo.split()[3])
    logo = bg.resize((112, 112))
    execs = [Image.open(x["photo"]).convert("RGB").resize((112, 112)) for x in C["execs"] if x["photo"]][:2]
    a = A[pid]
    s = S.loc[pid]

    def mini(imgx, cap, sub=""):
        return f'<figure class="mf"><img src="{b64(imgx, 80)}" alt="{e(cap)}"><figcaption><b>{e(cap)}</b>{("<br>" + e(sub)) if sub else ""}</figcaption></figure>'

    avh = mini(crop, "Avatar crop", f"{lay} box {box}") + mini(
        logo, "Client logo", f"clip {a['logo_clip']:.2f} · pixel {a['logo_ncc']:.2f}"
    )
    for i, x in enumerate(execs):
        avh += (
            mini(
                x,
                f"Executive {i + 1}",
                f"clip {a['exec_clip']:.2f} · pixel {a['exec_ncc']:.2f} (best of all)",
            )
            if i == 0
            else mini(x, f"Executive {i + 1}")
        )
    fields = [
        ["platform (from URL bar)", p["platform"]],
        ["layout (how the page is arranged)", LAYNAME[lay]],
        ["display name", p["name"]],
        ["handle", p["handle"]],
        ["bio", p["bio"][:90]],
        ["link", p["link"]],
        [
            "connections" if lay == "pro" else "followers / following / posts",
            (
                f"{int(p['connections'] or 0):,}"
                if lay == "pro"
                else f"{int(p['followers'] or 0):,} / {int(p['following'] or 0):,} / {int(p['posts_n'] or 0):,}"
            ),
        ],
        ["joined", p["joined"]],
        ["banner text", p["banner_text"]],
        ["posts read", len(p["posts"])],
        ["verified badge", "yes" if s.get("verified", 0) else "no"],
    ]
    frows = "".join(f"<tr><td>{e(k)}</td><td><code>{e(v)}</code></td></tr>" for k, v in fields)
    sig = [
        ("name vs brand (0–100)", f"{s.get('name_brand', 0):.0f}"),
        ("handle contains brand", "yes" if s.get("handle_has_brand", 0) else "no"),
        ("avatar = client logo", "yes" if s.get("av_logo_strong", 0) else "no"),
        ("avatar = executive photo", "yes" if s.get("av_exec_strong", 0) else "no"),
        ("default grey avatar", "yes" if s.get("av_default", 0) else "no"),
        ("link is official", "yes" if s.get("link_official", 0) else "no"),
        ("scam wording", "yes" if s.get("scam_any", 0) else "no"),
        ("fan / parody wording", "yes" if s.get("disc_any", 0) else "no"),
    ]
    srow = "".join(
        f'<span class="sg{" on" if v == "yes" else ""}"><b>{e(k)}</b> {e(v)}</span>' for k, v in sig
    )
    comp = R.loc[pid]
    comp = comp[comp != 0]
    crow = (
        "".join(
            f'<tr><td>{e(k)}</td><td class="{"pt+" if v > 0 else "pt-"}">{v:+.1f}</td></tr>'
            for k, v in comp.items()
        )
        or "<tr><td colspan=2>no rule fired</td></tr>"
    )
    raw = float(R.loc[pid].sum())
    sc = float(to01(pd.Series([raw]))[0])
    lg = (
        "".join(
            f'<span class="lg"><i style="background:{COL[k]}"></i>{k} ({n})</span>' for k, n in legend.items()
        )
        + '<span class="lg"><i style="background:#e0115f"></i>avatar box</span>'
    )
    truth = "Impersonation" if int(r["label"]) == 1 else "Not impersonation"
    return f"""<article class="ex"><h4>{e(title)} <code>{e(concept)}</code> <span class="truth {'pos' if int(r['label']) else 'neg'}">truth: {truth}</span></h4>
<div class="exg"><div class="c1"><div class="cap">① docTR OCR: every box is a text line the parser then labels by colour</div><img class="ocr" src="{ocr_img}" alt="Screenshot with OCR boxes"><div class="lgs">{lg}</div></div>
<div class="c2"><div class="cap">② Parser: fields recovered from the lines</div><table class="kv">{frows}</table>
<div class="cap" style="margin-top:12px">③ Avatar matching (client: {e(C['brand'])})</div><div class="mfs">{avh}</div><div class="cap sm">Default-avatar check: pixel match to a generic grey avatar = {a['sil_ncc']:.2f} (flag above 0.70)</div></div>
<div class="c3"><div class="cap">④ Signals built from ①–③ plus the client facts</div><div class="sgs">{srow}</div>
<div class="cap" style="margin-top:12px">⑤ Step 2 rule points these signals earn</div><table class="kv pts">{crow}<tr class="tot"><td>sum of points</td><td>{raw:+.1f}</td></tr><tr class="tot"><td>score 0–1</td><td>{sc:.3f}</td></tr></table>
<div class="cap sm">Impersonation if ≥ {json.load(open(f"{REPORTS}/frozen_thresholds.json"))['rule_hi']}, not if ≤ {json.load(open(f"{REPORTS}/frozen_thresholds.json"))['rule_lo']}, otherwise unclear → Step 3.</div></div></div></article>"""


# Extra CSS for these cards
CSS = """
.xin{width:100%}
.meth{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:14px 0 20px}
.meth div{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--bg)}
.meth b{display:block;margin-bottom:4px}.meth small{color:var(--muted);display:block;margin-top:6px}
.ex{border-top:1px solid var(--line);padding-top:16px;margin-top:18px}.ex h4{margin:0 0 10px;font-size:16px}
.exg{display:grid;grid-template-columns:580px 1fr 1fr;gap:18px;align-items:start}
@media(max-width:1250px){.exg{grid-template-columns:1fr}.meth{grid-template-columns:1fr 1fr}}
.ex .cap{font-weight:700;font-size:13px;margin-bottom:6px}.ex .cap.sm{font-weight:600;color:var(--muted);margin:8px 0 4px}
img.ocr{width:100%;max-width:560px;border:1px solid var(--line);border-radius:8px}
.lgs{display:flex;flex-wrap:wrap;gap:4px 12px;margin-top:6px;font-size:12px;color:var(--muted)}.lg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:4px;vertical-align:-1px}
table.kv{border-collapse:collapse;width:100%;font-size:13px}table.kv td{padding:4px 8px;border-bottom:1px solid var(--line);vertical-align:top}table.kv td:first-child{color:var(--muted);width:42%}
table.kv code{font-size:12px;word-break:break-word}.pt\\+{color:var(--bad);font-weight:700;text-align:right}.pt-{color:var(--good);font-weight:700;text-align:right}table.pts td:last-child{text-align:right}table.pts tr.tot td{font-weight:700;border-top:2px solid var(--line)}
.mfs{display:flex;gap:10px;flex-wrap:wrap}.mf{margin:0;width:112px;font-size:11px}.mf img{width:112px;height:112px;border-radius:50%;object-fit:cover;border:1px solid var(--line);display:block}.mf figcaption{margin-top:4px;color:var(--muted)}.mf figcaption b{color:var(--ink)}
.zbs{display:grid;gap:4px}.zb{position:relative;display:block;height:18px;background:var(--bg);border:1px solid var(--line);border-radius:4px;overflow:hidden}.zb i{position:absolute;left:0;top:0;bottom:0;background:var(--accent-soft)}.zb.top i{background:var(--accent);opacity:.55}.zb em{position:relative;font-style:normal;font-size:11px;padding-left:6px;line-height:18px}
.sgs{display:flex;flex-wrap:wrap;gap:6px}.sg{border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:12px;background:var(--bg)}.sg.on{background:var(--warn-soft);border-color:var(--warn)}
.truth{font-size:12px;border-radius:999px;padding:2px 9px;margin-left:8px}.truth.pos{background:var(--bad-soft);color:var(--bad)}.truth.neg{background:var(--good-soft);color:var(--good)}
table.rt{border-collapse:collapse;width:100%;font-size:13.5px;margin:10px 0}table.rt th,table.rt td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;vertical-align:top}table.rt th{background:var(--bg);font-size:12px;color:var(--muted)}table.rt td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bars{display:grid;gap:6px;margin:10px 0 4px}.bar{display:grid;grid-template-columns:230px 1fr 130px;gap:10px;align-items:center;font-size:13px}.bar div{height:16px;background:var(--bg);border:1px solid var(--line);border-radius:4px;position:relative}.bar div i{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:3px}
.fin td.okc{background:var(--good-soft);color:var(--good)}.fin td.bad{background:var(--bad-soft);color:var(--bad)}.fin td{font-size:20px;padding:10px}.tri th{text-align:center}.tri td{text-align:center;font-weight:700}.tri td.pass{background:var(--warn-soft);color:var(--warn)}.note{font-size:13px;color:var(--muted);margin-top:8px}
"""
# Number of ML features (set by build(), shown in the runtime table)
NFEAT = 0


# build() = returns (css, step-00 html, runtime html) using the saved outputs
def build(nfeat=0):
    # Remember the feature count for the runtime table
    global NFEAT
    NFEAT = nfeat

    # meta = csv row of every page, by id
    meta = {}
    for sp in ("train", "val", "test"):
        meta.update({r["id"]: r for r in load_split(sp)})

    # Load the validation signals, the rule points per page, the avatar scores and the OCR lines
    S = pd.read_csv(f"{WORK}/signals_val.csv").set_index("id")
    Sv = pd.read_csv(f"{WORK}/signals_val.csv")
    Rv = rule_scores(Sv)
    Rv.index = Sv.id
    A = json.load(open(f"{WORK}/avatar_val.json"))
    O = json.load(open(f"{WORK}/ocr_val.json"))

    # One block (screenshot + parsed fields + avatar + signals + rule points) per chosen page
    blocks = "".join(page_block(pid, t, c, S, Rv, A, O, meta) for pid, t, c in pick_pages())

    # enrich = the Step 00 html: the four-move explainer followed by the page blocks
    enrich = f"""<section class="xsec" id="enrich"><h2>How a screenshot becomes data (Step 1, page by page)</h2>
<p class="lead">The pipeline gets one picture. Before any rule runs, it is turned into structured facts in four moves. Nothing here decides anything; each move only adds information to the page. Below, six real validation pages, one per platform (X, Instagram, Facebook, Telegram, TikTok, LinkedIn), go through all four, ending in the rule points they earn.</p>
<div class="meth"><div><b>① OCR</b>docTR finds every text line in the picture and reads it, with its position and size.<small>db_resnet50 + crnn_vgg16_bn · 41M params</small></div>
<div><b>② Layout parser</b>Plain rules on positions: URL bar → platform; text in the banner band → banner; first line under the avatar → name; lines with @ → handle; "followers" lines → stats; post headers → posts.<small>no model · 0.1 ms per page</small></div>
<div><b>③ Avatar matching</b>Crop the avatar from its fixed box; compare with the client's logo and executive photos (CLIP + pixel match).<small>CLIP ViT-B/32 · 151M params</small></div>
<div><b>④ Signals</b>Fields + client facts → about 100 numbers: name vs brand, look-alike link, scam wording, verified badge, account age and counts, avatar flags.<small>no model · 17 ms per page</small></div></div>
{blocks}
<p class="note">Colours on the screenshot show which field the parser gave each OCR line to. Pink = the fixed avatar box. These are validation pages (never test).</p></section>"""

    # Load the measured runtime numbers (from step 09) and the frozen thresholds; count the pages that went to Luna on test
    R = json.load(open(f"{REPORTS}/runtime.json"))  # written by 09_measure_runtime.py
    thr = json.load(open(f"{REPORTS}/frozen_thresholds.json"))
    _ts = pd.read_csv(f"{WORK}/test_stage.csv")
    n_llm = int((_ts.stage == "llm").sum())
    n_unsure = n_llm / len(_ts)

    # Luna latency text (min-max seconds) if it was measured
    lu = R.get("luna_s_per_call")
    lu_txt = f"{min(lu):.1f}–{max(lu):.1f} s" if lu else "not measured"

    # rows = one line per pipeline step for the runtime table: what runs, parameters, size on disk, time per page on GPU / CPU, and which pages it runs on
    rows = [
        (
            "1 · docTR OCR",
            "db_resnet50 (detector, 25.4M) + crnn_vgg16_bn (reader, 15.8M)",
            "41.2M",
            "≈ 165 MB",
            f"{R['ocr_s_per_page_gpu'] * 1000:.0f} ms",
            f"{R['ocr_s_per_page_cpu']:.1f} s",
            "every page; the slowest step",
        ),
        (
            "2 · Layout parser",
            "regular expressions and position rules",
            "none",
            "a few KB of code",
            f"{R['parse_ms_per_page']:.1f} ms",
            f"{R['parse_ms_per_page']:.1f} ms",
            "every page",
        ),
        (
            "3 · Avatar matching",
            "CLIP ViT-B/32 (only the image tower, 87.8M, is used)",
            "87.8M used (151.3M loaded)",
            "≈ 578 MB (image tower alone ≈ 350 MB)",
            f"{R['clip_ms_per_page_gpu']:.1f} ms",
            f"{R['clip_ms_per_page_cpu']:.0f} ms",
            "every page; one 224×224 crop",
        ),
        (
            "4 · Signals",
            "fuzzy matching, lexicons, badge detector (OpenCV)",
            "none",
            "a few KB of code",
            f"{R['signals_ms_per_page']:.0f} ms",
            f"{R['signals_ms_per_page']:.0f} ms",
            "every page",
        ),
        (
            "5 · Rules (score 0–1)",
            "points added up and rescaled",
            "none",
            "a few KB of code",
            "< 0.1 ms",
            "< 0.1 ms",
            "every page",
        ),
        (
            "6 · LightGBM",
            "300 trees, 15 leaves, " + str(NFEAT) + " features",
            "no weights; 300 trees",
            "≈ 0.5 MB (frozen file)",
            f"{R['lgbm_ms_per_page']:.3f} ms",
            f"{R['lgbm_ms_per_page']:.3f} ms",
            "pages the rules left unclear (≈ 86% of test)",
        ),
        (
            "7 · Luna (Decisions API)",
            "gpt-6-luna, one choice question, screenshot + logo",
            "hosted by OpenAI",
            "nothing local",
            lu_txt,
            lu_txt + " (network)",
            f"only the unsure pages ({n_llm} of {len(_ts)} on test = {100 * n_unsure:.1f}%); ≈ $0.0001 each",
        ),
    ]

    # Turn the rows into html table rows
    tr = "".join(
        f"<tr><td><b>{a}</b></td><td>{b}</td><td class='n'>{c}</td><td class='n'>{d}</td><td class='n'>{e_}</td><td class='n'>{f}</td><td>{g}</td></tr>"
        for a, b, c, d, e_, f, g in rows
    )

    # Total time per page for all local steps on GPU, and on CPU only
    tot_gpu = (
        R["ocr_s_per_page_gpu"] * 1000
        + R["parse_ms_per_page"]
        + R["clip_ms_per_page_gpu"]
        + R["signals_ms_per_page"]
        + 0.1
        + R["lgbm_ms_per_page"]
    )
    tot_cpu = (
        R["ocr_s_per_page_cpu"] * 1000
        + R["parse_ms_per_page"]
        + R["clip_ms_per_page_cpu"]
        + R["signals_ms_per_page"]
        + 0.1
        + R["lgbm_ms_per_page"]
    )

    # mx = the longest step (the bars are drawn relative to it)
    mx = max(R["ocr_s_per_page_gpu"] * 1000, 1)

    # bars = (step, ms per page on GPU) for the little bar chart
    bars = [
        ("OCR (docTR)", R["ocr_s_per_page_gpu"] * 1000),
        ("Signals", R["signals_ms_per_page"]),
        ("CLIP avatar match", R["clip_ms_per_page_gpu"]),
        ("Layout parser", R["parse_ms_per_page"]),
        ("LightGBM", R["lgbm_ms_per_page"]),
    ]

    # Turn the bars into html
    bh = "".join(
        f'<div class="bar"><span>{n}</span><div><i style="width:{max(1.5, 100 * v / mx):.1f}%"></i></div><span>{v:.2f} ms</span></div>'
        for n, v in bars
    )

    # runtime = the Runtime card html: table, bars and notes
    runtime = f"""<section class="xsec" id="runtime"><h2>Runtime and model size</h2>
<p class="lead">Measured on this machine (Tesla T4 GPU, 8 CPU cores), per page, with the models already loaded. The local models together use about 130M parameters. Nothing but Luna leaves the machine.</p>
<table class="rt"><thead><tr><th>Step</th><th>What runs</th><th>Parameters</th><th>Size on disk</th><th>Per page, GPU</th><th>Per page, CPU only</th><th>Runs on</th></tr></thead><tbody>{tr}</tbody></table>
<div class="bars">{bh}</div>
<p class="note"><b>Per page, all local steps:</b> about {tot_gpu:.0f} ms on the GPU (≈ {1000 / tot_gpu:.1f} pages a second), about {tot_cpu / 1000:.1f} s on CPU only. OCR is {100 * R['ocr_s_per_page_gpu'] * 1000 / tot_gpu:.0f}% of it on the GPU and {100 * R['ocr_s_per_page_cpu'] * 1000 / tot_cpu:.0f}% on the CPU; CLIP is only {R['clip_ms_per_page_gpu']:.1f} ms on the GPU and {R['clip_ms_per_page_cpu']:.0f} ms on a CPU. Luna is added for the few unsure pages.</p>
<p class="note">GPU memory: about {R['gpu_mem_ocr_MB'] / 1024:.1f} GB peak for OCR in batches of 16, plus about 0.6 GB for CLIP. Model load: OCR {R['ocr_load_s']:.1f} s, CLIP {R['clip_load_s']:.1f} s, once per run. Timings use 80–96 validation pages after one warm-up batch; the pixel-level image reading in "Signals" runs once per page and is the same on GPU and CPU.</p></section>"""
    return CSS, enrich, runtime
