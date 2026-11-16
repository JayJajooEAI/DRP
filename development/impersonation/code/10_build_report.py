"""Build outputs/reports/impersonation_pipeline.html: the step-by-step pipeline page from the real run outputs (needs 05-08 and 09 run first).
Usage: python 10_build_report.py"""

# Libraries: base64/io/PIL embed screenshots in the page; json/re/pandas/numpy handle the data; joblib loads the frozen model
import base64, io, json, re, numpy as np, pandas as pd, joblib
from PIL import Image

# Project paths and client data
from utilities.config import *

# metrics() = confusion matrix + precision / recall / accuracy
from utilities.cascade import metrics

# The HTML template (sliding-card page design) and the output file path
TEMPLATE = str(__import__("pathlib").Path(__file__).resolve().parent / "utilities" / "report_template.html")
OUT_HTML = f"{REPORTS}/impersonation_pipeline.html"

# F = the frozen model + thresholds from step 05; ART = just the thresholds and feature list
F = joblib.load(f"{WORK}/frozen.joblib")
ART = F["art"]


# Short helper: read a json file
def lj(p):
    return json.load(open(p))


# ST[split] = the per-page stage table (which stage answered each page) for train, val and test
ST = {
    sp: pd.read_csv(f"{WORK}/{f}")
    for sp, f in (("train", "dev_train_stage.csv"), ("val", "dev_val_stage.csv"), ("test", "test_stage.csv"))
}

# LU[split] = Luna's saved answers for each split
LU = {sp: lj(f"{WORK}/luna_{sp}.json") for sp in ST}

# Add the columns the report needs, per split:
#  luna / luna_p = Luna's answer and its probability (-1 / NaN if the page did not go to Luna)
#  ml_guess = what the ML model alone would say (p >= 0.5)
#  p_rules / p_ml_step = the answer after the rules / after rules+ML
#  p_final = the answer of the whole pipeline (Luna's answer for unsure pages, ML guess if Luna failed)
for sp, d in ST.items():
    d["luna"] = d.id.map(
        lambda i: (int(LU[sp][i]["choice"] == "impersonation") if i in LU[sp] and LU[sp][i]["choice"] else -1)
    )
    d["luna_p"] = d.id.map(lambda i: LU[sp][i]["p_imp"] if i in LU[sp] else np.nan)
    d["ml_guess"] = (d.p_ml >= 0.5).astype(int)
    d["p_rules"] = np.where(
        d.stage == "rules", d.pred_pre_llm, 0
    )  # rules decided; unclear -> "not impersonation" (nothing else switched on yet)
    d["p_ml_step"] = np.where(
        d.stage == "llm", d.ml_guess, d.pred_pre_llm
    )  # + ML; pages still unsure take the ML guess
    d["p_final"] = np.where(d.stage == "llm", np.where(d.luna >= 0, d.luna, d.ml_guess), d.pred_pre_llm)
    d["label"] = d.label.astype(int)


# sc() = confusion matrix [[tp, fn], [fp, tn]] plus precision / recall / F1 for both classes, from true labels y and predictions p
def sc(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    M = metrics(y, p)
    tp, fp, fn, tn = M["tp"], M["fp"], M["fn"], M["tn"]

    def prf(a, b, c):
        pr = a / max(1, a + b)
        rc = a / max(1, a + c)
        return [round(pr, 4), round(rc, 4), round(2 * pr * rc / max(1e-9, pr + rc), 4)]

    pos = prf(tp, fp, fn)
    neg = prf(tn, fn, fp)
    return dict(
        m=[[tp, fn], [fp, tn]],
        p=pos[0],
        r=pos[1],
        f1=pos[2],
        pc=[pos, neg],
        acc=M["accuracy"],
        accuracy=M["accuracy"],
    )


# SPLITS = the numbers behind each step's confusion matrix, per split (step0 = baseline of 'always not impersonation', luna = only Luna's pages, final = the whole pipeline)
SPLITS = {}
SPLITS["step0"] = {sp: sc(d.label, np.zeros(len(d), int)) for sp, d in ST.items()}
SPLITS["luna"] = {
    sp: sc(d[d.stage == "llm"].label, d[d.stage == "llm"].p_final) for sp, d in ST.items()
}  # only the pages Luna saw
SPLITS["final"] = {sp: sc(d.label, d.p_final) for sp, d in ST.items()}


# tri_counts() = for one step: of the real impersonations and of the real harmless pages, how many it said yes / said no / left unclear (three-way table)
def tri_counts(d, outcome):
    o = {}
    for nm, lab in (("pos", 1), ("neg", 0)):
        g = d[d.label == lab]
        oc = outcome(g)
        o[nm] = [int((oc == 1).sum()), int((oc == 0).sum()), int((oc == -1).sum())]
    n = len(d)
    dec = o["pos"][0] + o["pos"][1] + o["neg"][0] + o["neg"][1]
    right = o["pos"][0] + o["neg"][1]
    o.update(n=n, decided=dec, right=right)
    return o


# final_block() = the final confusion matrix, headline scores and 'how many pages each stage answered and got right'
def final_block(d):
    m = metrics(d.label.values, d.p_final.values)
    by = {}
    for st in ("rules", "ml", "llm"):
        g = d[d.stage == st]
        by[st] = [int(len(g)), int((g.p_final == g.label).sum())]
    return dict(
        m=[[m["tp"], m["fn"]], [m["fp"], m["tn"]]],
        n=int(len(d)),
        acc=m["accuracy"],
        p=m["precision"],
        r=m["recall"],
        f1=m["f1"],
        by=by,
    )


# Final-result blocks per split
FINAL = {sp: final_block(d) for sp, d in ST.items()}


# What the rules stage output per page (-1 = left unclear)
def _rules_out(g):
    return np.where(g.stage == "rules", g.pred_pre_llm, -1)


# What the ML stage output per page (-1 = left unclear)
def _ml_out(g):
    return np.where(g.stage == "ml", g.pred_pre_llm, -1)


# Three-way tables for Step 2 (rules, all pages) and Step 3 (ML, only pages the rules left unclear)
TRI = {
    "step2": {sp: tri_counts(d, _rules_out) for sp, d in ST.items()},
    "step3": {sp: tri_counts(d[d.stage != "rules"], _ml_out) for sp, d in ST.items()},
}

# dev = train + val together (the pages the pipeline was developed on); n_dev = page count; n_pos = how many are impersonations
dev = pd.concat([ST["train"], ST["val"]])
n_dev = len(dev)
n_pos = int(dev.label.sum())


# Count rows where column c equals v
def cnt(df, c, v):
    return int(((df[c] == v)).sum())


# ---- stage counts (train + val)
# Stage counts on train + val, used in the text of each step card: pages the rules decided (and how many yes / no / wrong) ...
r_dec = dev[dev.stage == "rules"]
r_pos = int((r_dec.pred_pre_llm == 1).sum())
r_neg = int((r_dec.pred_pre_llm == 0).sum())
r_wrong = int((r_dec.pred_pre_llm != r_dec.label).sum())

# ... pages the rules left unclear, and what the ML model decided among them ...
unc = dev[dev.stage != "rules"]
m_dec = dev[dev.stage == "ml"]
m_wrong = int((m_dec.pred_pre_llm != m_dec.label).sum())
m_pos = int((m_dec.pred_pre_llm == 1).sum())
m_neg = len(m_dec) - m_pos

# ... pages that reached Luna, and how often Luna vs the ML guess was right
l_df = dev[dev.stage == "llm"]
l_pos = int(l_df.label.sum())
l_right = int((l_df.p_final == l_df.label).sum())
ml_right = int((l_df.ml_guess == l_df.label).sum())

# Format a number with thousands separators
fmtn = lambda x: f"{int(x):,}"

# ---- feature table (top features by importance, with plain-language description)
# DESC = plain-language description of the main features: name -> (group, what it is, how it is computed)
DESC = {
    "rule_score": (
        "Rules",
        "Total score of the Step 2 rules",
        "sum of the rule points for and against impersonation",
    ),
    "age_months": ("Account", "Account age in months", "read from 'Joined' / 'Member since' on the page"),
    "posts_n": ("Account", "Number of posts", "read from the stats line"),
    "followers": ("Account", "Followers (connections on LinkedIn)", "read from the stats line"),
    "following": ("Account", "Accounts followed", "read from the stats line"),
    "log_followers": ("Account", "Followers, log scale", "log(1 + followers)"),
    "foll_ratio": ("Account", "Followers per account followed", "followers / (following + 1)"),
    "p_scam": (
        "Wording",
        "Scam words in the posts",
        "count of giveaway / crypto / verify-your-account / refund / claim words in the post text",
    ),
    "scam_any": ("Wording", "Scam words anywhere", "1 if the bio, name or posts use scam wording"),
    "h_disc": (
        "Wording",
        "Fan / parody / review wording in name or bio",
        "count of fan, parody, satire, review, news, critic, not-affiliated words",
    ),
    "h_claim": (
        "Wording",
        "Official / support wording in name or bio",
        "count of official, support, help, customer care, billing words",
    ),
    "h_role": (
        "Wording",
        "Executive-title words in name or bio",
        "chief, CEO, CFO, VP, director, board member ...",
    ),
    "bio_len": ("Wording", "Bio length", "characters in the bio read by OCR"),
    "post_brand": (
        "Wording",
        "Posts that mention the brand",
        "number of the 3 posts containing the client brand",
    ),
    "name_ratio": (
        "Name",
        "Display name vs brand, closeness",
        "fuzzy ratio between the display name and the client brand (letters and digits only)",
    ),
    "handle_ratio": (
        "Name",
        "Handle vs brand, closeness",
        "fuzzy ratio between the handle and the client brand",
    ),
    "handle_brand": ("Name", "Brand inside the handle", "partial fuzzy match of the brand in the handle"),
    "handle_len": ("Name", "Handle length", "characters in the handle (taken from the URL bar)"),
    "handle_digits": ("Name", "Digits in the handle", "count of digits"),
    "handle_us": ("Name", "Dots / underscores in the handle", "count of . and _"),
    "handle_extra": (
        "Name",
        "Extra letters around the brand in the handle",
        "handle length minus brand length, when the brand is inside",
    ),
    "link_sim": (
        "Link",
        "Link domain close to the official one",
        "best fuzzy ratio between a non-official domain on the page and the client's domain",
    ),
    "link_none": ("Link", "No link on the page", "1 if no domain is found in the link field, bio or posts"),
    "av_logo_clip": (
        "Avatar",
        "Avatar looks like the client logo",
        "CLIP embedding cosine between the avatar crop and the client logo",
    ),
    "av_logo_ncc": (
        "Avatar",
        "Avatar pixels match the client logo",
        "normalised cross-correlation of the avatar disc with the logo, best of 5 crops",
    ),
    "av_exec_ncc": (
        "Avatar",
        "Avatar pixels match an executive photo",
        "same measure against the client's protected-executive photos",
    ),
    "av_exec_clip": (
        "Avatar",
        "Avatar looks like an executive photo",
        "CLIP cosine against the executive photos",
    ),
    "av_sil_ncc": (
        "Avatar",
        "Avatar is the default silhouette",
        "pixel match with a generic default-avatar picture",
    ),
    "av_sil_clip": (
        "Avatar",
        "Avatar looks like a default silhouette",
        "CLIP cosine with the generic default avatar",
    ),
    "av_av_std": ("Avatar", "Avatar contrast", "standard deviation of grey values in the avatar disc"),
    "av_av_mean": ("Avatar", "Avatar brightness", "mean grey value of the avatar disc"),
    "av_av_sat": ("Avatar", "Avatar colourfulness", "mean colour saturation of the avatar"),
    "verified": (
        "Account",
        "Verified badge",
        "a filled badge-sized disc in the platform colour beside the name",
    ),
    "n_egg": ("Rules", "Default avatar and empty bio", "rule penalty"),
    "n_scam": ("Rules", "Scam-words penalty", "rule penalty"),
    "exec_role_bio": (
        "Executive",
        "Bio carries the executive's role",
        "fuzzy match of the executive's title in the bio",
    ),
}

# Feature table for the report: tr = train rows; Xcols = the model's feature names
tr = ST["train"]
Xcols = ART["cols"]

# imp = the model's feature importances, largest first
imp = pd.Series(F["model"].feature_importances_, Xcols).sort_values(ascending=False)

# Keep the top 30 described features with their importance and their mean value on impersonations vs harmless pages
feats = []
for c in imp.index:
    if c in DESC and len(feats) < 30:
        g, what, how = DESC[c]
        v = tr[c] if c in tr.columns else None
        if v is None:
            continue
        feats.append(
            [
                g,
                c,
                what,
                how,
                round(float(imp[c]) / float(imp.sum()), 4),
                float(tr.loc[tr.label == 1, c].mean()),
                float(tr.loc[tr.label == 0, c].mean()),
            ]
        )


# ---- histogram of the model's p on pages that reach it
# hpan() = histogram data: the model's probability and the true label for every page that reached the model
def hpan(name, d):
    m = d[d.stage != "rules"]
    return dict(name=name, p=[round(float(x), 4) for x in m.p_ml], y=[int(x) for x in m.label])


# HIST = the histogram panels (val, and train out-of-fold) with the 0.02 / 0.98 decision lines
HIST = dict(
    title="Where the Step 3 model puts the pages that reach it",
    pos="impersonations",
    neg="not impersonations",
    xlab="model probability of impersonation",
    bins=20,
    band=[0.02, 0.98],
    lines=[dict(x=0.02, label="0.02"), dict(x=0.98, label="0.98")],
    panels=[
        hpan("Val (model trained on train)", ST["val"]),
        hpan("Train (predictions on companies held out in 5 folds)", ST["train"]),
    ],
    note="Pages in the shaded band go to Luna. Train probabilities come from a 5-fold split by company, so the model never scores a page of a company it was trained on.",
)

# ---- per-concept table
# Per-profile-type table (accuracy on test, val and train for each page type)
rows = []

# classes.json holds the profile-type definitions
cl = lj(f"{DATA}/classes.json")["concepts"]

# One table row per profile type: its truth, then test (count + accuracy), val accuracy, train accuracy
for c in sorted(ST["test"].concept.unique()):
    row = [c, "impersonation" if ST["test"][ST["test"].concept == c].label.iloc[0] == 1 else "not"]
    for sp in ("test", "val", "train"):
        g = ST[sp][ST[sp].concept == c]
        row += (
            [str(len(g)), f"{(g.p_final == g.label).mean():.3f}"]
            if sp == "test"
            else [f"{(g.p_final == g.label).mean():.3f}"]
        )
    rows.append(row)

# The finished per-profile-type table
UNCLEAR = dict(
    head=["Profile type", "Truth", "Test pages", "Test accuracy", "Val accuracy", "Train accuracy"], rows=rows
)

# ---- sample extraction record
# Pick one example test page (a fake brand-support page) to show what Step 1 extracts
S1 = ST["test"]
ex = S1[(S1.concept == "P1_brand_support_clone")].iloc[2]
exid = ex.id

# Its OCR lines
O = lj(f"{WORK}/ocr_test.json")[exid]

# The parser, to show the fields read from that page
from utilities.layout_parser import parse

# pp = the fields the parser reads from the example
pp = parse(O)

# sample = the record shown in the page: each parsed field and which steps use it
sample = dict(
    title=f"What Step 1 pulls out of one screenshot ({exid}) and which steps use it",
    lines=[
        [f'"platform": "{pp["platform"]}"', "2 3"],
        [f'"display_name": "{pp["name"]}"', "2 3 4"],
        [f'"handle": "{pp["handle"]}"', "2 3 4"],
        [f'"bio": "{pp["bio"][:70]}"', "2 3 4"],
        [f'"link": "{pp["link"]}"', "2 3 4"],
        [
            f'"followers": {int(pp["followers"] or 0)}, "following": {int(pp["following"] or 0)}, "posts": {int(pp["posts_n"] or 0)}',
            "3",
        ],
        [f'"joined": "{pp["joined"]}"', "3"],
        [f'"recent_posts": {len(pp["posts"])} read', "2 3 4"],
        [f'"avatar_matches_client_logo": {str(bool(ex.av_logo_strong)).lower()}', "2 3 4"],
        [f'"avatar_matches_executive_photo": {str(bool(ex.av_exec_strong)).lower()}', "2 3 4"],
        [f'"verified_badge": {str(bool(ex.verified)).lower()}', "2 3 4"],
        ["ground truth (label, concept, rationale)", "–"],
        ["the HTML behind the screenshot", "–"],
    ],
    verdict="The pipeline reads only the screenshot plus what the client supplies (brand, domain, logo, executives with photos). It never sees the label, the profile type, the CSV text or the HTML.",
)

# ---- hard cases
# Client name -> safe folder name
KEY = lambda c: re.sub(r"[^A-Za-z0-9]+", "_", c)


# Load an image, shrink it to `side` px, and return it as a base64 jpeg (to embed in the page)
def jpg(path, side, q):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    o = io.BytesIO()
    im.save(o, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(o.getvalue()).decode()


# HARDWHY = one-line reason why each tricky page type is hard
HARDWHY = {
    "P4_exec_full": "Copies a protected executive: the same name, title and photo as the real executive, on a new account with a different handle.",
    "HN10_exec_genuine": "The executive's real account: the same name, role and face as the fake copies, so name and photo alone cannot tell them apart. Only the badge, link and account history can.",
    "HP3_fan_claims_official": "Calls itself a fan page but says it is run by the brand's own team. The fan label looks safe, but the page claims to be the client.",
    "HN6_scam_precedence": "Uses the brand name and logo, but asks the visitor to verify an account or claim a refund. That is a scam page, a different incident type, so it is not scored as impersonation.",
    "P3_lookalike_handle": "A handle one or two characters away from the brand, with the brand's logo and support wording.",
    "HP6_exec_two_signals": "The executive's exact name and role bio but a different or default photo: two of the three signals, which is enough.",
    "HN11_exec_namesake": "A different person who happens to share the executive's name: a different face and an unrelated bio.",
    "HN3_critic_boycott": "Complains about the brand and says so. It names the brand all the time but is not pretending to be it.",
    "N7_exec_egg_name_match": "Only the executive's name on an empty default-photo profile: name alone is not enough, so it is an Executive Name Match, not impersonation.",
    "HN7_ato_precedence": "A genuine-looking old account that suddenly posts crypto or 'we are changing our name': a hacked account (behaviour change), not impersonation.",
    "HN1_fan_official_word": "A fan account whose bio uses the word 'official' while saying it is a fan page.",
    "P2_fake_official_page": "Calls itself the official brand page on a look-alike link.",
    "HP1_subtle_clone": "A near-identical copy of the brand page with only small differences.",
    "N1_official_client": "The client's own account, which looks exactly like what an impersonator would copy.",
    "HN8_unauthorized_seller": "An insider or reseller selling the brand's products unofficially: an Unauthorized Social Media Account, not impersonation.",
}


# pick() = choose one page of a profile type from a split (optionally only from one stage / only wrong ones)
def pick(sp, concept, stage=None, nth=0, wrong=None):
    d = ST[sp]
    d = d[d.concept == concept]
    if stage:
        d = d[d.stage == stage]
    if wrong is not None:
        d = d[(d.p_final != d.label) == wrong]
    return d.iloc[min(nth, len(d) - 1)] if len(d) else None


# sel = the list of (split, profile type, stage) shown as example cards in the page
sel = [
    ("test", "P4_exec_full", "llm"),
    ("test", "HN10_exec_genuine", None),
    ("test", "HP3_fan_claims_official", "llm"),
    ("val", "HN6_scam_precedence", None),
    ("test", "P3_lookalike_handle", "llm"),
    ("test", "HP6_exec_two_signals", None),
    ("test", "HN11_exec_namesake", None),
    ("val", "HN3_critic_boycott", "llm"),
    ("test", "N7_exec_egg_name_match", None),
    ("test", "HN7_ato_precedence", None),
    ("test", "HN1_fan_official_word", None),
    ("train", "HP1_subtle_clone", None),
    ("train", "N1_official_client", None),
    ("train", "HN8_unauthorized_seller", None),
]

# cards = the example cards; refs = each client's homepage image
cards, refs = [], {}

# meta[split][id] = the csv row (used for platform and customer)
meta = {sp: {r["id"]: r for r in load_split(sp)} for sp in ST}

# Build one card per selected example: screenshot, what each step said, the key signals, and whether it was right
for sp, conc, stg in sel:
    wrong = True if sp == "train" else None
    r = pick(sp, conc, stg, 0, wrong)
    if r is None:
        r = pick(sp, conc, None, 0, wrong)
    if r is None:
        continue
    m = meta[sp][r.id]
    cust = m["customer"]
    if cust not in refs:
        import os

        cand = [
            f"{DATA}/reference/{KEY(cust)}_homepage.png",
            f"{DATA}/reference/{KEY(cust)}_loginpage.png",
            CLIENT[cust]["logo"],
        ]
        refs[cust] = jpg(next(c for c in cand if os.path.exists(c)), 420, 60)
    if r.stage == "rules":
        dec = f"Step 2 rules: score {r.rule_score:.3f} on the 0–1 scale (impersonation at ≥ {ART['rule_hi']}, not at ≤ {ART['rule_lo']})."
    elif r.stage == "ml":
        dec = f"Step 3 model: p = {r.p_ml:.3f} (decides at ≥ 0.98 or ≤ 0.02)."
    else:
        dec = f"Steps 2–3 were unsure (rule score {r.rule_score:.3f}, model p = {r.p_ml:.2f}). Luna answered {'impersonation' if r.luna == 1 else 'not impersonation'} with p = {r.luna_p:.2f}."
    ok = r.p_final == r.label
    if not ok:
        dec = "WRONG. " + dec + " This is one of the 4 errors on train + val."
    sig = [
        ["platform", m["platform"]],
        ["verified badge", "yes" if r.verified else "no"],
        ["avatar = client logo", "yes" if r.av_logo_strong else "no"],
        ["avatar = executive photo", "yes" if r.av_exec_strong else "no"],
        ["link", "official" if r.link_official else ("other domain" if r.link_other else "none")],
        ["scam words", "yes" if r.scam_any else "no"],
        ["fan / parody words", "yes" if r.disc_any else "no"],
    ]
    cards.append(
        dict(
            side="pos" if r.label == 1 else "neg",
            id=r.id,
            split=sp,
            concept=conc,
            customer=cust,
            truth=int(r.label),
            pred=int(r.p_final),
            stage={"rules": "Step 2", "ml": "Step 3", "llm": "Step 4"}[r.stage],
            img=jpg(f"{SHOT}/{r.id}.png", 640, 68),
            hard=HARDWHY.get(conc, ""),
            decided=dec,
            sig=sig,
        )
    )

# Bundle the cards for the page
HARD = dict(cases=cards, refs=refs)

# ---- steps
# Format 'count (percent)'
real_pct = lambda a, b: f"{a:,} ({100 * a / max(1, b):.1f}%)"

# STEPS = one entry per sliding card (Step 0 ... Step 5): its title, text, numbers, confusion matrices and tables
STEPS = [
    dict(
        id="step0",
        tag="Step 0",
        status="done",
        title="The data and the starting score",
        runsOn=dict(
            order="Pipeline",
            text="Every profile screenshot goes through the steps in this order:",
            total=n_dev,
            real=n_pos,
            harmless=n_dev - n_pos,
            outs=[
                ["Step 1", "read the screenshot (OCR, avatar match)"],
                ["Step 2", "rules"],
                ["Step 3", "small model on what the rules leave unclear"],
                ["Step 4", "Luna on what the model is unsure about"],
            ],
            flow=True,
        ),
        goal="Find impersonation: a social profile that pretends to be the client, its brand or one of its protected executives (PS doc). Own and verified accounts, partners, fan, parody, review and critic pages, namesakes, and other incident types (scam, hacked account, unauthorized seller, executive name match) are not impersonation here.",
        uses=[
            [
                "Impersonations (P1–P6)",
                "brand support clone, fake official page, look-alike handle, full executive copy, fake recruiter, localized support clone",
            ],
            [
                "Hard impersonations (HP1–HP6)",
                "subtle clones, high-follower mismatch, fan page that claims to be official, misspelled name, dormant logo-and-name account, executive with two of three signals",
            ],
            [
                "Not impersonation (N1–N7)",
                "the client's own account, franchisee / partner, fan, parody, review / news, unrelated same-name business, executive name only on an empty profile",
            ],
            [
                "Hard negatives (HN1–HN11)",
                "fan page with 'official', employee personal account, critic, client alt account, partner co-branding, scam and hijacked-account precedence, unauthorized seller, inspired-name collision, genuine executive, namesake",
            ],
            [
                "Size and splits",
                f"4,470 profiles (25 clients × 30 profile types × 6, minus 10 skipped). Train 3,036 / val 720 / test 714; split by client (17 / 4 / 4) and by wording templates, so val and test clients and wording are never seen in training",
            ],
            [
                "What the pipeline sees",
                "only the 900×900 screenshot plus the client's brand, domain, logo and protected executives with photos. Never the label, profile type or the CSV text",
            ],
        ],
        method=[
            "Splits by client: val and test clients never appear in training; each wording template belongs to one split.",
            "Every threshold and model is fitted on train; val chooses between options; test is scored once at the end, on the frozen pipeline.",
        ],
        matrix=True,
        notes=[
            "Starting score with no steps: every profile answered 'not impersonation'. Every later step has to beat this."
        ],
    ),
    dict(
        id="step1",
        tag="Step 1",
        status="done",
        title="Read the screenshot (OCR and avatar matching)",
        runsOn=dict(
            order="Runs on",
            text="Every screenshot, first.",
            total=n_dev,
            real=n_pos,
            harmless=n_dev - n_pos,
            outs=[
                ["Parsed fields", "platform, name, handle, bio, link, followers, joined date, 3 posts"],
                ["Avatar matches", "client logo, executive photos, default-avatar, face / logo / initials"],
                ["Signals", f"{len(Xcols)} numbers for Steps 2–4"],
            ],
        ),
        goal="Turn the photo into structured facts the later steps can use, without ever reading the CSV text. This step makes no decision of its own.",
        flow=[
            ["Screenshot", "900×900 profile page"],
            ["docTR OCR", "text lines with positions"],
            [
                "Layout parser",
                "platform from the URL bar; name, handle, bio, link, stats, posts from line positions",
            ],
            ["Avatar crop", "fixed box per layout"],
            [
                "Match",
                "CLIP + pixel match to the client logo and executive photos; pixel match to a generic default avatar",
            ],
            ["Signals", "numbers", True],
        ],
        sample=sample,
        method=[
            "OCR: docTR (db_resnet50 + crnn_vgg16_bn) on GPU, about 5 pages a second. Not a Chinese-origin model.",
            "Parser: the URL bar gives the platform and handle; the first text under the avatar is the name; stats and joined lines are recognised by pattern; post headers ('name @handle · 5h') split off the 3 posts.",
            "Verified badge: a filled badge-sized disc in the platform colour next to the name (found by connected components).",
            "Client side: logo, wordmark and each protected executive's name, role and photo come from the client profile (here: the dataset's assets).",
        ],
        charts=[
            dict(
                title="How exactly the parser recovers each field (share of profiles), train / val",
                legend=["train", "val"],
                rows=[
                    ["Platform", 0.998, 0.99],
                    ["Handle", 0.997, 0.997],
                    ["Display name", 0.972, 0.992],
                    ["Bio", 0.97, 0.974],
                    ["Link", 0.996, 1.0],
                    ["Followers", 0.998, 0.997],
                    ["Posts count", 0.974, 0.979],
                ],
            )
        ],
        files=["Trial1/s0_ocr.py", "Trial1/s0_parse.py", "Trial1/s0_avatar.py", "Trial1/s1_signals.py"],
        matrix=False,
        notes=[
            "Checked against the CSV fields on train and val; the same check on test, run after freezing, gave 0.990 platform, 1.000 handle, 0.989 name, 0.982 bio, 0.996 link, 0.997 followers.",
            "Most misses are OCR noise on stylised or very short names, and the parser is built for the layout of these rendered pages, which is simpler than real social-media pages.",
        ],
    ),
    dict(
        id="step2",
        tag="Step 2",
        status="done",
        title="Rules: score the page against the definition",
        runsOn=dict(
            order="Runs on",
            text="Every page, first.",
            total=n_dev,
            real=n_pos,
            harmless=n_dev - n_pos,
            outs=[
                ["Clear impersonation", f"{r_pos:,}"],
                ["Clear not impersonation", f"{r_neg:,}"],
                [
                    "Unclear → Step 3",
                    f"{len(unc):,} ({int(unc.label.sum())} impersonations, {len(unc) - int(unc.label.sum()):,} not)",
                ],
            ],
        ),
        goal="Settle the obvious pages for free: brand-name, logo, 'official' claims and look-alike links add points; a verified badge, disclaimers, partner or personal wording, scam wording and an official link take points away.",
        flow=[
            ["Signals", "from Step 1"],
            ["Add points", "brand name, logo, claims, look-alike link, executive name + photo / role"],
            ["Subtract points", "verified, fan / parody, partner, personal, scam, official link"],
            [
                "Score 0–1",
                f"points added up, rescaled to 0–1; ≥ {ART['rule_hi']} impersonation · ≤ {ART['rule_lo']} not",
            ],
            ["Impersonation", "high score", True],
            ["Not impersonation", "low score", True],
            ["Unclear", "→ Step 3", True],
        ],
        rules=[
            [
                "Brand in the display name",
                "name contains or closely matches the brand",
                "+2.0 toward impersonation",
                "Adobe Care",
            ],
            ["Exactly the brand name", "name equals the brand", "+0.5 more", "Adobe"],
            ["Misspelled brand", "name close to but not containing the brand", "+1.5", "Ad0be"],
            [
                "Brand in the handle",
                "handle contains the brand (more if extra letters)",
                "+1.0 / +0.5",
                "@adobe.support57",
            ],
            ["Avatar is the client logo", "pixel / CLIP match to the logo", "+2.5", ""],
            [
                "Look-alike link",
                "domain close to the official one",
                "+2.0 (+0.5 any other domain)",
                "adobehelp24.com",
            ],
            [
                "'Official' / support wording",
                "official, support, help, customer care ...",
                "+1.5 (+0.5 in the handle)",
                "Fix your Adobe problems fast",
            ],
            ["Brand wordmark in the banner", "banner shows the brand name", "+1.0", ""],
            ["Executive name", "exact name of a protected executive", "+2.0", "Sofia Moreau"],
            ["Executive photo", "name + photo match", "+2.5 more", ""],
            ["Executive role bio", "name + role bio match", "+1.5 more", "Chief Financial Officer"],
            ["Verified badge", "badge disc detected", "−4.0 toward not impersonation", ""],
            [
                "Scam wording",
                "giveaway, verify your account, refund, crypto ...",
                "−4.0",
                "Claim your gift at ...",
            ],
            [
                "Fan / parody / review wording",
                "fan, parody, review, news, critic, not affiliated",
                "−2.5",
                "",
            ],
            ["Personal-account wording", "'views are my own', 'personal account'", "−2.0", ""],
            ["Partner / reseller wording", "partner, reseller, certified, agency", "−1.5", ""],
            ["Official link", "the link field is the client's domain", "−1.5", "adobe.com/help"],
            [
                "Default photo and empty bio",
                "pixel match to a generic grey default avatar, and no bio",
                "−1.5",
                "",
            ],
            ["Score ≥ high cut-off", f"0–1 score ≥ {ART['rule_hi']}", "Impersonation", ""],
            ["Score ≤ low cut-off", f"0–1 score ≤ {ART['rule_lo']}", "Not impersonation", ""],
            ["Anything else", "", "Unclear → Step 3", ""],
        ],
        method=[
            "Points are added up and rescaled to a 0–1 score with fixed bounds (−18.5 = every negative rule fires, +21.0 = every positive rule fires); the bounds are not fitted on data. 0 = strong evidence against impersonation, 1 = strong evidence for.",
            f"Cut-offs are searched on TRAIN in steps of 0.001: the lowest score at which 99.5% of pages above it are impersonations ({ART['rule_hi']}) and the highest at which 99.5% of pages below it are not ({ART['rule_lo']}).",
            "Point sizes are hand-set from the PS-doc definition; no model is trained in this step.",
        ],
        charts=[
            dict(
                title="What the rules decided (train + val)",
                unit="pages",
                rows=[
                    ["Clear impersonation", r_pos],
                    ["Clear not impersonation", r_neg],
                    ["Unclear → Step 3", len(unc)],
                    ["Wrong decisions", r_wrong],
                ],
            )
        ],
        files=["Trial1/s1_rules.py"],
        matrix=True,
        tri=TRI["step2"],
        triTitle="Pages the rules received: all of them. Each page ends in one of three boxes: said impersonation, said not, or passed on to Step 3.",
        notes=[
            f"The rules decide only {100 * len(r_dec) / n_dev:.0f}% of pages ({r_wrong} mistakes on train + val; none on test). The other {100 - 100 * len(r_dec) / n_dev:.0f}% are passed on, not answered, so read the scores below as 'right on the pages it dared to decide', not as the pipeline's accuracy.",
            "The pipeline's accuracy is on the last card (Step 5), after every page has an answer.",
        ],
    ),
    dict(
        id="step3",
        tag="Step 3",
        status="done",
        title="Small model (LightGBM) on the unclear pages",
        runsOn=dict(
            order="Runs on",
            text="Only the pages the rules called unclear.",
            total=len(unc),
            real=int(unc.label.sum()),
            harmless=len(unc) - int(unc.label.sum()),
            outs=[
                [
                    "Sure (p ≥ 0.98 or ≤ 0.02) → decided",
                    f"{len(m_dec):,} ({m_pos} impersonation, {m_neg} not)",
                ],
                [
                    "Unsure (0.02–0.98) → Step 4 Luna",
                    f"{len(l_df)} ({l_pos} impersonations, {len(l_df) - l_pos} not)",
                ],
            ],
        ),
        goal="Decide the grey-area pages: disguised, subtle, hard-positive and hard-negative profiles.",
        flow=[
            ["Unclear page", "from Step 2"],
            [f"{len(Xcols)} features", "signals from Step 1 plus the Step 2 rule points"],
            ["LightGBM", "300 trees, 15 leaves, trained on train"],
            ["Impersonation", "p ≥ 0.98", True],
            ["Not impersonation", "p ≤ 0.02", True],
            ["Unsure", "→ Step 4", True],
        ],
        rules=[
            ["Sure impersonation", "p ≥ 0.98", "Impersonation", "a clean support-clone page"],
            ["Sure not", "p ≤ 0.02", "Not impersonation", "a verified official account"],
            [
                "Unsure",
                "0.02 < p < 0.98",
                "To Step 4 (Luna)",
                "a fan page that claims to be run by the brand",
            ],
        ],
        method=[
            "Gradient-boosted trees on the signals below; trained on all train pages, so it also sees what the rules decide.",
            "The decision band (0.02 / 0.98) was chosen on train predictions made company by company and on val: it keeps decided pages at about 99.8% right while sending about 3% to Luna.",
            "Wording is represented only by keyword counts, never by the raw text, so the model cannot memorise the template sentences.",
        ],
        hists=[HIST],
        features=feats,
        featMeta=dict(
            title=f"The 30 features that matter most (of {len(Xcols)}): what each measures and how it is made from the data",
            use="Importance (share of all splits)",
            signed=False,
            pos="Avg, impersonation",
            neg="Avg, not",
            foot="Importance = share of all tree splits that use the feature. Averages are over train pages. Account age, post and follower counts matter a lot here because this is a generated dataset; with those removed the model still scores AUC 0.999 on val.",
        ),
        files=["Trial1/s2_ml.py", "Trial1/pipeline.py"],
        matrix=True,
        tri=TRI["step3"],
        triTitle="Pages the model received: only the ones the rules passed on. Each ends as said impersonation, said not, or passed on to Step 4 (Luna).",
        notes=[
            f"{len(m_dec):,} pages decided here with {m_wrong} mistakes on train + val; none on test. About 3% of the pages it received are passed on to Luna, not answered, so the scores below cover only the pages it decided.",
            "The pipeline's accuracy is on the last card (Step 5), after every page has an answer.",
        ],
    ),
    dict(
        id="step4",
        tag="Step 4",
        status="done",
        title="Luna (Decisions API) on the unsure pages",
        runsOn=dict(
            order="Runs on",
            text="Only the pages Step 3 was unsure about.",
            total=len(l_df),
            real=l_pos,
            harmless=len(l_df) - l_pos,
            outs=[
                ["Luna: impersonation", f"{int((l_df.luna == 1).sum())}"],
                ["Luna: not impersonation", f"{int((l_df.luna == 0).sum())}"],
                ["Right", f"{l_right} of {len(l_df)} (the model's guess alone: {ml_right})"],
            ],
        ),
        goal="Look at the screenshot the way an analyst would and decide the pages the small model cannot.",
        flow=[
            ["Unsure page", "from Step 3"],
            ["Screenshot + logo", "profile screenshot, client logo"],
            ["Client facts", "brand, domain, protected executives; automated checks from Step 1"],
            ["Luna", "gpt-6-luna, Decisions API, one choice question"],
            ["Impersonation", "", True],
            ["Not impersonation", "", True],
        ],
        rules=[
            [
                "Impersonation",
                "pretends to be the client, its brand or a protected executive (2 of name, photo, bio)",
                "Impersonation",
                "support clone with the client logo",
            ],
            [
                "Fan page run by the brand",
                "calls itself a fan page but says the brand's team runs it",
                "Impersonation",
                "'Our fan page is run by Amazon'",
            ],
            ["Own / verified account", "the client's genuine account", "Not impersonation", ""],
            [
                "Fan, parody, review, critic",
                "openly says what it is and is not run by the brand",
                "Not impersonation",
                "",
            ],
            [
                "Partner, employee, namesake",
                "mentions or shares the name without pretending",
                "Not impersonation",
                "",
            ],
            [
                "Scam, hacked account, unofficial seller",
                "other incident types, even with the brand's name and logo",
                "Not impersonation",
                "'verify your account'",
            ],
        ],
        method=[
            "Luna's Decisions API (gpt-6-luna), not the generative mode: one choice question, answered with a probability. Cost about $0.0001 a page; answers are cached.",
            "The prompt was revised once, after reading train errors: fan pages run by the brand count as impersonation, and verify-account / refund pages are scam, not impersonation. Val and test were not used to write it.",
            "Luna never sees the label, the profile type, the CSV text or the Step 3 probability.",
        ],
        charts=[
            dict(
                title="Luna on the unsure pages: right / wrong",
                legend=["right", "wrong"],
                rows=[
                    [
                        sp,
                        int((d[d.stage == "llm"].p_final == d[d.stage == "llm"].label).sum()),
                        int((d[d.stage == "llm"].p_final != d[d.stage == "llm"].label).sum()),
                    ]
                    for sp, d in ST.items()
                ],
            ),
            dict(
                title="Same pages: the Step 3 guess alone, right / wrong",
                legend=["right", "wrong"],
                rows=[
                    [
                        sp,
                        int((d[d.stage == "llm"].ml_guess == d[d.stage == "llm"].label).sum()),
                        int((d[d.stage == "llm"].ml_guess != d[d.stage == "llm"].label).sum()),
                    ]
                    for sp, d in ST.items()
                ],
            ),
        ],
        cases=True,
        files=["Trial1/s3_luna.py"],
        matrix=True,
        resultKey="luna",
        noCmp=True,
        resultTitle="Pages Luna received (only the ones Step 3 passed on), Luna against the truth",
        notes=[
            "Finished pipeline. Train still has 4 mistakes (unauthorized seller ×2, subtle clone, the client's own account); val and test have none."
        ],
    ),
    dict(
        id="step5",
        tag="Step 5",
        status="done",
        title="Frozen pipeline: test",
        runsOn=dict(
            order="Test",
            text="Run once on the 4 held-out companies, after everything was frozen.",
            total=len(ST["test"]),
            real=int(ST["test"].label.sum()),
            harmless=len(ST["test"]) - int(ST["test"].label.sum()),
            outs=[
                ["Step 2 rules", f"{int((ST['test'].stage == 'rules').sum())}"],
                ["Step 3 model", f"{int((ST['test'].stage == 'ml').sum())}"],
                ["Step 4 Luna", f"{int((ST['test'].stage == 'llm').sum())}"],
            ],
        ),
        goal="Score the finished pipeline once on pages from companies and wording it never saw. Nothing was changed after seeing this result.",
        diagram=True,
        method=[
            "Code, model and thresholds are frozen with checksums (reports/frozen_manifest.md5) before the test pages were read.",
            "Test is scored once; no threshold, feature, prompt or model was changed afterwards.",
        ],
        unclearTable=UNCLEAR,
        files=["Trial1/run_test.py"],
        matrix=True,
        resultKey="final",
        noCmp=True,
        final=FINAL,
        resultTitle="Every page, final answer of the whole pipeline",
        notes=[
            "Read the 100% with care. The data is generated: the wording comes from templates and account stats separate the classes cleanly, so the real-world number will be lower.",
            "The parser and avatar crops are calibrated to the layout of these rendered pages; real social-media pages are messier.",
            "Executive photos are AI-generated faces and the executives are fictional; a real protected-executive list with photos is needed in production.",
            "A few dozen real screenshots would be the check that matters.",
        ],
    ),
]

# ---- compose HTML from the web-clone template
# ---- Compose the HTML from the template: split it into style, script and page shell
s = open(TEMPLATE).read()
css = s[s.find("<style>") + 7 : s.find("</style>")]

# i = where the template's script starts; js = the script text
i = s.find("<script>")
js = s[i + 8 :]

# Replace embedded images by a stub (only to make searching the text easier)
js = re.sub(
    r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]{100,}", "X", js
)  # only for locating text; real code is taken below

# raw_js = the script with images intact (the real code is taken from here)
raw_js = s[i + 8 :]

# Cut out the template's helper functions (part ... casesHTML) and its render() function so they can be adapted
a = raw_js.find("function part(")
b = raw_js.find("function casesHTML")
helpers = raw_js[a:b]

# render = the template's render() function
c = raw_js.find("function render()")
render = raw_js[c : raw_js.rfind("</script>")]

# Re-word the template: 'clone' becomes 'impersonation'
helpers = helpers.replace("Clone (alert)", "Impersonation (alert)").replace("Test pages", "Pages in test")

# Remove template pieces that are redefined below
helpers = re.sub(r"const SPLIT_NAMES = .*?;\n", "", helpers, flags=re.S)

# Remove the template's verdictClass() (a new one is defined below)
helpers = re.sub(r"function verdictClass\(v\)\{.*?\n\}\n", "", helpers, flags=re.S)

# More wording changes (clones -> impersonations)
helpers = (
    helpers.replace("not clones", "not impersonations")
    .replace("clones", "impersonations")
    .replace("The unclear names only, per split", "Accuracy per profile type, finished pipeline")
)

# Patch render(): show the three-way result table (s.tri) when a step has one
render = (
    render.replace(
        "    if (s.matrix){",
        '    if (s.tri){ h += part(3, "Result", `<h4>${esc(s.triTitle)}</h4>` + triHTML(s.tri) + notesHTML(s.notes)); } else if (s.matrix){',
    )
    .replace("pk = s.compareKey || prev", "pk = s.noCmp ? null : (s.compareKey || prev)")
    .replace(
        '<h4>Train · val · test${vs ? ", " + vs : ""} (positive = web clone)</h4>',
        '<h4>${s.resultTitle ? esc(s.resultTitle) + " · " : ""}Train · val · test (positive = impersonation)</h4>',
    )
    .replace(
        "    if (s.cases) m += casesHTML(HARD);",
        "    if (s.raw) m += s.raw;\n    if (s.cases) m += casesHTML(HARD);",
    )
    .replace("(positive = web clone)", "(positive = impersonation)")
    .replace('STEPS.filter(s=>s.status==="done").length - 1', "0")
)

# Patch render(): show the final confusion matrix when a step has one
render = render.replace(
    "      if (s.unclearTable)",
    "      if (s.final) r = `<h4>Final confusion matrix: every page, the answer of the whole pipeline (positive = impersonation)</h4>` + finalHTML(s.final);\n      if (s.unclearTable)",
).replace("The unclear names only, per split", "Accuracy per profile type (final answer)")
MYJS = r"""
const COLS = ["Impersonation","Not impersonation"];
const fmt = v => v.toFixed(3);
const esc = s => String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const SPLIT_NAMES = [["train","Train","3,036 pages"],["val","Val","720 pages"],["test","Test","714 pages"]];
function verdictClass(v){ if (/^\+|^Impersonation/.test(v)) return "alert"; if (/^−|^Not impersonation/.test(v)) return "clear"; return "other"; }
function casesHTML(H){
  const card = c => {
    const ok = c.pred === c.truth;
    const sig = c.sig.map(([n, v]) => `<span class="sg${(v === "yes" && n !== "verified badge") ? " on" : ""}"><b>${esc(n)}</b> ${esc(v)}</span>`).join("");
    return `<article class="hc ${c.side}">` +
      `<div class="hc-imgs"><figure><img src="${c.img}" alt="Profile screenshot ${esc(c.id)}" loading="lazy"><figcaption>${esc(c.id)}</figcaption></figure>` +
      `<figure class="ref"><img src="${H.refs[c.customer]}" alt="The client's real homepage" loading="lazy"><figcaption>${esc(c.customer)}: real homepage</figcaption></figure></div>` +
      `<div class="hc-body"><div class="hc-top"><span class="truth ${c.truth ? "pos" : "neg"}">${c.truth ? "Impersonation" : "Not impersonation"}</span>` +
      `<code>${esc(c.concept)}</code><span class="sp-chip ${c.split}">${c.split}</span>` +
      `<span class="out ${ok ? "ok" : "err"}">${ok ? "✓ right" : "✗ wrong"} at ${esc(c.stage)}</span></div>` +
      `<p><b>Why it is hard:</b> ${esc(c.hard)}</p><p><b>What decided it:</b> ${esc(c.decided)}</p>` +
      `<div class="sgs">${sig}</div></div></article>`;
  };
  const side = (k, title) => `<div><h4>${title}</h4><div class="hcs">${H.cases.filter(c => c.side === k).map(card).join("")}</div></div>`;
  return side("pos", "Impersonations that are easy to miss") + side("neg", "Not impersonations that are easy to flag");
}
function triHTML(T){
  const f = n => n.toLocaleString();
  const one = ([k, nm, rows]) => { const t = T[k]; const dec = t.decided, acc = dec ? (100 * t.right / dec).toFixed(1) : "–", cov = (100 * dec / t.n).toFixed(1);
    const row = (lab, a) => `<tr><th>${lab}</th><td class="${a[0] && lab.startsWith("Not") ? "bad" : ""}">${f(a[0])}</td><td>${f(a[1])}</td><td class="pass">${f(a[2])}</td></tr>`;
    return `<div class="sp ${k}"><div class="sp-h">${nm}<span>${f(t.n)} pages</span></div><table class="rt tri"><thead><tr><th></th><th>said impersonation</th><th>said not</th><th>passed on</th></tr></thead><tbody>` +
      row("Truly impersonation", t.pos) + row("Truly not", t.neg) + `</tbody></table><p class="note"><b>Decided ${f(dec)} of ${f(t.n)} (${cov}%)</b>, right on ${acc}% of those. <b>${(100 - 100 * dec / t.n).toFixed(1)}% passed on</b> to the next step.</p></div>`; };
  return '<div class="splits small">' + SPLIT_NAMES.map(one).join("") + "</div>";
}
function finalHTML(F){
  const f = n => n.toLocaleString(), pc = (a, b) => b ? (100 * a / b).toFixed(1) + "%" : "–";
  const one = ([k, nm]) => { const t = F[k], m = t.m, nm_ = {rules: "Step 2 rules", ml: "Step 3 model", llm: "Step 4 Luna"};
    const rows = ["rules", "ml", "llm"].map(st => `<tr><td>${nm_[st]}</td><td>${f(t.by[st][0])}</td><td>${pc(t.by[st][1], t.by[st][0])}</td></tr>`).join("");
    return `<div class="sp ${k}"><div class="sp-h">${nm}<span>${f(t.n)} pages</span></div>
      <table class="rt tri fin"><thead><tr><th></th><th>said impersonation</th><th>said not</th></tr></thead><tbody>
      <tr><th>Truly impersonation</th><td class="okc">${f(m[0][0])}</td><td class="${m[0][1] ? "bad" : ""}">${f(m[0][1])}</td></tr>
      <tr><th>Truly not</th><td class="${m[1][0] ? "bad" : ""}">${f(m[1][0])}</td><td class="okc">${f(m[1][1])}</td></tr></tbody></table>
      <p class="note"><b>Every page answered. Accuracy ${(100 * t.acc).toFixed(2)}%</b> · precision ${t.p.toFixed(3)} · recall ${t.r.toFixed(3)} · F1 ${t.f1.toFixed(3)}</p>
      <table class="rt"><thead><tr><th>Who answered</th><th>Pages</th><th>Right</th></tr></thead><tbody>${rows}</tbody></table></div>`; };
  return '<div class="splits small">' + SPLIT_NAMES.map(one).join("") + "</div>";
}
function workflowSVG(){
  const bx = (x, y, w, h, t1, t2, t3, cls) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="var(--${cls==='out'?'accent-soft':'panel'})" stroke="var(--${cls==='luna'?'accent':'line'})" stroke-width="${cls==='luna'?2:1}"/>` +
    `<text x="${x+w/2}" y="${y+22}" text-anchor="middle" font-weight="700" font-size="13" fill="var(--ink)">${t1}</text><text x="${x+w/2}" y="${y+42}" text-anchor="middle" font-size="11.5" fill="var(--muted)">${t2}</text>` +
    (t3 ? `<text x="${x+w/2}" y="${y+58}" text-anchor="middle" font-size="11.5" fill="var(--muted)">${t3}</text>` : "");
  const ar = (d) => `<path d="${d}" fill="none" stroke="var(--muted)" stroke-width="1.4" marker-end="url(#ah)"/>`;
  return `<svg viewBox="0 0 1300 330" width="1300" role="img" aria-label="Workflow of the finished pipeline" style="max-width:none"><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="var(--muted)"/></marker></defs>` +
    bx(10,40,150,70,"Screenshot","900×900 profile","+ client facts","") + bx(210,40,190,70,"STEP 1 · READ","docTR OCR, parser,","avatar matching","") +
    bx(450,40,190,70,"STEP 2 · RULES","score 0–1 from points;","decides the clear 14%","") + bx(690,40,200,70,"STEP 3 · LightGBM","decides when p ≥ 0.98","or ≤ 0.02 (83%)","") +
    bx(940,40,190,70,"STEP 4 · LUNA","Decisions API;","the unsure ~3%","luna") + bx(1160,20,130,60,"Impersonation","","", "out") + bx(1160,95,130,60,"Not impersonation","","", "out") +
    bx(450,230,840,60,"Test (714 pages): rules 98 · model 605 · Luna 11 · all 714 right","train + val: 3,756 pages, 4 mistakes (all on train)","","none") +
    ar("M160,75 H206") + ar("M400,75 H446") + ar("M640,75 H686") + ar("M890,75 H936") + ar("M1130,60 H1156") + ar("M1130,90 H1145 V125 H1156") + ar("M545,110 V160 H1225 V159") + ar("M790,110 V190 H1225 V159") + `</svg>`;
}
"""
_t = ST["test"]
_tr, _va = ST["train"], ST["val"]
_terr = int((_t.p_final != _t.label).sum())
_dev_err = int((_tr.p_final != _tr.label).sum() + (_va.p_final != _va.label).sum())

# Fill the test-split line (page counts per stage) from the real data
MYJS = MYJS.replace(
    "Test (714 pages): rules 98 · model 605 · Luna 11 · all 714 right",
    f"Test ({len(_t)} pages): rules {int((_t.stage == 'rules').sum())} · model {int((_t.stage == 'ml').sum())} · Luna {int((_t.stage == 'llm').sum())} · {len(_t) - _terr} right",
).replace(
    "train + val: 3,756 pages, 4 mistakes (all on train)",
    f"train + val: {len(_tr) + len(_va):,} pages, {_dev_err} mistakes",
)

# head = everything before the style (title is renamed)
head = s[: s.find("<style>")]
head = re.sub(r"<title>.*?</title>", "<title>Impersonation Pipeline Log</title>", head)

# shell = the page body (header, track of cards, footer)
shell = s[s.find("</style>") : s.find("<script>")]

# Replace the page's eyebrow text (the small heading above the title)
shell = re.sub(
    r'<div class="eyebrow">.*?</div>',
    '<div class="eyebrow">DRP · Impersonation · screenshot in → Step 2 rules → Step 3 small model → Step 4 Luna (Decisions API)</div>',
    shell,
    flags=re.S,
)

# Rename the page title
shell = shell.replace("Web Clone Build Log", "Impersonation Pipeline Log")

# Replace the footer text
shell = re.sub(
    r"<footer>.*?</footer>",
    "<footer>Positive class = impersonation (PS doc: the profile pretends to be the client, its brand or a protected executive). Scam, hacked-account, unauthorized-seller and executive-name-match profiles are not impersonation here. Splits by client: train 17 clients (3,036 profiles), val 4 (720), test 4 (714); val and test clients and wording are never seen in training. Everything was fitted and chosen on train + val; test is the score of the frozen pipeline. The data is generated, so read the scores as a test of the pipeline, not of real-world accuracy.</footer>",
    shell,
    flags=re.S,
)

# Step 00 (enrichment) and Runtime cards are built in utilities/report_sections.py
from utilities.report_sections import build as _xb

# Their CSS, the Step 00 html and the Runtime html
XCSS, XENRICH, XRUNTIME = _xb(len(Xcols))


# Take the inner content of a section (strip the wrapper) so it can go inside a card
def _inner(h):
    lead = re.search(r'<p class="lead">(.*?)</p>', h, re.S).group(1)
    body = re.sub(r'^<section[^>]*>\s*<h2>.*?</h2>\s*<p class="lead">.*?</p>', "", h, flags=re.S)
    body = re.sub(r"</section>\s*$", "", body.strip())
    return lead, '<div class="xin">' + body + "</div>"


_el, _eb = _inner(XENRICH)
_rl, _rb = _inner(XRUNTIME)

# Put Step 00 first in the sliding cards
STEPS.insert(
    0,
    dict(
        id="step00",
        tag="Step 00",
        status="done",
        title="How a screenshot becomes data (page by page)",
        goal=_el,
        raw=_eb,
        matrix=False,
        notes=[
            "Colours on each screenshot show which field the parser gave each OCR line to; pink is the fixed avatar box.",
            "The six pages (one per platform) are validation pages, never test.",
            "Nothing in this step decides anything; it only adds information to the page for Steps 2 to 4.",
        ],
    ),
)

# Put the Runtime card last
STEPS.append(
    dict(
        id="runtime",
        tag="Runtime",
        status="done",
        title="Runtime and model size",
        goal=_rl,
        raw=_rb,
        matrix=False,
        notes=[
            "Timings are per page with the models already loaded, on 80 to 96 validation pages after one warm-up batch.",
            "Luna's timing comes from 5 calls, so treat it as rough.",
        ],
    )
)

# Add the extra CSS for the new cards
css = css + XCSS

# Assemble the whole page: head + style + shell + script (helpers, data, render)
html = (
    head
    + "<style>"
    + css
    + shell
    + "<script>\n"
    + MYJS
    + "\nconst STEPS = "
    + json.dumps(STEPS, ensure_ascii=False)
    + ";\nconst SPLITS = "
    + json.dumps(SPLITS)
    + ";\nconst HARD = "
    + json.dumps(HARD, ensure_ascii=False)
    + ";\n"
    + helpers
    + render
    + "</script>\n"
    + "</body></html>"
    if not raw_js.rstrip().endswith("</html>")
    else None
)

# Write the file and print its size
open(OUT_HTML, "w").write(html)
print("wrote", OUT_HTML, len(html) // 1024, "KB", len(cards), "cards")
