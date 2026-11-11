"""Utility for Step 4: Luna (OpenAI Decisions API, gpt-6-luna) on the pages Stages 1-2 left unsure.  Same client pattern as SusDom/WebClone/05_luna.py.
Input: profile screenshot + client logo + client facts + Stage-0 automated checks.  One choice question: impersonation / not_impersonation.
Answers are cached (a rerun costs nothing); spend is capped; past the cap the remaining pages keep the ML guess.  API key read from the git-ignored .env, never printed.
Called by 06_luna_unsure_pages.py (train / val) and 08_run_test.py (test)."""

# base64 / io / PIL: turn images into text the API accepts; hashlib: cache key; concurrent.futures: several calls at once
import base64, hashlib, io, json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from PIL import Image

# Paths, client data and load_split
from utilities.config import *

# MODEL = the Luna model name; USD_PER_MTOK_INPUT = price per million input tokens (used for the cost estimate)
MODEL, USD_PER_MTOK_INPUT = "gpt-6-luna", 0.10

# BUDGET_USD = spend cap; CONCURRENCY = parallel calls; MAX_RETRIES = retries on temporary errors; TIMEOUT_S = seconds before a call is abandoned
BUDGET_USD, CONCURRENCY, MAX_RETRIES, TIMEOUT_S = 2.0, 8, 3, 60

# Folder where every answer is saved, so a rerun costs nothing
CACHE = CACHE_DIR / "luna"

# File holding OPENAI_API_KEY (or set the environment variable instead)
ENV = Path(
    os.environ.get("OPENAI_ENV_FILE", "/home/jay/Desktop/DRP/WebP/zf-webp/.env")
)  # file holding OPENAI_API_KEY (or export OPENAI_API_KEY)

# INSTRUCTIONS = the written definition Luna judges by: what IS and is NOT an impersonation (do not edit casually: it changes the cache key and therefore the results)
INSTRUCTIONS = """You are a brand-protection analyst. Decide whether a social-media profile is an IMPERSONATION of the client.

IMPERSONATION = the account pretends to BE the client: it uses the client's brand name (exact, misspelled, or look-alike) and presents itself as the client or its official / support / customer-care / careers presence, usually with the client's logo, copied branding or a look-alike link; OR it pretends to be one of the client's protected executives by matching at least TWO of: exact name, photo, bio/role.

A page that calls itself a "fan page" or "fan club" but says it is run by the brand, its team or its social team is claiming to be the client, so it IS an impersonation.

NOT IMPERSONATION: the client's own genuine account (verified, official link, long history); fan, parody, satire, review, news or critic accounts that openly say what they are and are not run by the brand; partners, resellers, franchisees; employees' personal accounts that merely mention the employer; a different person or business that only shares the name; an empty or default-photo profile that only shares an executive's name; scam / phishing / crypto-giveaway pages (for example pages asking to verify an account, confirm payment details, claim a refund or reward, or click a link), hacked genuine accounts and unofficial sellers: these are other incident types, not impersonation, even when they use the brand name and logo.

You get (1) the client's facts and logo, (2) a screenshot of the profile, (3) automated checks computed outside the model. Judge from what is visible on the screenshot."""

# QUESTIONS = the single multiple-choice question Luna must answer: impersonation / not_impersonation
QUESTIONS = [
    {
        "type": "choice",
        "name": "impersonation",
        "instructions": "Is this profile an impersonation of the client?",
        "choices": [
            {
                "value": "impersonation",
                "description": "pretends to be the client, its brand or one of its protected executives",
            },
            {
                "value": "not_impersonation",
                "description": "does not pretend to be the client (genuine, fan/parody/review, partner, namesake, personal, other incident type)",
            },
        ],
    }
]


# Make sure OPENAI_API_KEY is set: take it from the environment, else from the .env file; stop if missing (the key is never printed)
def _key():
    if not os.environ.get("OPENAI_API_KEY") and ENV.exists():
        for line in ENV.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not found")


# Shrink an image to at most `side` px, compress it as jpeg and wrap it in the format the API expects
def _img(path, side=900, q=80):
    im = Image.open(path).convert("RGB")
    im.thumbnail((side, side))
    o = io.BytesIO()
    im.save(o, "JPEG", quality=q)
    return {
        "type": "input_image",
        "image_url": "data:image/jpeg;base64," + base64.b64encode(o.getvalue()).decode(),
    }


# build_input() = the complete message for one page: the instructions, the client facts, the automated checks, the client logo and the screenshot
def build_input(r, s):
    C = CLIENT[r["customer"]]
    client = dict(
        name=C["brand"],
        official_domain=C["domain"],
        protected_executives=[dict(name=e["name"], role=e["role"]) for e in C["execs"]],
    )
    checks = dict(
        avatar_matches_client_logo=bool(s["av_logo_strong"]),
        avatar_matches_protected_executive_photo=bool(s["av_exec_strong"]),
        verified_badge_detected=bool(s["verified"]),
        link_on_page=(
            "official domain" if s["link_official"] else "other domain" if s["link_other"] else "none"
        ),
    )
    text = "## 1. Client\n" + json.dumps(client) + "\n\n## 3. Automated checks\n" + json.dumps(checks)
    content = [
        {"type": "input_text", "text": text},
        {"type": "input_text", "text": "Client's official logo:"},
        _img(C["logo"], 300),
        {"type": "input_text", "text": "## 2. Screenshot of the profile to judge:"},
        _img(f"{SHOT}/{r['id']}.png"),
    ]
    return [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": INSTRUCTIONS}]},
        {"type": "message", "role": "user", "content": content},
    ]


# estimate() = rough cost in dollars of a list of requests (text length / 4 tokens + ~1500 tokens per image)
def estimate(inputs):
    n = sum(1 for i in inputs for m in i for c in m["content"] if c["type"] == "input_image")
    ch = sum(len(c.get("text", "")) for i in inputs for m in i for c in m["content"])
    return (ch / 4 + 1500 * n) * USD_PER_MTOK_INPUT / 1e6


# _decide() = ask Luna about ONE page (with cache lookup, retries and cost tracking); returns choice, probability, cost, error
def _decide(client, input_):
    import openai

    key = hashlib.sha256(
        json.dumps({"m": MODEL, "q": QUESTIONS, "i": input_}, sort_keys=True).encode()
    ).hexdigest()
    path = CACHE / f"{key}.json"
    if path.exists():
        return {**json.loads(path.read_text()), "cache_hit": True}
    out = {"choice": None, "p_imp": None, "error": None, "cost": 0.0, "cache_hit": False}
    for a in range(MAX_RETRIES + 1):
        if a:
            time.sleep(min(2**a, 30) + random.random())
        try:
            resp = client.decisions.create(
                model=MODEL,
                input=input_,
                questions=QUESTIONS,
                prompt_cache_key="drp-impersonation",
                timeout=TIMEOUT_S,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError):
            raise
        except (
            openai.RateLimitError,
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        ) as e:
            out["error"] = type(e).__name__
            continue
        except openai.APIStatusError as e:
            out["error"] = f"{type(e).__name__} {getattr(e, 'status_code', '')}"
            break
        d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        u = d.get("usage") or {}
        out["cost"] += (
            (u.get("input_tokens", 0) - (u.get("input_tokens_details") or {}).get("cached_tokens", 0))
            * USD_PER_MTOK_INPUT
            / 1e6
        )
        an = next((x for x in d.get("answers", []) if x.get("name") == "impersonation"), None)
        if an and an.get("choice") in ("impersonation", "not_impersonation"):
            pr = {p["value"]: p["probability"] for p in an.get("probabilities", [])}
            out.update(choice=an["choice"], p_imp=pr.get("impersonation"), error=None)
            break
        out["error"] = "invalid response"
    if out["error"] is None:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
    return out


# decide_many() = ask Luna about many pages in parallel: build all requests, refuse if the estimate passes the cap, call the API, return {page id: answer}
def decide_many(df, sp):
    meta = {r["id"]: r for r in load_split(sp)}
    rows = [s for _, s in df.iterrows()]
    inputs = [build_input(meta[s["id"]], s) for s in rows]
    if estimate(inputs) > BUDGET_USD:
        sys.exit(f"Luna estimate ${estimate(inputs):.2f} passes the ${BUDGET_USD} cap; not called")
    _key()
    from openai import OpenAI

    client = OpenAI(max_retries=0)
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        outs = list(pool.map(lambda i: _decide(client, i), inputs))
    return {s["id"]: o for s, o in zip(rows, outs)}
