"""Step 4 (LLM): Luna (OpenAI Decisions API) answers the pages the rules and the ML model leave unsure.
One screenshot + the client's logo + client facts + automated checks go in; one multiple-choice answer (impersonation / not_impersonation) comes out.
Needs the environment variable OPENAI_API_KEY. If the key is missing or the call fails, the caller keeps the ML model's best guess."""

import base64
import io
import json
import os

from PIL import Image

from impersonation.settings import LUNA_ENABLED, LUNA_MODEL, LUNA_TIMEOUT_S

# The written definition Luna judges by (identical to the one used when the pipeline was frozen and tested)
INSTRUCTIONS = """You are a brand-protection analyst. Decide whether a social-media profile is an IMPERSONATION of the client.

IMPERSONATION = the account pretends to BE the client: it uses the client's brand name (exact, misspelled, or look-alike) and presents itself as the client or its official / support / customer-care / careers presence, usually with the client's logo, copied branding or a look-alike link; OR it pretends to be one of the client's protected executives by matching at least TWO of: exact name, photo, bio/role.

A page that calls itself a "fan page" or "fan club" but says it is run by the brand, its team or its social team is claiming to be the client, so it IS an impersonation.

NOT IMPERSONATION: the client's own genuine account (verified, official link, long history); fan, parody, satire, review, news or critic accounts that openly say what they are and are not run by the brand; partners, resellers, franchisees; employees' personal accounts that merely mention the employer; a different person or business that only shares the name; an empty or default-photo profile that only shares an executive's name; scam / phishing / crypto-giveaway pages (for example pages asking to verify an account, confirm payment details, claim a refund or reward, or click a link), hacked genuine accounts and unofficial sellers: these are other incident types, not impersonation, even when they use the brand name and logo.

You get (1) the client's facts and logo, (2) a screenshot of the profile, (3) automated checks computed outside the model. Judge from what is visible on the screenshot."""

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


def _img(im, side=900, q=80):
    """PIL image -> shrink to at most `side` px, compress as jpeg, wrap in the format the API expects."""
    im = im.convert("RGB").copy()
    im.thumbnail((side, side))
    o = io.BytesIO()
    im.save(o, "JPEG", quality=q)
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(o.getvalue()).decode()}


def build_input(client, s, screenshot):
    """The complete message for one page: instructions, client facts, automated checks, client logo, screenshot. s = the page's signals."""
    facts = dict(
        name=client["brand"],
        official_domain=client["domain"],
        protected_executives=[dict(name=e["name"], role=e["role"]) for e in client["execs"]],
    )
    checks = dict(
        avatar_matches_client_logo=bool(s["av_logo_strong"]),
        avatar_matches_protected_executive_photo=bool(s["av_exec_strong"]),
        verified_badge_detected=bool(s["verified"]),
        link_on_page=("official domain" if s["link_official"] else "other domain" if s["link_other"] else "none"),
    )
    text = "## 1. Client\n" + json.dumps(facts) + "\n\n## 3. Automated checks\n" + json.dumps(checks)
    content = [
        {"type": "input_text", "text": text},
        {"type": "input_text", "text": "Client's official logo:"},
        _img(client["logo_image"], 300),
        {"type": "input_text", "text": "## 2. Screenshot of the profile to judge:"},
        _img(screenshot),
    ]
    return [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": INSTRUCTIONS}]},
        {"type": "message", "role": "user", "content": content},
    ]


class Luna:
    def __init__(self):
        # Available only if enabled and a key is set; the key is read from the environment and never printed
        self.client = None
        self.reason = None
        if not LUNA_ENABLED:
            self.reason = "disabled (LUNA_ENABLED=0)"
        elif not os.environ.get("OPENAI_API_KEY"):
            self.reason = "OPENAI_API_KEY is not set"
        else:
            from openai import OpenAI

            self.client = OpenAI(max_retries=1)

    @property
    def available(self):
        return self.client is not None

    def decide(self, client, s, screenshot):
        """Ask Luna about ONE page. Returns {choice, p_imp, error}; choice is None when the call failed."""
        out = {"choice": None, "p_imp": None, "error": None}
        try:
            resp = self.client.decisions.create(
                model=LUNA_MODEL,
                input=build_input(client, s, screenshot),
                questions=QUESTIONS,
                prompt_cache_key="drp-impersonation",
                timeout=LUNA_TIMEOUT_S,
            )
        except Exception as e:
            out["error"] = type(e).__name__
            return out
        d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        an = next((x for x in d.get("answers", []) if x.get("name") == "impersonation"), None)
        if an and an.get("choice") in ("impersonation", "not_impersonation"):
            pr = {p["value"]: p["probability"] for p in an.get("probabilities", [])}
            out.update(choice=an["choice"], p_imp=pr.get("impersonation"))
        else:
            out["error"] = "invalid response"
        return out
