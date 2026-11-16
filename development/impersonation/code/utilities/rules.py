"""Stage 1b: hand-written rules (PS-doc definition) -> rule score + decision (pos / neg / unsure)."""

import numpy as np, pandas as pd


def rule_scores(d):
    """Each rule adds points (+ = looks like impersonation, - = looks innocent). Returns a table: one column per rule, one row per page."""
    # z = the points table (starts empty, same rows as d)
    z = pd.DataFrame(index=d.index)

    # nb = name is close to the brand; claim = the bio or handle claims to be official/support
    nb = d.name_brand >= 85
    claim = (d.h_claim > 0) | (d.handle_claim > 0)

    # --- brand impersonation evidence (+)
    # Name contains / closely matches the brand: +2 (+0.5 more if it is exactly the brand)
    z["r_name_brand"] = 2.0 * (d.name_has_brand | (d.name_brand >= 85)).astype(int) + 0.5 * (
        d.name_equals_brand == 1
    )

    # Name is a near-miss (typo) of the brand without containing it: +1.5
    z["r_typo_name"] = 1.5 * ((d.name_brand >= 70) & (d.name_has_brand == 0)).astype(int)

    # Handle contains / matches the brand: +1 (+0.5 if it has extra characters, e.g. brand_support)
    z["r_handle"] = 1.0 * (d.handle_has_brand | (d.handle_brand >= 80)).astype(int) + 0.5 * (
        (d.handle_has_brand == 1) & (d.handle_equals_brand == 0)
    ).astype(int)

    # Avatar is the client's logo (strong match): +2.5
    z["r_logo"] = 2.5 * d.av_logo_strong

    # Avatar only weakly resembles the logo: +1
    z["r_logo_weak"] = 1.0 * ((d.av_logo_strong == 0) & (d.av_logo_clip > 0.84)).astype(int)

    # Claims to be official/support: +1.5 for the bio, +0.5 for the handle
    z["r_claim"] = 1.5 * (d.h_claim > 0).astype(int) + 0.5 * d.handle_claim

    # Link goes to a look-alike of the official domain: +2 (+0.5 for any other external link)
    z["r_link_look"] = 2.0 * d.link_lookalike + 0.5 * d.link_other

    # Banner shows the brand wordmark: +1
    z["r_banner"] = 1.0 * d.banner_brand

    # Asks people to DM / WhatsApp / message: +0.5
    z["r_dm"] = 0.5 * (d.h_dm + d.p_dm > 0).astype(int)

    # --- executive impersonation: name + (photo | role bio)
    # en = the display name matches a protected executive
    en = d.exec_name_exact == 1

    # Exact executive name: +2
    z["r_exec_name"] = 2.0 * en

    # Executive name AND the avatar matches that executive's photo: +2.5
    z["r_exec_photo"] = 2.5 * (en & (d.av_exec_strong == 1)).astype(int)

    # Executive name AND a bio with the executive's role (or any role + brand): +1.5
    z["r_exec_bio"] = 1.5 * (
        en & ((d.exec_role_bio == 1) | ((d.exec_role_any == 1) & (d.exec_brand_bio == 1)))
    ).astype(int)

    # --- evidence against
    # Scam / giveaway / phishing wording is a different incident type, not impersonation: -4
    z["n_scam"] = -4.0 * ((d.h_scam > 0) | (d.p_scam > 0)).astype(int)

    # Verified badge: -4
    z["n_verified"] = -4.0 * d.verified

    # Says it is a fan / parody / review / unofficial page: -2.5
    z["n_disc"] = -2.5 * ((d.h_disc > 0) | (d.p_disc > 0)).astype(int)

    # "Opinions are my own" / personal account: -2
    z["n_own"] = -2.0 * ((d.h_own > 0) | (d.p_own > 0)).astype(int)

    # Partner / reseller / authorised agency wording: -1.5
    z["n_partner"] = -1.5 * ((d.h_partner > 0) | (d.p_partner > 0)).astype(int)

    # The profile's link field is the official domain: -1.5
    z["n_link_official"] = -1.5 * d.link_field_official

    # Personal-life wording (hobbies, family): -1
    z["n_personal"] = -1.0 * (d.h_personal > 0).astype(int)

    # Default avatar and empty bio (an "egg" account): -1.5 (never fires on this data)
    z["n_egg"] = -1.5 * ((d.av_default == 1) & (d.no_bio == 1)).astype(int)

    # Brand-new account (< 10 followers, no posts): -0.5
    z["n_young_nothing"] = -0.5 * ((d.followers >= 0) & (d.followers < 10) & (d.no_posts == 1)).astype(int)
    return z


# Fixed bounds of the raw points (most any page can lose / gain); the score is rescaled to 0..1 with them (not fitted on data).
RMIN, RMAX = -18.5, 21.0


def to01(raw):
    return ((raw - RMIN) / (RMAX - RMIN)).clip(
        0, 1
    )  # linear rescale of the summed points to 0..1 (values outside the bounds are clipped)


def rule_score(d):
    return to01(
        rule_scores(d).sum(axis=1)
    )  # 0 = strong evidence against impersonation, 1 = strong evidence for


def decide(score, hi, lo):
    return np.where(score >= hi, 1, np.where(score <= lo, 0, -1))  # -1 = unsure -> next stage
