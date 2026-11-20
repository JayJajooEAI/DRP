"""Step 5: legitimate third parties (PS doc, "Leverage Incident Types" clues; plan Step 8b). Rules only, no model.

Runs on domains already found relevant, before the parked label is chosen. Clears a domain as None when:

    1  Tranco shortcut   it has a Tranco rank, and its WHOIS owner is named and is not the client
    2  legitimacy clues  it was flagged because of a NAME (brand + extra word, product or executive name), and
                         its WHOIS owner is named and is not the client, and the page shows its own business name,
                         and the page has a contact page
    safeguards           real typos (misspelling, lookalike characters, other ending, country-code abuse, plural,
                         missing dot) are never cleared by rule 2: the doc's first clue for a legitimate site is
                         "no typosquatting"; nor is an executive's full name (a site in a person's name is the
                         doc's own example of a relevant match, iamfoster.com)

The page facts used by rule 2 (own business name, contact page) are generated in this dataset; in production they
must be read from the real page.

Not being on the Tranco list proves nothing (doc). Privacy page, client logo and links to the client are not used:
they are weak or point both ways (franchisees show the client's logo; phishing pages link to the client).
"""
from importlib import import_module

O = import_module("ownership")

NAME_REASONS = {"brand + extra word", "string match"}       # not "executive full name": see the safeguards


def named_other_owner(rdap, profile):
    org = (rdap or {}).get("registrant_org") or ""
    if not org or O.REDACTED.search(org):
        return None
    if O.norm_org(org) == O.norm_org(profile["registrant_org"]):
        return None
    return org


def check(x, flagged_by):
    """flagged_by: the typosquat kind, or 'string match'. -> (rule, reason) when legitimate, else None."""
    owner = named_other_owner(x["rdap"], x["profile"])
    if x["tranco_rank"] and owner:
        return "1 Tranco shortcut", f"Tranco #{x['tranco_rank']:,}, owner {owner}"
    page = x["page"] or {}
    if (flagged_by in NAME_REASONS and owner and page.get("shows_own_business_name")
            and page.get("has_contact_page")):
        return "2 legitimacy clues", f"named owner {owner}, own business name, contact page"
    return None
