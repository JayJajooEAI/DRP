"""Build one day's intake feed: the 2,870 labelled rows mixed with unrelated new registrations (noise).

Writes
  data/intake.json        what the data sources know about each domain (no labels): read only by data_api.py
  data/intake_truth.json  domain -> row id, company, case, expected; read only by 03_evaluate.py

Noise domains are two common English words plus a TLD, with ordinary metadata. Their truth is "not relevant" for
every client; a few will share letters with a client and test Steps 1-4 the way real daily volume does.

    python code/01_build_intake.py [--noise 0]
"""
import json
import random
import sys
from pathlib import Path

from wordfreq import top_n_list

WORLD = Path(__file__).resolve().parents[1] / "data"
SEED = 20261027
TLDS = [".com"] * 12 + [".net", ".org", ".shop", ".online", ".xyz", ".info", ".co", ".io", ".store", ".site"]
HOSTING_NS = [("ns1.digitalocean.com", "ns2.digitalocean.com"), ("dns1.registrar-servers.com", "dns2.registrar-servers.com"),
              ("kate.ns.cloudflare.com", "rob.ns.cloudflare.com"), ("ns1.hostinger.com", "ns2.hostinger.com")]
PARKING_NS = [("ns1.sedoparking.com", "ns2.sedoparking.com"), ("ns1.bodis.com", "ns2.bodis.com")]
REGISTRARS = ["NameCheap, Inc.", "GoDaddy.com, LLC", "Porkbun LLC", "Tucows Domains Inc.", "IONOS SE", "Squarespace Domains II LLC"]
PAGES = ["business"] * 4 + ["blank", "coming_soon", "hosting_default", "generic_landing", "none", "personal"]


def noise_rows(n, taken, day):
    rng = random.Random(f"{SEED}-noise")
    words = [w for w in top_n_list("en", 8000) if w.isascii() and w.isalpha() and 3 <= len(w) <= 9]
    out = []
    while len(out) < n:
        a, b = rng.sample(words, 2)
        d = a + rng.choice(["", "", "-"]) + b + rng.choice(TLDS)
        if d in taken:
            continue
        taken.add(d)
        parked = rng.random() < 0.08
        kind = rng.choice(["parked", "parked_brand_ads"]) if parked else rng.choice(PAGES)
        status = None if kind == "none" else 200
        out.append({"domain": d, "domain_unicode": d,
                    "rdap": {"registrant_org": "REDACTED FOR PRIVACY" if rng.random() < 0.7 else
                             f"{a.title()} {b.title()} {rng.choice(['LLC', 'Ltd', 'Inc.', 'Studio'])}",
                             "registrant_country": rng.choice(["US", "GB", "CA", "DE", "IN", "AU"]),
                             "registrar": rng.choice(REGISTRARS), "created": day, "updated": day,
                             "expires": f"{int(day[:4]) + 1}{day[4:]}", "status": ["client transfer prohibited"]},
                    "dns": {"nameservers": list(rng.choice(PARKING_NS if parked else HOSTING_NS)), "mx": rng.random() < 0.3},
                    "http": {"status": status, "redirect_to": None, "final_url": f"https://{d}/" if status else None},
                    "cert": {"issuer": "Let's Encrypt", "not_before": day, "not_after": day} if status else None,
                    "page": {"kind": kind, "has_login_form": False}, "known_phish": False})
    return out


def main():
    n = int(sys.argv[sys.argv.index("--noise") + 1]) if "--noise" in sys.argv else 0
    data = json.loads((WORLD / "domains.json").read_text())
    rows, day = data["rows"], data["world_date"]
    feed, truth = [], {}
    for r in rows:
        m = r["metadata"]
        feed.append({"domain": r["domain"], "domain_unicode": r["domain_unicode"], **{k: m[k] for k in
                     ("rdap", "dns", "http", "cert", "page", "known_phish")}})
        truth.setdefault(r["domain"], []).append({k: r[k] for k in ("id", "company_id", "split", "case", "variant",
                                                                     "group", "label", "incident_type", "tags",
                                                                     "expected")})
    dupes = [d for d, t in truth.items() if len(t) > 1]
    if dupes:
        sys.exit(f"domain used by more than one row: {dupes[:5]}")
    noise = noise_rows(n, set(truth), day)
    feed += noise
    for x in noise:
        truth[x["domain"]] = [{"id": "noise-" + x["domain"], "company_id": None, "split": "noise", "case": "NOISE",
                               "variant": "noise", "group": "noise", "label": "not_relevant", "incident_type": None,
                               "tags": [], "expected": {"stop_step": "1", "outcome": "not relevant"}}]
    random.Random(f"{SEED}-order").shuffle(feed)
    (WORLD / "intake.json").write_text(json.dumps({"day": day, "domains": feed}, ensure_ascii=False))
    (WORLD / "intake_truth.json").write_text(json.dumps(truth, ensure_ascii=False))
    print(f"intake {day}: {len(feed):,} domains ({len(rows):,} labelled rows + {len(noise):,} noise)")


if __name__ == "__main__":
    main()
