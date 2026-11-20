"""Step 6: parked check (PS doc, "Parked Domains"; plan Step 5). Nameserver and HTTP facts only, no page loading.

Only decides the label of a domain already found relevant: Domains - Parked instead of Suspicious Domain
Registration. It never makes a domain more or less relevant, and brand words in parking ads are not a content match.

    parked when  the nameservers belong to a parking service, or
                 the HTTP response is a known parking / for-sale template
"""
PARKING_NS = ("sedoparking.com", "bodis.com", "parkingcrew.net", "above.com", "parklogic.com", "dan.com",
              "afternic.com", "uniregistry-dns.com", "namebrightdns.com", "hugedomains.com")


def check(nameservers, http):
    """-> reason when the domain is parked, else None."""
    ns = [n.lower().rstrip(".") for n in nameservers or []]
    hit = next((n for n in ns if any(n == p or n.endswith("." + p) for p in PARKING_NS)), None)
    if hit:
        return f"parking nameserver {hit}"
    if (http or {}).get("body_is_parking_template"):
        return "response is a parking / for-sale page"
    return None
