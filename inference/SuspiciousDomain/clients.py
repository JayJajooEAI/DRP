"""Everything the service reads from RDS (schema drp_clients). Nothing is cached: every request reads what it needs.

    customers + pipeline_metadata   the client exists and is used by the suspicious-domain pipeline ('susdom'); sector, languages
    customer_domains                the client's owned domains: main, regional, defensive
    domain_ownership                the registrant organisation the client registers its domains under
    brand_terms, products           the client's names (full brand, brand parts, product names)
    executives, third_parties       protected executives; partners, resellers and franchisees with their domains
    domain_lookups                  the stored facts of a domain: WHOIS, nameservers, HTTP, page facts
    tranco_ranks                    rank in the top-sites list
"""

import psycopg

from SuspiciousDomain.settings import RDS, RDS_SCHEMA


class RdsUnavailable(Exception):
    """RDS is not configured or cannot be reached."""


class RdsClients:
    def __init__(self):
        missing = [k for k in ("host", "user", "password") if not RDS[k]]
        if missing:
            raise RdsUnavailable("set the environment variables " + ", ".join("RDS_" + k.upper() for k in missing))

    def _connect(self):
        try:
            return psycopg.connect(**RDS, connect_timeout=10, options=f"-c search_path={RDS_SCHEMA}")
        except psycopg.Error as e:
            raise RdsUnavailable(f"cannot connect to RDS ({type(e).__name__})")

    def ids(self):
        """The clients this pipeline protects."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT c.customer_id, c.display_name,
                          (SELECT domain FROM customer_domains d WHERE d.customer_id = c.customer_id AND d.domain_type = 'main' LIMIT 1)
                   FROM customers c JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'susdom'
                   ORDER BY c.customer_id"""
            ).fetchall()
        return [dict(id=a, name=b, domain=c) for a, b, c in rows]

    def profile(self, client_id):
        """The client's profile in the shape the detector's steps expect, or None if it is not a suspicious-domain client."""
        with self._connect() as con:
            row = con.execute(
                """SELECT c.customer_id, c.display_name, c.sector, c.languages FROM customers c
                   JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'susdom' WHERE c.customer_id = %s""",
                (client_id,),
            ).fetchone()
            if row is None:
                return None
            owned = {"main": [], "regional": [], "defensive_listed": []}
            for kind, d in con.execute("SELECT domain_type, domain FROM customer_domains WHERE customer_id = %s ORDER BY domain_id", (client_id,)):
                if kind in ("main", "regional", "defensive"):
                    owned["defensive_listed" if kind == "defensive" else kind].append(d)
            own = con.execute("SELECT registrant_org FROM domain_ownership WHERE customer_id = %s", (client_id,)).fetchone()
            return dict(
                id=row[0], display_name=row[1], sector=row[2], languages=list(row[3] or []),
                brand_terms=[dict(term=t, kind=k, common_word=c) for t, k, c in con.execute(
                    "SELECT term, kind, is_common_word FROM brand_terms WHERE customer_id = %s ORDER BY term_id", (client_id,))],
                owned_domains=owned,
                products=[n for (n,) in con.execute("SELECT name FROM products WHERE customer_id = %s ORDER BY product_id", (client_id,))],
                executives=[dict(name=n, title=t) for n, t in con.execute(
                    "SELECT name, title FROM executives WHERE customer_id = %s ORDER BY executive_id", (client_id,))],
                registrant_org=own[0] if own else "",
                known_third_parties=[dict(domain=d, org=o, kind=k) for d, o, k in con.execute(
                    "SELECT domain, org, kind FROM third_parties WHERE customer_id = %s ORDER BY third_party_id", (client_id,))],
            )

    def domain(self, domain):
        """The stored lookup of a domain (what WHOIS, DNS, HTTP and a headless browser said), or None if there is none.
        Returns the pieces the detector reads: rdap, nameservers, http, page (None when nothing answered), tranco_rank."""
        with self._connect() as con:
            r = con.execute(
                """SELECT domain, registrant_org, nameservers, http_status, redirect_to, final_url, parking_template,
                          has_password_input, links_to_client_site, shows_own_business_name, has_contact_page,
                          has_privacy_page, shows_client_logo
                   FROM domain_lookups WHERE domain = %s OR domain_unicode = %s LIMIT 1""",
                (domain, domain),
            ).fetchone()
            if r is None:
                return None
            rank = con.execute("SELECT rank FROM tranco_ranks WHERE domain = %s", (r[0],)).fetchone()
        dom, org, ns, status, redirect, final, parking, pw, links, own, contact, privacy, logo = r
        page = None if status is None else dict(
            status=status, final_url=final, has_password_input=bool(pw), links_to_client_site=links, shows_own_business_name=own,
            has_contact_page=contact, has_privacy_page=privacy, shows_client_logo=logo)
        return dict(domain=dom, rdap={"registrant_org": org}, nameservers=list(ns or []),
                    http=dict(status=status, redirect_to=redirect, final_url=final, body_is_parking_template=bool(parking)),
                    page=page, tranco_rank=rank[0] if rank else None)
