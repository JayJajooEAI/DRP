"""Everything the service reads from RDS (schema drp_clients). Nothing is cached: every request reads what it needs.

    customers + pipeline_metadata   the client exists and is used by the web-phish pipeline ('webp'); its display name
    customer_domains                the client's own domains (check 4), user-content domains and owned TLDs
    brand_terms, customer_domains   all clients' brand names and own domains (two of the URL model's features)
    assets + webp_logo_embeddings   the client's logo images (the gallery the page's logos are compared with)
    content + page_captures         the stored pages for a link (a link can have several captures): the HTML and the screenshot
"""

import io

import psycopg
from PIL import Image

from WebPhish.settings import RDS, RDS_SCHEMA


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
                   FROM customers c JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'webp'
                   ORDER BY c.customer_id"""
            ).fetchall()
        return [dict(id=a, name=b, domain=c) for a, b, c in rows]

    def get(self, client_id):
        """The profile of one client, or None if it is not a web-phish client."""
        with self._connect() as con:
            row = con.execute(
                """SELECT c.customer_id, c.display_name FROM customers c
                   JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'webp' WHERE c.customer_id = %s""",
                (client_id,),
            ).fetchone()
            if row is None:
                return None
            doms = {}
            for kind, d in con.execute("SELECT domain_type, domain FROM customer_domains WHERE customer_id = %s", (client_id,)):
                doms.setdefault(kind, set()).add(d.lower())
            logos = []
            for (data,) in con.execute(
                """SELECT a.data FROM assets a JOIN webp_logo_embeddings w ON w.asset_id = a.asset_id
                   WHERE a.customer_id = %s ORDER BY a.asset_id""",
                (client_id,),
            ):
                try:
                    im = Image.open(io.BytesIO(bytes(data)))
                    im.load()
                    logos.append(im)
                except Exception:   # an unreadable logo is skipped (the batch pipeline does the same)
                    continue
        return dict(
            id=row[0],
            name=row[1],
            allowlist=doms.get("allowlist", set()),
            user_content=doms.get("user_content", set()),
            owned_tlds=doms.get("owned_tld", set()),
            logos=logos,
        )

    def brand_context(self):
        """(aliases, owned domains) of all web-phish clients: the two lists the URL model's brand-in-host feature uses.
        A client's owned domains are its `allowlist` rows (its `main` row is just one of them)."""
        with self._connect() as con:
            aliases = [t for (t,) in con.execute(
                """SELECT b.term FROM brand_terms b JOIN pipeline_metadata p ON p.customer_id = b.customer_id AND p.pipeline = 'webp'
                   WHERE b.kind = 'alias'""")]
            owned = [d.lower() for (d,) in con.execute(
                """SELECT d.domain FROM customer_domains d JOIN pipeline_metadata p ON p.customer_id = d.customer_id AND p.pipeline = 'webp'
                   WHERE d.domain_type = 'allowlist'""")]
        return aliases, owned

    def captures(self, client_id, url):
        """The stored pages of a link: every capture of that link for this client (ordered by content_id). Each is
        dict(content_id, url, final_url, html, screenshot). Empty list when there is none."""
        with self._connect() as con:
            rows = con.execute(
                """SELECT p.content_id, p.url, p.final_url, p.html, p.screenshot
                   FROM page_captures p JOIN content c ON c.content_id = p.content_id
                   WHERE c.customer_id = %s AND c.pipeline = 'webp' AND p.url = %s ORDER BY p.content_id""",
                (client_id, url),
            ).fetchall()
        return [dict(content_id=cid, url=u, final_url=fu or u, html=bytes(html or b""), screenshot=bytes(shot or b""))
                for cid, u, fu, html, shot in rows]
