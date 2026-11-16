"""Client profiles, fetched from RDS (schema drp_clients): what the protected client supplies and what was precomputed for it.

For one client id the service reads
    customers + pipeline_metadata   the client exists and is used by the impersonation pipeline; brand name
    customer_domains                the official (main) domain
    executives                      protected executives: name and title
    impersonation_embeddings        the stored CLIP vectors of the logo (2 sizes) and of each executive photo
    assets                          the logo image and the executive photos (needed for the pixel-correlation check and for Luna)
Nothing is cached: every request reads the client from RDS, so a change in the database applies to the next request."""

import io

import numpy as np
import psycopg
from PIL import Image

from impersonation.settings import RDS, RDS_SCHEMA


class RdsUnavailable(RuntimeError):
    """RDS is not configured or cannot be reached."""


def _img(data):
    """bytes -> PIL image (fully loaded, so the connection data is not kept alive)."""
    im = Image.open(io.BytesIO(bytes(data)))
    im.load()
    return im


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
        """Every client the impersonation pipeline can check: [{id, name, brand, domain}]."""
        with self._connect() as con:
            cur = con.execute(
                """SELECT c.customer_id, c.display_name, c.brand,
                          (SELECT domain FROM customer_domains d WHERE d.customer_id = c.customer_id AND d.domain_type = 'main' LIMIT 1)
                   FROM customers c JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'impersonation'
                   ORDER BY c.customer_id"""
            )
            return [dict(id=a, name=b, brand=c, domain=d) for a, b, c, d in cur.fetchall()]

    def get(self, client_id):
        """The full profile of one client, or None if it is not an impersonation client."""
        return self._fetch(client_id)

    def _fetch(self, cid):
        with self._connect() as con:
            row = con.execute(
                """SELECT c.customer_id, c.display_name, c.brand FROM customers c
                   JOIN pipeline_metadata p ON p.customer_id = c.customer_id AND p.pipeline = 'impersonation' WHERE c.customer_id = %s""",
                (cid,),
            ).fetchone()
            if row is None:
                return None
            dom = con.execute("SELECT domain FROM customer_domains WHERE customer_id = %s AND domain_type = 'main' LIMIT 1", (cid,)).fetchone()
            execs = con.execute("SELECT executive_id, name, title FROM executives WHERE customer_id = %s ORDER BY executive_id", (cid,)).fetchall()

            # the logo: its image and its two stored CLIP vectors (full size and 78 %)
            logo = con.execute(
                """SELECT a.data, i.variant, i.embedding FROM impersonation_embeddings i JOIN assets a USING (asset_id)
                   WHERE i.customer_id = %s AND i.kind = 'logo' ORDER BY i.variant""",
                (cid,),
            ).fetchall()
            if not logo:
                raise RdsUnavailable(f"client {cid} has no stored logo embedding")

            # the executive photos: image + stored CLIP vector, per executive
            photos = con.execute(
                """SELECT i.executive_id, a.data, i.embedding FROM impersonation_embeddings i JOIN assets a USING (asset_id)
                   WHERE i.customer_id = %s AND i.kind = 'executive_photo' ORDER BY i.executive_id""",
                (cid,),
            ).fetchall()

        return dict(
            id=row[0], name=row[1], brand=row[2], domain=dom[0] if dom else "",
            execs=[dict(name=n, role=t) for _, n, t in execs],
            logo_image=_img(logo[0][0]),
            logo_vecs=np.array([e for _, _, e in logo], dtype=np.float32),
            photos=[(_img(d).convert("RGB"), np.array(e, dtype=np.float32)) for _, d, e in photos],
        )
