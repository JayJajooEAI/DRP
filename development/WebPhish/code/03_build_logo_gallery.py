"""Step 03: embed every customer logo once (the profiles' empty `logo_gallery`) with Phishpedia's
siamese logo matcher.

    python code/03_build_logo_gallery.py

Reads  data/customer_assets/*/profile.json -> "logos" (463 images for the 25 customers).
Writes outputs/03_logo_similarity/logo_gallery.npz:
       emb (n_logos x 2048, L2-normalised), customer_id, corporate_group, path, plus the embedder id.
"""
import json
import time

import numpy as np
from PIL import Image

from logo_common import EMBEDDER_ID, GALLERY, OUT, ROOT, Embedder, load_profiles, to_rgb


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    profiles = load_profiles()
    images, customers, groups, paths, skipped = [], [], [], [], []
    for cid, prof in profiles.items():
        for rel in prof.get("logos") or []:
            try:
                images.append(to_rgb(Image.open(ROOT / rel)))
            except Exception as exc:   # unreadable file: report it, keep going
                skipped.append(f"{rel}: {type(exc).__name__}")
                continue
            customers.append(cid)
            groups.append(prof.get("corporate_group") or cid)
            paths.append(rel)

    t0 = time.time()
    emb = Embedder(device)(images)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(GALLERY, emb=emb.astype(np.float32), customer_id=np.array(customers), corporate_group=np.array(groups),
             path=np.array(paths), embedder=EMBEDDER_ID)
    info = {"logos": len(paths), "customers": len(set(customers)), "skipped": skipped, "dim": int(emb.shape[1]),
            "embedder": EMBEDDER_ID, "device": device, "seconds": round(time.time() - t0, 1)}
    (OUT / "gallery_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
