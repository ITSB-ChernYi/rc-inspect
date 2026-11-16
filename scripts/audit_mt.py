"""Audit Magnetic Tile: sizes, mask-derived boxes, exact and near-duplicate images (perceptual hash)."""
import hashlib
import json
import os
import sys

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import mt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "audit")
its = mt.items()
rows, B = [], []
for i in its:
    im = Image.open(i["img"]).convert("L")
    rows.append({"stem": i["stem"], "folder": i["folder"], "w": im.size[0], "h": im.size[1],
                 "md5": hashlib.md5(np.asarray(im).tobytes()).hexdigest(),
                 "phash": str(imagehash.phash(im.resize((128, 128)), hash_size=16))})
    for b in mt.boxes(i):
        B.append({"stem": i["stem"], "cls": mt.CLASSES[b[0]], "w": b[3] - b[1], "h": b[4] - b[2]})
M, B = pd.DataFrame(rows), pd.DataFrame(B)
H = np.array([[int(c, 16) for c in h] for h in M.phash], dtype=np.uint8)
bits = np.unpackbits(H[:, :, None], axis=2)[:, :, 4:].reshape(len(M), -1)
D = (bits[:, None, :] != bits[None, :, :]).sum(-1)
iu = np.triu_indices(len(M), 1)
sel = D[iu] <= 20
ND = pd.DataFrame({"a": M.stem.values[iu[0][sel]], "b": M.stem.values[iu[1][sel]], "hamming": D[iu][sel]})
ND.to_csv(f"{OUT}/mt_near_duplicates.csv", index=False)
groups = {}
for a, b in zip(ND.a, ND.b):  # union into the alphabetically first representative
    ra, rb = groups.get(a, a), groups.get(b, b)
    r = min(ra, rb)
    for k, v in list(groups.items()):
        if v in (ra, rb):
            groups[k] = r
    groups[a], groups[b] = r, r
groups = {k: v for k, v in groups.items() if k != v}
rep = {"n_images": len(M), "per_folder": M.folder.value_counts().to_dict(),
       "defective_images_without_box": int(len(set(M[M.folder != "MT_Free"].stem) - set(B.stem))),
       "boxes_per_class": B.cls.value_counts().to_dict(), "boxes_per_defective_image_mean": float(B.groupby("stem").size().mean()),
       "box_side_quantiles": np.sqrt(B.w * B.h).quantile([0, .05, .5, .95, 1]).round(1).to_dict(),
       "image_w_range": [int(M.w.min()), int(M.w.max())], "image_h_range": [int(M.h.min()), int(M.h.max())],
       "exact_duplicates": int(M.duplicated("md5", keep=False).sum()), "near_dup_pairs_le20": len(ND),
       "near_dup_cross_folder_pairs": int(sum(M.set_index("stem").folder[a] != M.set_index("stem").folder[b] for a, b in zip(ND.a, ND.b))),
       "groups": groups}
json.dump(rep, open(f"{OUT}/mt_audit.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "groups"}, indent=1), "n_grouped", len(groups))
