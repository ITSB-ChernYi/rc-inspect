"""Group Magnetic Tile images that show the same physical tile under different exposures.

Two images are linked when their aspect ratios differ by less than 10% and the Pearson correlation
of their standardised 64x64 grey-level thumbnails exceeds THR (the correlation distribution of all
pairs is reported for comparison; exposures of one tile differ in brightness, not in structure). Groups are connected components; every group stays inside one split partition.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import mt  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "audit")
its = mt.items()
Z, size = {}, {}
for i in its:
    im = Image.open(i["img"]).convert("L")
    a = np.asarray(im.resize((64, 64)), float)
    Z[i["stem"]] = ((a - a.mean()) / (a.std() + 1e-6)).ravel() / 64.0
    size[i["stem"]] = im.size
THR = 0.95
parent = {s: s for s in Z}


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


L = list(Z)
M = np.stack([Z[s] for s in L])
C = M @ M.T
ar = np.array([size[s][0] / size[s][1] for s in L])
okar = np.abs(np.log(ar[:, None] / ar[None, :])) < np.log(1.1)
iu = np.triu_indices(len(L), 1)
corrs = C[iu]
links = 0
for a, b in zip(*np.nonzero(np.triu((C > THR) & okar, 1))):
    links += 1
    ra, rb = find(L[a]), find(L[b])
    parent[max(ra, rb)] = min(ra, rb)
groups = {s: find(s) for s in Z if find(s) != s}
comp = defaultdict(list)
for s in Z:
    comp[find(s)].append(s)
sizes = np.array([len(v) for v in comp.values()])
folders = [len({x.split("_")[0] for x in v}) for v in comp.values()]
rep = {"pairs": int(len(corrs)), "corr_quantiles_all_pairs": np.quantile(corrs, [.5, .9, .99, .999]).round(3).tolist(),
       "threshold": THR, "links": links, "n_groups": int(len(comp)), "group_size_counts": {int(k): int(v) for k, v in zip(*np.unique(sizes, return_counts=True))},
       "groups_spanning_two_folders": int(sum(f > 1 for f in folders))}
json.dump({"report": rep, "groups": groups}, open(f"{OUT}/mt_tile_groups.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
