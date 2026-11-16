"""Magnetic Tile defect dataset (github.com/abin24/Magnetic-tile-defect-datasets. @cc4eb53) to YOLO format.

Boxes are the bounding rectangles of 8-connected components of the binary defect masks (pixel > 127);
components smaller than MIN_AREA pixels are ignored (annotation specks). Defect-free images (MT_Free)
are kept with empty label files: they are the negatives needed for false-alarm control.
Split: stratified by folder class, 60/20/20; near-duplicate images (perceptual hash, audited in
scripts/audit_mt.py) are kept in the same partition.
"""
import glob
import hashlib
import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from .paths import DATA  # noqa: E402

SRC = f"{DATA}/raw/mt_repo"
FOLDERS = ["MT_Blowhole", "MT_Break", "MT_Crack", "MT_Fray", "MT_Uneven", "MT_Free"]
CLASSES = ["blowhole", "break", "crack", "fray", "uneven"]
MIN_AREA = 9


def items():
    out = []
    for fi, f in enumerate(FOLDERS):
        for p in sorted(glob.glob(f"{SRC}/{f}/Imgs/*.jpg")):
            out.append({"stem": f"{f[3:].lower()}_{os.path.basename(p)[:-4]}", "img": p, "mask": p[:-4] + ".png",
                        "folder": f, "cls": fi if fi < 5 else None})
    return out


def boxes(item):
    if item["cls"] is None:
        return []
    m = np.asarray(Image.open(item["mask"]).convert("L")) > 127
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    out = []
    for sl, k in zip(ndimage.find_objects(lab), range(1, n + 1)):
        if (lab[sl] == k).sum() < MIN_AREA:
            continue
        y1, y2, x1, x2 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        out.append((item["cls"], x1, y1, x2, y2))
    return out


def build(root, seed=0, groups=None, frac=(0.6, 0.2, 0.2)):
    """groups: dict stem -> representative stem (near-duplicates), from the audit."""
    groups = groups or {}
    rng = np.random.default_rng(seed)
    its = items()
    members = {}
    for i in its:
        members.setdefault(groups.get(i["stem"], i["stem"]), []).append(i)
    # stratum of a tile group: its most frequent defect folder (ties: first in FOLDERS order), or MT_Free if all
    # members are free. The tie-break used to depend on set iteration order, i.e. on PYTHONHASHSEED; the one
    # tied group (blowhole_exp1_num_9473: 3 Blowhole, 3 Crack) went to MT_Blowhole in the split used for all
    # runs, which this rule reproduces.
    strata = {}
    for r, ms in members.items():
        f = [m["folder"] for m in ms if m["folder"] != "MT_Free"]
        strata.setdefault(min(set(f), key=lambda x: (-f.count(x), FOLDERS.index(x))) if f else "MT_Free",
                          []).append(r)
    assign = {}
    for f in FOLDERS:
        reps = sorted(strata.get(f, []))
        rng.shuffle(reps)
        # assign whole groups, filling partitions to their image-count targets
        n_img = sum(len(members[r]) for r in reps)
        cum = 0
        for r in reps:
            q = cum / max(n_img, 1)
            assign[r] = "train" if q < frac[0] else "val" if q < frac[0] + frac[1] else "test"
            cum += len(members[r])
    lists = {"train": [], "val": [], "test": []}
    for s in lists:
        os.makedirs(f"{root}/images/{s}", exist_ok=True)
        os.makedirs(f"{root}/labels/{s}", exist_ok=True)
    stats = {s: {"images": 0, "defective": 0, "boxes": [0] * len(CLASSES)} for s in lists}
    for i in its:
        s = assign[groups.get(i["stem"], i["stem"])]
        W, H = Image.open(i["img"]).size
        dst = f"{root}/images/{s}/{i['stem']}.jpg"
        if not os.path.exists(dst):
            os.link(i["img"], dst)
        bx = boxes(i)
        open(f"{root}/labels/{s}/{i['stem']}.txt", "w").write(
            "".join(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n"
                    for c, x1, y1, x2, y2 in bx))
        lists[s].append(i["stem"])
        stats[s]["images"] += 1
        stats[s]["defective"] += int(i["cls"] is not None)
        for b in bx:
            stats[s]["boxes"][b[0]] += 1
    for s, L in lists.items():
        L.sort()
        stats[s]["sha256"] = hashlib.sha256("\n".join(L).encode()).hexdigest()
    open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                         + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
    json.dump({"seed": seed, "min_area": MIN_AREA, "groups": groups, "splits": stats}, open(f"{root}/split_meta.json", "w"), indent=1)
    return stats
