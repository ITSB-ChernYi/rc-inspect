"""DeepPCB (github.com/tangsanli5201/DeepPCB @08e98c4) to YOLO format.

Decisions (documented in results/audit/deeppcb_audit.json and the manuscript):
  * only the tested images are used (the defect-free templates are not given to the detector);
  * all 1,500 tested images have a valid annotation file; no image or box is excluded;
  * split by board group (the 11 groupXXXXX folders, one printed-circuit design each): every crop of a
    design falls in one partition, so the test designs are never seen in training. The official
    trainval/test lists place 7 of the 11 designs on both sides;
  * groups are assigned greedily to keep per-class box shares close to 60/20/20 (same rule as GC10-DET);
  * 'official' mode reproduces the official lists (the last 20 % of trainval, by sorted name within each
    group, becomes the validation split), for the leakage comparison only.
"""
import glob
import hashlib
import json
import os

import numpy as np

from .paths import DATA  # noqa: E402

SRC = f"{DATA}/raw/deeppcb_repo/PCBData"
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]
S = 640.0


def items():
    """-> dict stem -> {'path': tested image, 'boxes': [(cls, x1, y1, x2, y2)], 'group': str}"""
    it = {}
    for p in sorted(glob.glob(f"{SRC}/group*/*/*_test.jpg")):
        p = p.replace("\\", "/")
        stem, group = os.path.basename(p)[:-9], p.split("/")[-3]
        boxes = []
        for line in open(f"{SRC}/{group}/{group[5:]}_not/{stem}.txt"):
            v = line.replace(",", " ").split()
            if len(v) == 5:
                x1, y1, x2, y2, c = (int(float(u)) for u in v)
                boxes.append((c - 1, x1, y1, x2, y2))
        it[stem] = {"path": p, "boxes": boxes, "group": group}
    return it


def grouped_split(it, seed, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    groups = {}
    for k, v in it.items():
        groups.setdefault(v["group"], []).append(k)
    order = sorted(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups[g]))          # largest designs first
    target = np.array(frac)
    hist = {g: np.bincount([b[0] for k in groups[g] for b in it[k]["boxes"]], minlength=len(CLASSES)).astype(float)
            for g in groups}
    tot = sum(hist.values())
    cnt = np.zeros((3, len(CLASSES)))
    assign = {}
    for g in order:
        deficit = [(target[p] * tot - cnt[p]) / np.maximum(tot, 1) for p in range(3)]
        score = [float((deficit[p] * hist[g]).sum()) + 1e-3 * rng.random() for p in range(3)]
        p = int(np.argmax(score))
        cnt[p] += hist[g]
        for k in groups[g]:
            assign[k] = ("train", "val", "test")[p]
    return assign


def official_split(it):
    off = {}
    for split, f in (("trainval", "trainval.txt"), ("test", "test.txt")):
        for line in open(f"{SRC}/{f}"):
            if line.strip():
                off[os.path.basename(line.split()[0])[:-4]] = split
    assign = {k: "test" for k in it if off[k] == "test"}
    by_group = {}
    for k in sorted(it):
        if off[k] == "trainval":
            by_group.setdefault(it[k]["group"], []).append(k)
    for g, L in by_group.items():
        n_val = int(round(0.2 * len(L)))
        for i, k in enumerate(L):
            assign[k] = "val" if i >= len(L) - n_val else "train"
    return assign


def build(root, seed=0, mode="grouped"):
    # links of an earlier build are never removed, so a second build would leave images in two partitions
    assert not os.path.exists(f"{root}/images"), f"{root} already holds a split; delete it before rebuilding"
    it = items()
    assign = grouped_split(it, seed) if mode == "grouped" else official_split(it)
    lists = {"train": [], "val": [], "test": []}
    for s in lists:
        os.makedirs(f"{root}/images/{s}", exist_ok=True)
        os.makedirs(f"{root}/labels/{s}", exist_ok=True)
    for k, v in sorted(it.items()):
        s = assign[k]
        dst = f"{root}/images/{s}/{k}.jpg"
        if not os.path.exists(dst):
            os.link(v["path"], dst)
        L = [f"{c} {(x1 + x2) / 2 / S:.6f} {(y1 + y2) / 2 / S:.6f} {(x2 - x1) / S:.6f} {(y2 - y1) / S:.6f}"
             for c, x1, y1, x2, y2 in v["boxes"]]
        open(f"{root}/labels/{s}/{k}.txt", "w").write("\n".join(L) + "\n")
        lists[s].append(k)
    meta = {"mode": mode, "seed": seed, "n_items": len(it), "splits": {}}
    for s, L in lists.items():
        L.sort()
        boxes = np.bincount([b[0] for k in L for b in it[k]["boxes"]], minlength=len(CLASSES)).tolist()
        meta["splits"][s] = {"n": len(L), "groups": sorted({it[k]["group"] for k in L}), "boxes_per_class": boxes,
                             "sha256": hashlib.sha256("\n".join(L).encode()).hexdigest()}
        open(f"{root}/{s}.txt", "w").write("\n".join(f"{root}/images/{s}/{x}.jpg" for x in L) + "\n")
    tr = set(meta["splits"]["train"]["groups"]) | set(meta["splits"]["val"]["groups"])
    meta["test_images_from_designs_seen_in_train_or_val"] = sum(it[k]["group"] in tr for k in lists["test"])
    open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                         + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    json.dump(meta, open(f"{root}/split_meta.json", "w"), indent=1)
    return meta
