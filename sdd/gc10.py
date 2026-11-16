"""GC10-DET (github.com/wyx19980727/GC10-DET-VOC @2459a71) to YOLO format.

Decisions (documented in results/audit/gc10_audit.json and the manuscript):
  * 26 images without an annotation file are excluded;
  * the single box labelled 'd' (unknown class) is dropped;
  * 12 original images stored in two class folders are merged into one item whose boxes are the
    union of both annotation files (identical boxes removed);
  * XML labels are authoritative (73 images whose folder class is absent from the XML keep their XML);
  * split by acquisition sequence (img_<cam>_<seq>): all frames of a sequence fall in the same
    partition; sequences are assigned greedily to keep per-class box shares close to 60/20/20.
"""
import hashlib
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np

from .paths import DATA  # noqa: E402

SRC = f"{DATA}/raw/gc10_repo/VOC2007"
CLASSES = ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", "silk_spot",
           "inclusion", "rolled_pit", "crease", "waist_folding"]
NUM = {str(i + 1): c for i, c in enumerate(CLASSES)}


def parse(stem):
    r = ET.parse(f"{SRC}/Annotations/{stem}.xml").getroot()
    out = []
    for o in r.findall("object"):
        n = o.find("name").text.strip().split("_")[0]
        if n not in NUM:
            continue
        b = o.find("bndbox")
        out.append((CLASSES.index(NUM[n]),) + tuple(float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")))
    return out


def items():
    """-> dict orig_name -> {'files': [stems], 'boxes': [...], 'seq': str}"""
    it = {}
    for f in sorted(os.listdir(f"{SRC}/JPEGImages")):
        stem = f[:-4]
        m = re.match(r"(.+?)_(img_(\d+)_([A-Za-z]*\d+)_(\d+))$", stem)
        orig, seq = m.group(2), m.group(3) + "_" + m.group(4)
        if not os.path.exists(f"{SRC}/Annotations/{stem}.xml"):
            continue
        d = it.setdefault(orig, {"files": [], "boxes": [], "seq": seq})
        d["files"].append(stem)
        for b in parse(stem):
            if b not in d["boxes"]:
                d["boxes"].append(b)
    return {k: v for k, v in it.items() if v["boxes"]}


def grouped_split(it, seed, frac=(0.6, 0.2, 0.2)):
    rng = np.random.default_rng(seed)
    seqs = defaultdict(list)
    for k, v in it.items():
        seqs[v["seq"]].append(k)
    order = list(seqs)
    rng.shuffle(order)
    # largest sequences first so they do not unbalance the end of the assignment
    order.sort(key=lambda s: -len(seqs[s]))
    target = np.array(frac)
    tot = np.zeros(len(CLASSES))
    for v in it.values():
        for b in v["boxes"]:
            tot[b[0]] += 1
    cnt = np.zeros((3, len(CLASSES)))
    assign = {}
    for s in order:
        h = np.zeros(len(CLASSES))
        for k in seqs[s]:
            for b in it[k]["boxes"]:
                h[b[0]] += 1
        # choose the partition whose per-class deficit (relative to target) is largest for this histogram
        deficit = [(target[p] * tot - cnt[p]) / np.maximum(tot, 1) for p in range(3)]
        score = [float((deficit[p] * h).sum()) + 1e-3 * rng.random() for p in range(3)]
        p = int(np.argmax(score))
        cnt[p] += h
        for k in seqs[s]:
            assign[k] = ("train", "val", "test")[p]
    return assign


def random_split(it, seed, frac=(0.6, 0.2, 0.2)):
    """Image-level random split (the common protocol), for the leakage comparison only."""
    rng = np.random.default_rng(seed)
    keys = sorted(it)
    rng.shuffle(keys)
    a, b = int(frac[0] * len(keys)), int((frac[0] + frac[1]) * len(keys))
    return {k: ("train" if i < a else "val" if i < b else "test") for i, k in enumerate(keys)}


def build(root, seed=0, mode="grouped"):
    it = items()
    assign = grouped_split(it, seed) if mode == "grouped" else random_split(it, seed)
    lists = {"train": [], "val": [], "test": []}
    for s in lists:
        os.makedirs(f"{root}/images/{s}", exist_ok=True)
        os.makedirs(f"{root}/labels/{s}", exist_ok=True)
    for k, v in sorted(it.items()):
        s = assign[k]
        dst = f"{root}/images/{s}/{k}.jpg"
        if not os.path.exists(dst):
            os.link(f"{SRC}/JPEGImages/{v['files'][0]}.jpg", dst)
        L = [f"{c} {(x1 + x2) / 2 / 2048:.6f} {(y1 + y2) / 2 / 1000:.6f} {(x2 - x1) / 2048:.6f} {(y2 - y1) / 1000:.6f}"
             for c, x1, y1, x2, y2 in v["boxes"]]
        open(f"{root}/labels/{s}/{k}.txt", "w").write("\n".join(L) + "\n")
        lists[s].append(k)
    meta = {"mode": mode, "seed": seed, "n_items": len(it), "splits": {}}
    for s, L in lists.items():
        L.sort()
        seqs = {it[k]["seq"] for k in L}
        boxes = np.bincount([b[0] for k in L for b in it[k]["boxes"]], minlength=len(CLASSES)).tolist()
        meta["splits"][s] = {"n": len(L), "n_sequences": len(seqs), "boxes_per_class": boxes,
                             "sha256": hashlib.sha256("\n".join(L).encode()).hexdigest()}
        open(f"{root}/{s}.txt", "w").write("\n".join(f"{root}/images/{s}/{x}.jpg" for x in L) + "\n")
    if mode == "grouped":
        tr = {it[k]["seq"] for k in lists["train"]}
        meta["sequence_overlap_train_test"] = len(tr & {it[k]["seq"] for k in lists["test"]})
    else:
        tr = {it[k]["seq"] for k in lists["train"]}
        te = [k for k in lists["test"] if it[k]["seq"] in tr]
        meta["test_images_sharing_sequence_with_train"] = len(te)
    open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                                         + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    json.dump(meta, open(f"{root}/split_meta.json", "w"), indent=1)
    return meta
