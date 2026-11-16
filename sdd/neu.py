"""NEU-DET conversion to YOLO format with leakage-safe, stratified, hashed splits."""
import glob
import hashlib
import json
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
from .paths import DATA  # noqa: E402

SRC = f"{DATA}/raw/neu_repo"
# exact duplicate pair found by scripts/audit_neu.py; kept together on one side of every split
GROUPS = {"patches_105": "patches_101"}


def items():
    out = []
    for p in sorted(glob.glob(f"{SRC}/IMAGES/*.jpg") + glob.glob(f"{SRC}/Validation_Images/*.jpg")):
        stem = os.path.basename(p)[:-4]
        a = p.replace("IMAGES", "ANNOTATIONS").replace("Validation_Images", "Validation_Annotations")[:-4] + ".xml"
        out.append((stem, p, a))
    return out


def yolo_lines(xml_path):
    r = ET.parse(xml_path).getroot()
    W, H = int(r.find("size/width").text), int(r.find("size/height").text)
    lines = []
    for o in r.findall("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        c = CLASSES.index(o.find("name").text.strip())
        lines.append(f"{c} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def make_split(seed, frac=(0.6, 0.2, 0.2)):
    """Stratified by image class; duplicate groups assigned together."""
    rng = np.random.default_rng(seed)
    its = items()
    split = {}
    for c in CLASSES:
        stems = sorted({GROUPS.get(s, s) for s, _, _ in its if s.rsplit("_", 1)[0] == c})
        rng.shuffle(stems)
        n = len(stems)
        a, b = int(round(frac[0] * n)), int(round((frac[0] + frac[1]) * n))
        for s in stems[:a]:
            split[s] = "train"
        for s in stems[a:b]:
            split[s] = "val"
        for s in stems[b:]:
            split[s] = "test"
    for child, parent in GROUPS.items():
        split[child] = split[parent]
    return split


def build(root, seed):
    """Write <root>/images/{train,val,test}, labels/..., data.yaml and split lists with hashes."""
    split = make_split(seed)
    for sub in ("images", "labels"):
        for s in ("train", "val", "test"):
            os.makedirs(f"{root}/{sub}/{s}", exist_ok=True)
    lists = {"train": [], "val": [], "test": []}
    for stem, img, xml in items():
        s = split[stem]
        dst = f"{root}/images/{s}/{stem}.jpg"
        if not os.path.exists(dst):
            shutil.copy(img, dst)
        open(f"{root}/labels/{s}/{stem}.txt", "w").write("\n".join(yolo_lines(xml)) + "\n")
        lists[s].append(stem)
    meta = {}
    for s, L in lists.items():
        L.sort()
        open(f"{root}/{s}.txt", "w").write("\n".join(f"{root}/images/{s}/{x}.jpg" for x in L) + "\n")
        meta[s] = {"n": len(L), "sha256": hashlib.sha256("\n".join(L).encode()).hexdigest()}
    open(f"{root}/data.yaml", "w").write(
        f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" +
        "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    json.dump({"seed": seed, "fractions": [0.6, 0.2, 0.2], "groups": GROUPS, "splits": meta},
              open(f"{root}/split_meta.json", "w"), indent=1)
    return meta
