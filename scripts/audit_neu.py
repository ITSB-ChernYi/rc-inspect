"""Audit NEU-DET: annotation validity, class/box statistics, exact and near-duplicate images.

Input: the NEU-DET mirror (IMAGES/, ANNOTATIONS/, Validation_Images/, Validation_Annotations/).
Output: results/audit/neu_audit.json, results/audit/neu_boxes.csv, results/audit/neu_near_duplicates.csv
"""
import glob
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

SRC = sys.argv[1] if len(sys.argv) > 1 else "/home/user/obj1data/raw/neu_repo"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "audit")
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def main():
    imgs = sorted(glob.glob(f"{SRC}/IMAGES/*.jpg") + glob.glob(f"{SRC}/Validation_Images/*.jpg"))
    ann_dir = {os.path.basename(p)[:-4]: p for p in glob.glob(f"{SRC}/ANNOTATIONS/*.xml") + glob.glob(f"{SRC}/Validation_Annotations/*.xml")}
    rows, problems, meta = [], [], []
    for p in imgs:
        stem = os.path.basename(p)[:-4]
        cls_from_name = stem.rsplit("_", 1)[0]
        raw = open(p, "rb").read()
        im = Image.open(p)
        g = np.asarray(im.convert("L"), dtype=np.uint8)
        rgb = np.asarray(im.convert("RGB")).astype(int)
        meta.append({"stem": stem, "cls": cls_from_name, "md5": hashlib.md5(raw).hexdigest(),
                     "pix_md5": hashlib.md5(g.tobytes()).hexdigest(),
                     "phash": str(imagehash.phash(im.convert("L"), hash_size=16)),
                     "w": im.size[0], "h": im.size[1],
                     "gray_channels_equal": bool((rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()),
                     "mean": float(g.mean()), "std": float(g.std())})
        if stem not in ann_dir:
            problems.append({"stem": stem, "issue": "missing annotation"}); continue
        root = ET.parse(ann_dir[stem]).getroot()
        W, H = int(root.find("size/width").text), int(root.find("size/height").text)
        objs = root.findall("object")
        if not objs:
            problems.append({"stem": stem, "issue": "no objects"})
        for o in objs:
            name = o.find("name").text.strip()
            b = o.find("bndbox")
            x1, y1, x2, y2 = (int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax"))
            issue = []
            if name != cls_from_name:
                issue.append(f"label {name} != file class {cls_from_name}")
            if (W, H) != im.size:
                issue.append(f"xml size {W}x{H} != image {im.size}")
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                issue.append("box outside image or degenerate")
            if issue:
                problems.append({"stem": stem, "issue": "; ".join(issue), "box": [x1, y1, x2, y2]})
            rows.append({"stem": stem, "cls": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                         "w": x2 - x1, "h": y2 - y1, "area_frac": (x2 - x1) * (y2 - y1) / (W * H)})
    M = pd.DataFrame(meta)
    B = pd.DataFrame(rows)
    # exact duplicates (file bytes and decoded pixels)
    dup_file = M[M.duplicated("md5", keep=False)].sort_values("md5")
    dup_pix = M[M.duplicated("pix_md5", keep=False)].sort_values("pix_md5")
    # near duplicates: Hamming distance of 256-bit perceptual hashes
    H = np.array([[int(c, 16) for c in h] for h in M.phash])  # hex digits
    bits = np.unpackbits(H.astype(np.uint8)[:, :, None], axis=2)[:, :, 4:].reshape(len(M), -1)
    D = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    iu = np.triu_indices(len(M), 1)
    near = [(M.stem[i], M.stem[j], int(D[i, j])) for i, j in zip(*iu) if D[i, j] <= 20]
    ND = pd.DataFrame(near, columns=["a", "b", "phash_hamming"]).sort_values("phash_hamming")
    # pixel-level check of candidate pairs (mean abs difference on 0-255 grayscale)
    def mad(a, b):
        A = np.asarray(Image.open(f"{SRC}/{'IMAGES' if os.path.exists(f'{SRC}/IMAGES/{a}.jpg') else 'Validation_Images'}/{a}.jpg").convert("L"), float)
        Bm = np.asarray(Image.open(f"{SRC}/{'IMAGES' if os.path.exists(f'{SRC}/IMAGES/{b}.jpg') else 'Validation_Images'}/{b}.jpg").convert("L"), float)
        return float(np.abs(A - Bm).mean()), float(np.abs(A - np.fliplr(Bm)).mean())
    if len(ND):
        ND[["mad", "mad_flipped"]] = [mad(a, b) for a, b in zip(ND.a, ND.b)]
    os.makedirs(OUT, exist_ok=True)
    B.to_csv(os.path.join(OUT, "neu_boxes.csv"), index=False)
    ND.to_csv(os.path.join(OUT, "neu_near_duplicates.csv"), index=False)
    M.to_csv(os.path.join(OUT, "neu_images.csv"), index=False)
    smallest = B.assign(side=np.sqrt(B.w * B.h))
    rep = {
        "n_images": len(M), "n_annotation_files": len(ann_dir), "n_boxes": len(B),
        "images_per_class": M.cls.value_counts().to_dict(),
        "boxes_per_class": B.cls.value_counts().to_dict(),
        "boxes_per_image": {"mean": float(B.groupby("stem").size().mean()), "max": int(B.groupby("stem").size().max())},
        "image_sizes": (M.w.astype(str) + "x" + M.h.astype(str)).value_counts().to_dict(),
        "all_rgb_channels_equal": bool(M.gray_channels_equal.all()),
        "problems": problems,
        "exact_duplicate_files": dup_file[["stem", "md5"]].values.tolist(),
        "exact_duplicate_pixels": dup_pix[["stem", "pix_md5"]].values.tolist(),
        "near_duplicate_pairs_hamming_le_20": len(ND),
        "near_duplicate_pairs_mad_lt_5": int((ND.get("mad", pd.Series(dtype=float)) < 5).sum()) if len(ND) else 0,
        "box_side_px_quantiles": smallest.side.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(),
        "boxes_side_lt_32px_at_200": int((smallest.side < 32).sum()),
        "area_frac_by_class_median": B.groupby("cls").area_frac.median().round(3).to_dict(),
        "boxes_per_image_by_class_mean": B.groupby(["cls", "stem"]).size().groupby("cls").mean().round(2).to_dict(),
    }
    json.dump(rep, open(os.path.join(OUT, "neu_audit.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "problems"}, indent=1, default=str)[:3000])
    print("problems:", len(problems), problems[:10])


if __name__ == "__main__":
    main()
