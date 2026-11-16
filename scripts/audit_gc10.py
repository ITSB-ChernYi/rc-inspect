"""Audit the GC10-DET copy (github.com/wyx19980727/GC10-DET-VOC, commit 2459a71):
annotations vs folder class, missing labels, exact and near duplicates, acquisition groups, boxes.
Output: results/audit/gc10_audit.json, gc10_images.csv, gc10_boxes.csv, gc10_near_duplicates.csv
"""
import glob
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

SRC = "/home/user/obj1data/raw/gc10_repo/VOC2007"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "audit")
# folder prefix -> English class name (Lv et al., Sensors 2020, Table 1)
NAMES = {"punching_hole": "punching_hole", "welding_line": "welding_line", "crescent_gap": "crescent_gap",
         "water_spot": "water_spot", "oil_spot": "oil_spot", "silk_spot": "silk_spot", "inclusion": "inclusion",
         "rolled_pit": "rolled_pit", "crease": "crease", "waist_folding": "waist_folding"}
XMLNAME = {"1": "punching_hole", "2": "welding_line", "3": "crescent_gap", "4": "water_spot", "5": "oil_spot",
           "6": "silk_spot", "7": "inclusion", "8": "rolled_pit", "9": "crease", "10": "waist_folding"}


def main():
    files = sorted(glob.glob(f"{SRC}/JPEGImages/*.jpg"))
    meta, boxes, issues = [], [], []
    for p in files:
        stem = os.path.basename(p)[:-4]
        m = re.match(r"(.+?)_(img_(\d+)_([A-Za-z]*\d+)_(\d+))$", stem)
        folder_cls, orig, seq, frame = m.group(1).replace(" ", "_"), m.group(2), m.group(3) + "_" + m.group(4), m.group(5)
        raw = open(p, "rb").read()
        im = Image.open(p).convert("L")
        small = im.resize((512, 250))
        meta.append({"stem": stem, "folder_cls": folder_cls, "orig_name": orig, "seq": seq, "frame": int(frame),
                     "md5": hashlib.md5(raw).hexdigest(), "pix_md5": hashlib.md5(np.asarray(im).tobytes()).hexdigest(),
                     "phash": str(imagehash.phash(small, hash_size=16)), "w": im.size[0], "h": im.size[1]})
        x = f"{SRC}/Annotations/{stem}.xml"
        if not os.path.exists(x):
            issues.append({"stem": stem, "issue": "no annotation file"}); continue
        r = ET.parse(x).getroot()
        objs = r.findall("object")
        if not objs:
            issues.append({"stem": stem, "issue": "annotation without objects"})
        xml_classes = set()
        for o in objs:
            raw_name = o.find("name").text.strip()
            num = raw_name.split("_")[0]
            cls = XMLNAME.get(num, "UNKNOWN:" + raw_name)
            xml_classes.add(cls)
            b = o.find("bndbox")
            x1, y1, x2, y2 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            ok = 0 <= x1 < x2 <= im.size[0] and 0 <= y1 < y2 <= im.size[1]
            if not ok:
                issues.append({"stem": stem, "issue": "box outside image or degenerate", "box": [x1, y1, x2, y2]})
            boxes.append({"stem": stem, "cls": cls, "raw_name": raw_name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "w": x2 - x1, "h": y2 - y1, "valid": ok})
        if folder_cls not in xml_classes:
            issues.append({"stem": stem, "issue": "folder class absent from annotation", "xml": sorted(xml_classes)})
    M, B = pd.DataFrame(meta), pd.DataFrame(boxes)
    H = np.array([[int(c, 16) for c in h] for h in M.phash], dtype=np.uint8)
    bits = np.unpackbits(H[:, :, None], axis=2)[:, :, 4:].reshape(len(M), -1).astype(np.uint8)
    D = (bits[:, None, :] != bits[None, :, :]).sum(-1)
    iu = np.triu_indices(len(M), 1)
    sel = D[iu] <= 24
    ND = pd.DataFrame({"a": M.stem.values[iu[0][sel]], "b": M.stem.values[iu[1][sel]], "phash_hamming": D[iu][sel]})
    ND["same_orig_name"] = [M.set_index("stem").orig_name[a] == M.set_index("stem").orig_name[b] for a, b in zip(ND.a, ND.b)]
    ND["same_seq"] = [M.set_index("stem").seq[a] == M.set_index("stem").seq[b] for a, b in zip(ND.a, ND.b)]
    os.makedirs(OUT, exist_ok=True)
    M.to_csv(f"{OUT}/gc10_images.csv", index=False)
    B.to_csv(f"{OUT}/gc10_boxes.csv", index=False)
    ND.to_csv(f"{OUT}/gc10_near_duplicates.csv", index=False)
    pd.DataFrame(issues).to_csv(f"{OUT}/gc10_issues.csv", index=False)
    seq_sizes = M.groupby("seq").size()
    side = np.sqrt(B.w * B.h)
    rep = {
        "n_images": len(M), "n_annotation_files": len(glob.glob(f"{SRC}/Annotations/*.xml")), "n_boxes": len(B),
        "image_sizes": (M.w.astype(str) + "x" + M.h.astype(str)).value_counts().to_dict(),
        "images_per_folder_class": M.folder_cls.value_counts().to_dict(),
        "boxes_per_class": B.cls.value_counts().to_dict(),
        "issue_counts": pd.DataFrame(issues).issue.value_counts().to_dict() if issues else {},
        "exact_duplicate_file_groups": int(M[M.duplicated("md5", keep=False)].md5.nunique()),
        "exact_duplicate_files": int(M.duplicated("md5", keep=False).sum()),
        "same_original_name_in_two_folders": int(M[M.duplicated("orig_name", keep=False)].orig_name.nunique()),
        "near_dup_pairs_hamming_le_24": len(ND),
        "near_dup_pairs_different_orig_name": int((~ND.same_orig_name).sum()) if len(ND) else 0,
        "near_dup_pairs_within_sequence": int(ND.same_seq.sum()) if len(ND) else 0,
        "n_sequences": int(seq_sizes.size), "images_per_sequence_quantiles": seq_sizes.quantile([.5, .9, .99, 1]).to_dict(),
        "sequences_with_multiple_classes": int((M.groupby("seq").folder_cls.nunique() > 1).sum()),
        "box_side_px_quantiles": side.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(),
        "boxes_side_lt_32px_native": int((side < 32).sum()),
        "boxes_side_lt_32px_at_640_input": int((side * 640 / 2048 < 32).sum()),
    }
    json.dump(rep, open(f"{OUT}/gc10_audit.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
