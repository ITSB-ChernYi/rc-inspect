"""Audit the DeepPCB copy (github.com/tangsanli5201/DeepPCB, commit 08e98c4):
annotations, exact and near duplicates of tested and template images, board groups, official split.
Output: results/audit/deeppcb_audit.json, deeppcb_images.csv, deeppcb_boxes.csv, deeppcb_near_duplicates.csv
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd.paths import DATA  # noqa: E402

SRC = f"{DATA}/raw/deeppcb_repo/PCBData"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "audit")
CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]


def dhash_bits(im, n=16):
    a = np.asarray(im.resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).reshape(-1)


def official():
    out = {}
    for split, f in (("trainval", "trainval.txt"), ("test", "test.txt")):
        for line in open(f"{SRC}/{f}"):
            if line.strip():
                out[os.path.basename(line.split()[0])[:-4]] = split
    return out


def main():
    off = official()
    meta, boxes, issues, T, S = [], [], [], [], []
    for p in sorted(glob.glob(f"{SRC}/group*/*/*_test.jpg")):
        stem = os.path.basename(p)[:-9]
        group = p.replace("\\", "/").split("/")[-3]
        tp = p[:-9] + "_temp.jpg"
        ann = f"{SRC}/{group}/{group[5:]}_not/{stem}.txt"
        im = Image.open(p).convert("L")
        row = {"stem": stem, "group": group, "official": off.get(stem, "unlisted"), "w": im.size[0], "h": im.size[1],
               "md5": hashlib.md5(open(p, "rb").read()).hexdigest(),
               "pix_md5": hashlib.md5(np.asarray(im).tobytes()).hexdigest(),
               "white_frac": float((np.asarray(im) > 127).mean()), "has_template": os.path.exists(tp)}
        if row["has_template"]:
            t = Image.open(tp).convert("L")
            row["temp_pix_md5"] = hashlib.md5(np.asarray(t).tobytes()).hexdigest()
            row["test_vs_temp_diff_frac"] = float(((np.asarray(t) > 127) != (np.asarray(im) > 127)).mean()) \
                if t.size == im.size else float("nan")
            T.append(dhash_bits(t))
        else:
            T.append(np.zeros(256, bool))
        S.append(dhash_bits(im))
        if not os.path.exists(ann):
            issues.append({"stem": stem, "issue": "no annotation file"})
            row["n_boxes"] = 0
            meta.append(row)
            continue
        n = 0
        for line in open(ann):
            v = line.replace(",", " ").split()
            if len(v) != 5:
                if v:
                    issues.append({"stem": stem, "issue": "malformed line", "line": line.strip()})
                continue
            x1, y1, x2, y2, c = (int(float(u)) for u in v)
            ok = 0 <= x1 < x2 <= im.size[0] and 0 <= y1 < y2 <= im.size[1] and 1 <= c <= 6
            if not ok:
                issues.append({"stem": stem, "issue": "box outside image, degenerate or unknown class",
                               "line": line.strip()})
            boxes.append({"stem": stem, "cls": CLASSES[c - 1] if 1 <= c <= 6 else f"UNKNOWN:{c}", "x1": x1, "y1": y1,
                          "x2": x2, "y2": y2, "w": x2 - x1, "h": y2 - y1, "valid": ok})
            n += 1
        row["n_boxes"] = n
        if n == 0:
            issues.append({"stem": stem, "issue": "annotation without boxes"})
        meta.append(row)
    M, B = pd.DataFrame(meta), pd.DataFrame(boxes)
    S, T = np.array(S), np.array(T)
    g = M.group.values
    iu = np.triu_indices(len(M), 1)
    nd = {}
    for name, H in (("tested", S), ("template", T)):
        D = (H[:, None, :] != H[None, :, :]).sum(-1)
        sel = D[iu] <= 20                                   # of 256 bits
        df = pd.DataFrame({"a": M.stem.values[iu[0][sel]], "b": M.stem.values[iu[1][sel]], "hamming": D[iu][sel],
                           "same_group": g[iu[0][sel]] == g[iu[1][sel]],
                           "official_a": M.official.values[iu[0][sel]], "official_b": M.official.values[iu[1][sel]]})
        df.insert(0, "kind", name)
        nd[name] = df
    ND = pd.concat(nd.values())
    os.makedirs(OUT, exist_ok=True)
    M.to_csv(f"{OUT}/deeppcb_images.csv", index=False)
    B.to_csv(f"{OUT}/deeppcb_boxes.csv", index=False)
    ND.to_csv(f"{OUT}/deeppcb_near_duplicates.csv", index=False)
    pd.DataFrame(issues).to_csv(f"{OUT}/deeppcb_issues.csv", index=False)
    side = np.sqrt(B.w * B.h)
    cross = lambda d: int(((d.official_a != d.official_b)).sum())  # noqa: E731
    rep = {
        "n_tested_images": len(M), "n_templates": int(M.has_template.sum()), "n_boxes": len(B),
        "image_sizes": (M.w.astype(str) + "x" + M.h.astype(str)).value_counts().to_dict(),
        "official_split": M.official.value_counts().to_dict(),
        "images_per_group": M.group.value_counts().sort_index().to_dict(),
        "official_split_per_group": M.groupby(["group", "official"]).size().unstack(fill_value=0).to_dict("index"),
        "groups_in_both_official_partitions": int((M.groupby("group").official.nunique() > 1).sum()),
        "boxes_per_class": B.cls.value_counts().to_dict(),
        "boxes_per_image_quantiles": M.n_boxes.quantile([0, .25, .5, .75, 1]).to_dict(),
        "issue_counts": pd.DataFrame(issues).issue.value_counts().to_dict() if issues else {},
        "exact_duplicate_tested_files": int(M.duplicated("pix_md5", keep=False).sum()),
        "exact_duplicate_template_images": int(M.duplicated("temp_pix_md5", keep=False).sum()),
        "exact_duplicate_template_groups": int(M[M.duplicated("temp_pix_md5", keep=False)].temp_pix_md5.nunique()),
        "test_vs_template_diff_frac_quantiles": M.test_vs_temp_diff_frac.quantile([0, .5, .9, 1]).round(4).to_dict(),
        "near_dup_tested_pairs": len(nd["tested"]), "near_dup_tested_cross_group": int((~nd["tested"].same_group).sum()),
        "near_dup_tested_across_official_split": cross(nd["tested"]),
        "near_dup_template_pairs": len(nd["template"]),
        "near_dup_template_cross_group": int((~nd["template"].same_group).sum()),
        "near_dup_template_across_official_split": cross(nd["template"]),
        "official_test_images_with_near_dup_template_in_trainval": int(len(
            set(nd["template"].query("official_a=='test' and official_b=='trainval'").a)
            | set(nd["template"].query("official_b=='test' and official_a=='trainval'").b))),
        "box_side_px_quantiles": side.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict(),
        "boxes_side_lt_32px": int((side < 32).sum()),
    }
    json.dump(rep, open(f"{OUT}/deeppcb_audit.json", "w"), indent=1, default=str)
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
