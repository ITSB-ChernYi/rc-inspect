"""Check whether Faster R-CNN post-processing truncates the candidate boxes that the review rule relies on (G3, R1 W10 /
R2 W5 / R4 M7). Re-predicts the NEU-DET test images with the trained weights under looser post-processing and
recomputes, for each setting: boxes per image above the score floor, the share of defects never found at the lowest
threshold, and the RC-Inspect operating point (Learn-then-Test, union review rule, alpha 0.10, budget 0.5) over
--reps calibration/evaluation partitions. Nothing is retrained. Output: results/risk_extra/frcnn_postproc_<run>.json

usage: python scripts/frcnn_postproc_check.py gpu_neu_s0_frcnn_256_seed1 [--reps 200]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from sdd import risk  # noqa: E402
from analyze_risk import LAMS, size_edges_from_train  # noqa: E402
from analyze_review_extra import pick  # noqa: E402
import frcnn  # noqa: E402

SETTINGS = {  # name: (box score floor, box NMS IoU, detections per image, RPN pre/post-NMS proposals at test time)
    "as_trained": (0.001, 0.5, 300, 1000, 1000),
    "nms0.7_det1000": (0.001, 0.7, 1000, 1000, 1000),
    "nms0.7_det1000_rpn3000": (0.001, 0.7, 1000, 3000, 3000),
    "floor1e-4_nms0.7_det1000_rpn3000": (0.0001, 0.7, 1000, 3000, 3000),
    "floor1e-4_det1000": (0.0001, 0.5, 1000, 1000, 1000),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    a = ap.parse_args()
    rec = json.load(open(f"{frcnn.RUNS}/{a.run}/run_record.json"))
    root = frcnn.dataset_root(rec)
    ck = torch.load(f"{frcnn.RUNS}/{a.run}/weights/best.pt", map_location="cpu")
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fn = f"results/risk_extra/frcnn_postproc_{a.run}.json"
    out = json.load(open(fn)) if os.path.exists(fn) else {"run": a.run, "settings": {}}
    for name in a.settings:
        floor, nms, det, pre, post = SETTINGS[name]
        m = frcnn.build(ck["nc"], ck["imgsz"], coco=False)
        m.load_state_dict(ck["state_dict"])
        m.roi_heads.score_thresh, m.roi_heads.nms_thresh, m.roi_heads.detections_per_img = floor, nms, det
        m.rpn._pre_nms_top_n["testing"], m.rpn._post_nms_top_n["testing"] = pre, post
        m.to(dev)
        pred, gt = frcnn.infer(m, root, "test", dev, bs=8)
        stems = sorted(pred)
        recs = risk.summarise({s: pred[s] for s in stems}, {s: gt[s] for s in stems}, class_aware=False, fragment=True)
        tab = risk.Table(recs, LAMS, size_edges=size_edges_from_train(a.run))
        nbox = np.array([(pred[s]["scores"] >= 0.001).sum() for s in stems])
        never = float(np.mean(np.concatenate([r["lstar"] for r in recs]) == 0))
        n = len(recs)
        rng = np.random.default_rng(2026)
        esc, rev, feas, viol = [], [], [], []
        for _ in range(a.reps):
            perm = rng.permutation(n)
            cal, ev = perm[:n // 2], perm[n // 2:]
            b = pick(tab, cal)
            d = tab.defective[ev]
            if b is None:
                esc.append(0.0); rev.append(1.0); feas.append(False)
            else:
                (k, rho), lam = b[0], b[1]
                j = int(np.argmin(np.abs(LAMS - lam)))
                Lo = tab.losses(ev, rho, k)
                esc.append(float(Lo["inst"][d, j].mean())); rev.append(float(Lo["review"][:, j].mean())); feas.append(True)
            viol.append(esc[-1] > 0.10)
        out["settings"][name] = {"floor": floor, "nms": nms, "det_per_img": det, "rpn_pre": pre, "rpn_post": post,
                                 "boxes_per_img_median": float(np.median(nbox)), "boxes_per_img_p90": float(np.percentile(nbox, 90)),
                                 "defects_never_found": never, "escape": float(np.mean(esc)), "review": float(np.mean(rev)),
                                 "feasible": float(np.mean(feas)), "violations": float(np.mean(viol))}
        print(name, json.dumps(out["settings"][name]), flush=True)
        del m
        torch.cuda.empty_cache()
        os.makedirs("results/risk_extra", exist_ok=True)
        json.dump(out, open(fn, "w"), indent=1)


if __name__ == "__main__":
    main()
