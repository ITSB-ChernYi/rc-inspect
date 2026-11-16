"""Cache predictions and standard metrics of a trained run on its validation and test partitions.

usage: python scripts/predict.py <run_name> [--splits val test] [--corrupt none]
Writes /home/user/obj1data/preds/<run>/<split>[_<corruption>].pkl  (stem -> boxes, scores, cls; plus GT)
and    results/metrics/<run>_<split>.json (Ultralytics validator: P, R, mAP50, mAP50-95, per-class AP).
The test partition is used only after the run is finished; no setting is chosen on it.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import yolosec as Y  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

Y.register()
from ultralytics import YOLO  # noqa: E402

RUNS, PREDS = f"{DATA}/runs", f"{DATA}/preds"
MET = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def load_gt(label_dir, img_dir, stems, sizes):
    gt = {}
    for s in stems:
        W, H = sizes[s]
        p = os.path.join(label_dir, s + ".txt")
        a = np.loadtxt(p, ndmin=2) if os.path.exists(p) and os.path.getsize(p) > 0 else np.zeros((0, 5))
        b = np.column_stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                             (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H]) if len(a) else np.zeros((0, 4))
        gt[s] = {"boxes": b, "cls": a[:, 0].astype(int) if len(a) else np.zeros(0, int)}
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--corrupt", default="none")     # e.g. noise:2 (see sdd/corrupt.py)
    ap.add_argument("--no-metrics", action="store_true")
    a = ap.parse_args()
    rec = json.load(open(os.path.join(RUNS, a.run, "run_record.json")))
    imgsz = rec["args"]["imgsz"]
    root = dataset_root(rec)
    if str(rec.get("model", "")).startswith("rtdetr"):
        from ultralytics import RTDETR
        model = RTDETR(os.path.join(RUNS, a.run, "weights", "best.pt"))
    else:
        model = YOLO(os.path.join(RUNS, a.run, "weights", "best.pt"))
    os.makedirs(os.path.join(PREDS, a.run), exist_ok=True)
    os.makedirs(MET, exist_ok=True)
    for split in a.splits:
        img_dir = os.path.join(root, "images", split)
        files = sorted(os.listdir(img_dir))
        stems = [f[:-4] for f in files]
        out, sizes, t0 = {}, {}, time.time()
        if a.corrupt == "none":
            sources = [os.path.join(img_dir, f) for f in files]
        else:
            from sdd import corrupt
            kind, sev = a.corrupt.split(":")
            sources = [corrupt.apply(os.path.join(img_dir, f), kind, int(sev), seed=k) for k, f in enumerate(files)]
        for k in range(0, len(sources), 32):
            res = model.predict(sources[k:k + 32], imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device="cpu",
                                verbose=False)
            for s, r in zip(stems[k:k + 32], res):
                bx = r.boxes
                out[s] = {"boxes": bx.xyxy.numpy().astype(np.float32), "scores": bx.conf.numpy().astype(np.float32),
                          "cls": bx.cls.numpy().astype(np.int16)}
                sizes[s] = (r.orig_shape[1], r.orig_shape[0])
        gt = load_gt(os.path.join(root, "labels", split), img_dir, stems, sizes)
        tag = split if a.corrupt == "none" else f"{split}_{a.corrupt.replace(':', '')}"
        pickle.dump({"pred": out, "gt": gt, "names": model.names, "imgsz": imgsz, "seconds": time.time() - t0},
                    open(os.path.join(PREDS, a.run, f"{tag}.pkl"), "wb"))
        if a.corrupt == "none" and not a.no_metrics:
            m = model.val(data=f"{root}/data.yaml", split=split, imgsz=imgsz, batch=16, device="cpu", conf=0.001, iou=0.7,
                          plots=False, verbose=False, project=f"{DATA}/val_tmp", name=a.run, exist_ok=True)
            names = m.names
            per_class = {names[int(c)]: {"ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
                         for i, c in enumerate(m.box.ap_class_index)}
            json.dump({"run": a.run, "split": split, "precision": float(m.box.mp), "recall": float(m.box.mr),
                       "map50": float(m.box.map50), "map50_95": float(m.box.map), "per_class": per_class,
                       "speed_ms": m.speed}, open(os.path.join(MET, f"{a.run}_{split}.json"), "w"), indent=1)
        print(a.run, tag, len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
