"""Faster R-CNN (ResNet-50-FPN, torchvision, COCO-pretrained) under the common protocol of scripts/train.py.

usage: python scripts/frcnn.py train --data <data.yaml> --seed 0 --imgsz 256 --epochs 80 --name <run>
                                     [--batch 16] [--device 0] [--workers 2]
       python scripts/frcnn.py predict <run> [--splits val test]

Kept from the common protocol: splits, epochs, batch size, SGD (lr 0.01, momentum 0.937, weight decay 5e-4),
cosine decay to 1e-4 with 3 warm-up epochs, fixed seed with deterministic algorithms where available, and
checkpoint selection on validation mAP@0.5:0.95 (sdd/evalmap.py, which reproduces the Ultralytics validator).
Differences forced by the architecture: the image is resized so that its longer side equals imgsz (torchvision
transform, ImageNet normalisation, no letterbox); augmentation is a horizontal flip only, because the
Ultralytics mosaic/HSV pipeline belongs to the YOLO trainer; the first ResNet stage and all batch-norm layers stay
frozen as in the torchvision reference recipe; post-processing is the torchvision default (per-class NMS at IoU
0.5) with a score floor of 0.001 and at most 300 detections. Predictions for the cached pickles are made on the
CPU, as in scripts/predict.py.

Outputs mirror scripts/train.py and scripts/predict.py: $OBJ1DATA/runs/<run>/{weights/best.pt, results.csv,
args.yaml, run_record.json}, $OBJ1DATA/preds/<run>/<split>.pkl, results/metrics/<run>_<split>.json.
"""
import argparse
import functools
import json
import math
import os
import pickle
import platform
import random
import sys
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.models import resnet50
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from torchvision.transforms.functional import pil_to_tensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import evalmap  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

W = f"{DATA}/weights/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
RUNS, PREDS = f"{DATA}/runs", f"{DATA}/preds"
MET = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def build(nc, imgsz, coco=True):
    bb = resnet50(weights=None, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    bb = _resnet_fpn_extractor(bb, 3)
    m = FasterRCNN(bb, num_classes=91, min_size=imgsz, max_size=imgsz, box_score_thresh=0.001,
                   box_detections_per_img=300)
    if coco:
        m.load_state_dict(torch.load(W, map_location="cpu"))
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, nc + 1)
    return m


def read_labels(root, split, stem, w, h):
    p = f"{root}/labels/{split}/{stem}.txt"
    a = np.loadtxt(p, ndmin=2) if os.path.exists(p) and os.path.getsize(p) > 0 else np.zeros((0, 5))
    if not len(a):
        return np.zeros((0, 4), np.float32), np.zeros(0, np.int64)
    b = np.column_stack([(a[:, 1] - a[:, 3] / 2) * w, (a[:, 2] - a[:, 4] / 2) * h,
                         (a[:, 1] + a[:, 3] / 2) * w, (a[:, 2] + a[:, 4] / 2) * h]).astype(np.float32)
    return b, a[:, 0].astype(np.int64)


class DS(torch.utils.data.Dataset):
    def __init__(self, root, split, train):
        self.root, self.split, self.train = root, split, train
        self.files = sorted(os.listdir(f"{root}/images/{split}"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        im = Image.open(f"{self.root}/images/{self.split}/{f}").convert("RGB")
        w, h = im.size
        b, c = read_labels(self.root, self.split, f[:-4], w, h)
        b[:, [0, 2]] = b[:, [0, 2]].clip(0, w)
        b[:, [1, 3]] = b[:, [1, 3]].clip(0, h)
        keep = ((b[:, 2] - b[:, 0]) >= 1) & ((b[:, 3] - b[:, 1]) >= 1)   # torchvision rejects degenerate boxes
        b, c = b[keep], c[keep]
        x = pil_to_tensor(im).float() / 255
        if self.train and random.random() < 0.5:
            x = x.flip(-1)
            b = b.copy()
            b[:, [0, 2]] = w - b[:, [2, 0]]
        return x, {"boxes": torch.from_numpy(b).reshape(-1, 4), "labels": torch.from_numpy(c + 1)}, f[:-4]


def collate(batch):
    return tuple(zip(*batch))


def seed_worker(base, k):
    random.seed(base * 1000 + k)
    np.random.seed(base * 1000 + k)


@torch.no_grad()
def infer(model, root, split, device, bs=16):
    model.eval()
    ds = DS(root, split, False)
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0, collate_fn=collate)
    pred, gt = {}, {}
    for xs, ts, stems in dl:
        out = model([x.to(device) for x in xs])
        for o, t, s in zip(out, ts, stems):
            pred[s] = {"boxes": o["boxes"].cpu().numpy().astype(np.float32),
                       "scores": o["scores"].cpu().numpy().astype(np.float32),
                       "cls": (o["labels"].cpu().numpy() - 1).astype(np.int16)}
    for f in ds.files:   # ground truth without the degenerate-box filter, as in scripts/predict.py
        w, h = Image.open(f"{root}/images/{split}/{f}").size
        b, c = read_labels(root, split, f[:-4], w, h)
        gt[f[:-4]] = {"boxes": b.astype(np.float64), "cls": c.astype(int)}
    return pred, gt


def train(a):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    root = os.path.dirname(a.data.replace("\\", "/"))
    names = yaml.safe_load(open(a.data))["names"]
    nc = len(names)
    dev = torch.device(f"cuda:{a.device}" if a.device != "cpu" else "cpu")
    model = build(nc, a.imgsz).to(dev)
    g = torch.Generator()
    g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(
        DS(root, "train", True), batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=collate,
        generator=g, worker_init_fn=functools.partial(seed_worker, a.seed), persistent_workers=a.workers > 0,
        drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.937, weight_decay=5e-4)
    nb, nw = len(dl), max(3 * len(dl), 100)
    out = f"{RUNS}/{a.name}"
    os.makedirs(f"{out}/weights", exist_ok=True)
    yaml.safe_dump(vars(a), open(f"{out}/args.yaml", "w"))
    log = open(f"{out}/results.csv", "w")
    log.write("epoch,train_loss,lr,val_map50,val_map50_95,seconds\n")
    best, t0 = -1.0, time.time()
    for ep in range(a.epochs):
        model.train()
        lr_ep = 0.01 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * ep / a.epochs)))   # cosine 0.01 -> 1e-4
        tot, te = 0.0, time.time()
        for k, (xs, ts, _) in enumerate(dl):
            it = ep * nb + k
            lr = lr_ep * min(1.0, (it + 1) / nw)    # linear warm-up over the first 3 epochs
            for pg in opt.param_groups:
                pg["lr"] = lr
            loss = sum(model([x.to(dev) for x in xs], [{kk: v.to(dev) for kk, v in t.items()} for t in ts]).values())
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        pred, gt = infer(model, root, "val", dev)
        m = evalmap.summary(evalmap.prepare(pred, gt, nc))
        log.write(f"{ep + 1},{tot / nb:.4f},{lr_ep:.6f},{m['map50']:.5f},{m['map50_95']:.5f},{time.time() - te:.1f}\n")
        log.flush()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / nb:.3f} val mAP50 {m['map50']:.4f} mAP50-95 {m['map50_95']:.4f}"
              f" ({time.time() - te:.0f}s)", flush=True)
        if m["map50_95"] > best:
            best = m["map50_95"]
            torch.save({"state_dict": model.state_dict(), "nc": nc, "imgsz": a.imgsz, "epoch": ep + 1,
                        "names": names}, f"{out}/weights/best.pt")
    hw = {"device": a.device, "torch": torch.__version__, "platform": platform.platform(),
          "cpu": platform.processor() or platform.machine(),
          "gpu": torch.cuda.get_device_name(0) if a.device != "cpu" and torch.cuda.is_available() else None}
    rec = {"args": {"data": a.data, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch, "workers": a.workers,
                    "device": a.device, "seed": a.seed, "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
                    "momentum": 0.937, "weight_decay": 5e-4, "cos_lr": True, "warmup_epochs": 3},
           "model": "fasterrcnn_r50fpn", "init": "full COCO checkpoint except the box predictor",
           "best_val_map50_95": best, "train_seconds": round(time.time() - t0, 1), "hardware": hw}
    json.dump(rec, open(f"{out}/run_record.json", "w"), indent=1)


def predict(a):
    torch.set_num_threads(os.cpu_count() or 4)
    rec = json.load(open(f"{RUNS}/{a.run}/run_record.json"))
    root = dataset_root(rec)
    ck = torch.load(f"{RUNS}/{a.run}/weights/best.pt", map_location="cpu")
    model = build(ck["nc"], ck["imgsz"], coco=False)
    model.load_state_dict(ck["state_dict"])
    names = ck["names"]
    os.makedirs(f"{PREDS}/{a.run}", exist_ok=True)
    os.makedirs(MET, exist_ok=True)
    for split in a.splits:
        t0 = time.time()
        pred, gt = infer(model, root, split, torch.device("cpu"), bs=8)
        pickle.dump({"pred": pred, "gt": gt, "names": names, "imgsz": ck["imgsz"], "seconds": time.time() - t0},
                    open(f"{PREDS}/{a.run}/{split}.pkl", "wb"))
        m = evalmap.summary(evalmap.prepare(pred, gt, len(names)))
        per_class = {names[c]: {"ap50": m["ap50_per_class"][c], "ap50_95": m["ap50_95_per_class"][c]}
                     for c in range(len(names))}
        json.dump({"run": a.run, "split": split, "precision": float("nan"), "recall": float("nan"),
                   "map50": m["map50"], "map50_95": m["map50_95"], "per_class": per_class,
                   "evaluator": "sdd/evalmap.py (Ultralytics integration)"},
                  open(os.path.join(MET, f"{a.run}_{split}.json"), "w"), indent=1)
        print(a.run, split, len(pred), f"mAP50 {m['map50']:.4f} mAP50-95 {m['map50_95']:.4f}",
              f"{time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--imgsz", type=int, default=256)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--name", required=True)
    t.add_argument("--device", default="0")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--model", default="fasterrcnn_r50fpn")   # accepted for queue compatibility; ignored
    p = sub.add_parser("predict")
    p.add_argument("run")
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
