"""Train one detector under the common protocol and save the run record.

usage: python scripts/train.py --data <data.yaml> --model yolo11n --seed 0 --imgsz 256 --epochs 80 --name <run>
The test split is never touched here; model selection uses the validation split only.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO  # noqa: E402

from sdd import yolosec as Y  # noqa: E402
from sdd.paths import DATA  # noqa: E402

W = f"{DATA}/weights"
RUNS = f"{DATA}/runs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11n")      # weight name, or a .yaml (custom architecture)
    ap.add_argument("--pretrained", default=None)      # weights to initialise a custom .yaml from
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", default="{}")          # JSON of extra ultralytics train args
    ap.add_argument("--sec", default=None)            # YOLO-SEC variant: JSON of make_yaml kwargs
    ap.add_argument("--sofl", default="none")         # none | full | reg | focal
    ap.add_argument("--bb-only", action="store_true")  # initialise only the backbone (layers 0-8) from COCO weights
    ap.add_argument("--device", default="cpu")        # cpu, or a CUDA index such as 0
    ap.add_argument("--workers", type=int, default=2)  # data-loader workers
    a = ap.parse_args()
    trainer = None
    if a.sec is not None:
        Y.register()
        cfg = Y.make_yaml(a.name, **json.loads(a.sec))
        m = YOLO(cfg)
        wpath = os.path.join(W, (a.pretrained or "yolo12n") + ".pt")
        if a.bb_only:
            import torch
            src = torch.load(wpath, weights_only=False)["model"].float().state_dict()
            dst = m.model.state_dict()
            keep = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape
                    and int(k.split(".")[1]) <= 8}
            dst.update(keep)
            m.model.load_state_dict(dst)
            print(f"backbone-only initialisation: {len(keep)} tensors from {wpath}")
            init_info = {"init": "backbone-only", "tensors": len(keep)}
        else:
            m = m.load(wpath)
            init_info = {"init": "all matching layers"}
        trainer = Y.make_trainer(sofl=a.sofl != "none", focal=a.sofl in ("full", "focal"), reg=a.sofl in ("full", "reg"))
    else:
        if a.model.startswith("rtdetr"):
            from ultralytics import RTDETR
            m = RTDETR(os.path.join(W, a.model + ".pt"))
        else:
            m = YOLO(os.path.join(W, a.model + ".pt"))
        init_info = {"init": "full pretrained checkpoint"}
    t = time.time()
    args = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
                seed=a.seed, deterministic=True, optimizer="SGD", lr0=0.01, lrf=0.01, momentum=0.937,
                weight_decay=5e-4, cos_lr=True, warmup_epochs=3, close_mosaic=10, patience=1000,
                project=RUNS, name=a.name, exist_ok=True, plots=False, verbose=False, amp=False)
    args.update(json.loads(a.extra))
    if trainer is not None:
        m.train(trainer=trainer, **args)
    else:
        m.train(**args)
    import platform

    import torch
    import ultralytics
    hw = {"device": a.device, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
          "platform": platform.platform(), "cpu": platform.processor() or platform.machine(),
          "gpu": torch.cuda.get_device_name(0) if a.device != "cpu" and torch.cuda.is_available() else None}
    rec = {"args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in args.items()},
           "model": a.model, "pretrained": a.pretrained, "sec": a.sec, "sofl": a.sofl, **init_info, "train_seconds": round(time.time() - t, 1),
           "hardware": hw}
    json.dump(rec, open(os.path.join(RUNS, a.name, "run_record.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
