"""CPU latency of the detectors and of the risk layer (run only when no training job is active).

Model-only latency: fused PyTorch model, batch 1, fixed input size, 30 warm-up and 200 timed forward passes,
torch.inference_mode, reported as median and 90th percentile for 1 and 4 intra-op threads.
End-to-end latency: Ultralytics predict on real test images (decode + letterbox + inference + NMS) at the
deployment threshold, plus the risk layer (threshold, residual-evidence score, decision), per image.
Also times the label-free acquisition-quality gate (sdd/quality.py) per image.
Output: results/latency.json, or --out (e.g. results/latency_pc.json on the benchmarking PC).

usage (PC): python objective1/scripts/bench_latency.py --runs-dir objective1/pc_runs --out objective1/results/latency_pc.json
With no run names, the seed-0 GPU model of every architecture and dataset is benchmarked (PAPER_RUNS).
"""
import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import quality  # noqa: E402
from sdd.paths import DATA  # noqa: E402
from sdd import yolosec as Y  # noqa: E402

Y.register()
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

RUNS = "/home/user/obj1data/runs"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "latency.json")
PAPER_RUNS = [f"gpu_neu_s0_{m}_256_seed0" for m in ("yolo11n", "yolo12n", "yolov8n", "secfull", "secnosofl", "v12nbb",
                                                    "hcsa", "ewbifpn", "bifpn")] + \
             ["gpu_mt_s0_yolo11n_320_seed0", "gpu_gc10g_s0_yolo11n_640_seed0"]


def cpu_name():
    try:
        return [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if "model name" in l][0]
    except Exception:
        pass
    try:  # Windows
        return subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return platform.processor()


def model_only(net, imgsz, threads, n=200, warm=30):
    torch.set_num_threads(threads)
    x = torch.zeros(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        for _ in range(warm):
            net(x)
        t = []
        for _ in range(n):
            t0 = time.perf_counter()
            net(x)
            t.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90))}


def risk_and_gate(s, f):
    """Time the decision module on the scores s of one image and the quality gate on image file f."""
    t0 = time.perf_counter()
    lam, rho = 0.2, 0.5
    keep = s >= lam
    u = s[~keep].max(initial=0.0)
    _decision = "review" if u >= rho * lam and u > 0 else ("reject" if keep.any() else "accept")
    rt = (time.perf_counter() - t0) * 1e3
    from PIL import Image
    g = np.asarray(Image.open(f).convert("L"))
    t0 = time.perf_counter()
    quality.features(g)
    return rt, (time.perf_counter() - t0) * 1e3


def bench_frcnn(d, rec, files):
    """Faster R-CNN (scripts/frcnn.py). The model-only pass uses the first test image rather than a blank input,
    because the number of region proposals, and with it the cost of the second stage, depends on the content.
    GFLOPs from torch.utils.flop_counter (2 per multiply-accumulate, as in Ultralytics) on the same image."""
    from PIL import Image
    from torch.utils.flop_counter import FlopCounterMode
    from torchvision.transforms.functional import pil_to_tensor
    import frcnn as F
    ck = torch.load(f"{d}/best.pt" if os.path.exists(f"{d}/best.pt") else f"{d}/weights/best.pt", map_location="cpu")
    net = F.build(ck["nc"], ck["imgsz"], coco=False)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    x0 = pil_to_tensor(Image.open(files[0]).convert("RGB")).float() / 255
    with torch.inference_mode(), FlopCounterMode(display=False) as fc:
        net([x0])
    entry = {"imgsz": ck["imgsz"], "params": int(sum(p.numel() for p in net.parameters())),
             "gflops_at_imgsz": round(fc.get_total_flops() / 1e9, 3), "flops_counter": "torch.utils.flop_counter"}
    mo = {}
    for thr in (1, 4):
        torch.set_num_threads(thr)
        with torch.inference_mode():
            for _ in range(10):
                net([x0])
            t = []
            for _ in range(50):
                t0 = time.perf_counter()
                net([x0])
                t.append((time.perf_counter() - t0) * 1e3)
        mo[f"{thr}thr"] = {"median_ms": float(np.median(t)), "p90_ms": float(np.percentile(t, 90)), "n": 50}
    entry["model_only"] = mo
    torch.set_num_threads(4)
    pre, inf, risk_t, gate_t = [], [], [], []
    with torch.inference_mode():
        for f in files[:5]:
            net([pil_to_tensor(Image.open(f).convert("RGB")).float() / 255])
        for f in files:
            t0 = time.perf_counter()
            x = pil_to_tensor(Image.open(f).convert("RGB")).float() / 255
            t1 = time.perf_counter()
            o = net([x])[0]
            s = o["scores"].numpy()
            t2 = time.perf_counter()
            pre.append((t1 - t0) * 1e3); inf.append((t2 - t1) * 1e3)
            rt, gt_ = risk_and_gate(s, f)
            risk_t.append(rt); gate_t.append(gt_)
    tot = np.array(pre) + np.array(inf) + np.array(risk_t)
    entry["end_to_end_4thr"] = {"preprocess_ms": float(np.median(pre)), "inference_ms": float(np.median(inf)),
                                "postprocess_ms": 0.0, "risk_layer_ms": float(np.median(risk_t)),
                                "quality_gate_ms": float(np.median(gate_t)), "total_ms": float(np.median(tot)),
                                "total_with_gate_ms": float(np.median(tot + np.array(gate_t))), "n_images": len(files),
                                "note": "inference includes the internal resize, both stages and box post-processing"}
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--runs-dir", default=RUNS)       # objective1/pc_runs on the benchmarking PC
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    runs = a.runs or PAPER_RUNS
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    res["_env"] = {"cpu": platform.processor() or platform.machine(), "cpu_model": cpu_name(), "torch": torch.__version__,
                   "python": platform.python_version(), "platform": platform.platform(), "cores": os.cpu_count()}
    for run in runs:
        d = f"{a.runs_dir}/{run}"
        rec = json.load(open(f"{d}/run_record.json"))
        imgsz = rec["args"]["imgsz"]
        import yaml
        ds = rec["args"]["data"].replace("\\", "/").rstrip("/").split("/")[-2]   # e.g. neu_s0
        root = yaml.safe_load(open(f"{DATA}/{ds}/data.yaml"))["path"]
        files = sorted(glob.glob(f"{root}/images/test/*.jpg"))[:100]
        if rec.get("model") == "fasterrcnn_r50fpn":
            res[run] = entry = bench_frcnn(d, rec, files)
            print(f"{run}: model-only {entry['model_only']['1thr']['median_ms']:.1f} / "
                  f"{entry['model_only']['4thr']['median_ms']:.1f} ms; end-to-end "
                  f"{entry['end_to_end_4thr']['total_ms']:.1f} ms", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
            continue
        w = f"{d}/weights/best.pt" if os.path.exists(f"{d}/weights/best.pt") else f"{d}/best.pt"
        if str(rec.get("model", "")).startswith("rtdetr"):
            from ultralytics import RTDETR
            yolo = RTDETR(w)
        else:
            yolo = YOLO(w)
        net = yolo.model.float().eval().fuse()
        size = os.path.getsize(w)
        entry = {"imgsz": imgsz, "params": int(sum(p.numel() for p in net.parameters())),
                 "gflops_at_imgsz": round(get_flops(net, imgsz), 3), "checkpoint_mb_fp16": round(size / 2**20, 2),
                 "model_only": {f"{t}thr": model_only(net, imgsz, t) for t in (1, 4)}}
        torch.set_num_threads(4)
        yolo.predict(files[:5], imgsz=imgsz, conf=0.001, device="cpu", verbose=False)
        pre, inf, post, risk_t, gate_t = [], [], [], [], []
        for f in files:
            r = yolo.predict(f, imgsz=imgsz, conf=0.001, iou=0.7, max_det=300, device="cpu", verbose=False)[0]
            pre.append(r.speed["preprocess"]); inf.append(r.speed["inference"]); post.append(r.speed["postprocess"])
            t0 = time.perf_counter()
            s = r.boxes.conf.numpy()
            lam, rho = 0.2, 0.5
            keep = s >= lam
            u = s[~keep].max(initial=0.0)
            _decision = "review" if u >= rho * lam and u > 0 else ("reject" if keep.any() else "accept")
            risk_t.append((time.perf_counter() - t0) * 1e3)
            from PIL import Image
            g = np.asarray(Image.open(f).convert("L"))
            t0 = time.perf_counter()
            quality.features(g)
            gate_t.append((time.perf_counter() - t0) * 1e3)
        entry["end_to_end_4thr"] = {"preprocess_ms": float(np.median(pre)), "inference_ms": float(np.median(inf)),
                                    "postprocess_ms": float(np.median(post)), "risk_layer_ms": float(np.median(risk_t)),
                                    "quality_gate_ms": float(np.median(gate_t)),
                                    "total_ms": float(np.median(np.array(pre) + np.array(inf) + np.array(post) + np.array(risk_t))),
                                    "total_with_gate_ms": float(np.median(np.array(pre) + np.array(inf) + np.array(post)
                                                                          + np.array(risk_t) + np.array(gate_t))),
                                    "n_images": len(files)}
        res[run] = entry
        e2e = entry["end_to_end_4thr"]
        print(f"{run}: model-only {entry['model_only']['1thr']['median_ms']:.1f} ms (1 thread), "
              f"{entry['model_only']['4thr']['median_ms']:.1f} ms (4 threads); end-to-end {e2e['total_ms']:.1f} ms; "
              f"quality gate {e2e['quality_gate_ms']:.1f} ms", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
