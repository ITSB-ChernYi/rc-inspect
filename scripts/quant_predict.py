"""Predictions of a trained run at several numerical precisions (quantisation study).

usage: python scripts/quant_predict.py <run> [--modes pt32 pt16 onnx32 int8] [--splits val test]

Modes
  pt32    PyTorch, 32-bit float, CPU            (the pipeline of scripts/predict.py; reference)
  pt16    PyTorch, 16-bit float, CUDA           (half precision)
  onnx32  ONNX Runtime, 32-bit float, CPU       (reference for int8: same runtime and letterbox)
  int8    ONNX Runtime, 8-bit integer, CPU      (static post-training quantisation, QDQ, per channel,
                                                 MinMax calibration on training images; the box-decoding
                                                 part of the detection head is kept in float)
  int8e / int8p   as int8 with entropy / percentile calibration of the activation ranges
  int8h   as int8, but the whole detection head (last module) stays in float
  int8c   as int8, but only the convolutions are quantised (activations, additions, pooling in float)

Writes  $OBJ1DATA/preds_quant/<run>/<mode>_<split>.pkl   (same layout as scripts/predict.py)
        $OBJ1DATA/quant_models/<run>/{fp32,int8}.onnx
The test partition is only read; no setting is chosen on it. INT8 calibration uses training images.
"""
import argparse
import json
import os
import pickle
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import yolosec as Y  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

Y.register()
from ultralytics import YOLO  # noqa: E402

RUNS, PREDS, MODELS = f"{DATA}/runs", f"{DATA}/preds_quant", f"{DATA}/quant_models"
N_CALIB = 200


def fresh(path, src):
    """A cached file is reused only if it is newer than the file it was made from (a retrained run keeps its name)."""
    return os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src)


def load_gt(label_dir, stems, sizes):
    gt = {}
    for s in stems:
        W, H = sizes[s]
        p = os.path.join(label_dir, s + ".txt")
        a = np.loadtxt(p, ndmin=2) if os.path.exists(p) and os.path.getsize(p) > 0 else np.zeros((0, 5))
        b = np.column_stack([(a[:, 1] - a[:, 3] / 2) * W, (a[:, 2] - a[:, 4] / 2) * H,
                             (a[:, 1] + a[:, 3] / 2) * W, (a[:, 2] + a[:, 4] / 2) * H]) if len(a) else np.zeros((0, 4))
        gt[s] = {"boxes": b, "cls": a[:, 0].astype(int) if len(a) else np.zeros(0, int)}
    return gt


def export_fp32(run, imgsz):
    out = f"{MODELS}/{run}/fp32.onnx"
    if not fresh(out, f"{RUNS}/{run}/weights/best.pt"):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = f"{MODELS}/{run}/best.pt"
        shutil.copy(f"{RUNS}/{run}/weights/best.pt", tmp)
        f = YOLO(tmp).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, half=False, device="cpu")
        os.replace(f, out)
        os.remove(tmp)
    return out


def export_int8(run, imgsz, root, method="minmax"):
    import cv2
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
    from ultralytics.data.augment import LetterBox

    src = export_fp32(run, imgsz)
    out = f"{MODELS}/{run}/int8{'' if method == 'minmax' else '_' + method}.onnx"
    calib = {"minmax": CalibrationMethod.MinMax, "entropy": CalibrationMethod.Entropy,
             "percentile": CalibrationMethod.Percentile, "convonly": CalibrationMethod.MinMax,
             "floathead": CalibrationMethod.MinMax}[method]
    extra = {"op_types_to_quantize": ["Conv"]} if method == "convonly" else {}
    if fresh(out, src):
        return out
    m = onnx.load(src)
    # the detection head is the last module; its convolutions are quantised, the box decoding that follows
    # (distribution focal loss, anchors, strides, concatenation with the class scores) stays in float
    idx = [int(n.name.split("/")[1].split(".")[1]) for n in m.graph.node if n.name.startswith("/model.")]
    last = max(idx)
    exclude = [n.name for n in m.graph.node if n.name.startswith(f"/model.{last}/")
               and (method == "floathead" or n.op_type != "Conv" or "/dfl/" in n.name)]
    inp = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).get_inputs()[0]
    files = sorted(os.listdir(f"{root}/images/train"))
    files = files[::max(1, len(files) // N_CALIB)][:N_CALIB]
    lb = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(files)

        def get_next(self):
            f = next(self.it, None)
            if f is None:
                return None
            im = lb(image=cv2.imread(f"{root}/images/train/{f}"))
            x = np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0
            return {inp.name: x}

    quantize_static(src, out, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=calib, nodes_to_exclude=exclude, **extra)
    q = onnx.load(out)                       # keep the Ultralytics metadata (class names, stride, image size)
    del q.metadata_props[:]
    q.metadata_props.extend(m.metadata_props)
    onnx.save(q, out)
    json.dump({"calibration_images": len(files), "excluded_nodes": len(exclude), "head_module": last, "calibration_method": method,
               "nodes_total": len(m.graph.node), "fp32_bytes": os.path.getsize(src), "int8_bytes": os.path.getsize(out)},
              open(out[:-5] + "_meta.json", "w"), indent=1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--modes", nargs="+", default=["pt32", "pt16", "onnx32", "int8"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    rec = json.load(open(f"{RUNS}/{a.run}/run_record.json"))
    imgsz = rec["args"]["imgsz"]
    root = dataset_root(rec)
    os.makedirs(f"{PREDS}/{a.run}", exist_ok=True)
    for mode in a.modes:
        if mode in ("pt32", "pt16"):
            model, kw, bs = YOLO(f"{RUNS}/{a.run}/weights/best.pt"), (
                {"device": "cpu"} if mode == "pt32" else {"device": 0, "half": True}), 32
        else:
            f = export_fp32(a.run, imgsz) if mode == "onnx32" else export_int8(
                a.run, imgsz, root, {"int8": "minmax", "int8e": "entropy", "int8p": "percentile", "int8c": "convonly", "int8h": "floathead"}[mode])
            model, kw, bs = YOLO(f, task="detect"), {"device": "cpu"}, 1
        for split in a.splits:
            dst = f"{PREDS}/{a.run}/{mode}_{split}.pkl"
            if fresh(dst, f"{RUNS}/{a.run}/weights/best.pt"):
                continue
            img_dir = f"{root}/images/{split}"
            files = sorted(os.listdir(img_dir))
            stems = [f[:-4] for f in files]
            out, sizes, t0 = {}, {}, time.time()
            for k in range(0, len(files), bs):
                res = model.predict([f"{img_dir}/{f}" for f in files[k:k + bs]], imgsz=imgsz, conf=0.001, iou=0.7,
                                    max_det=300, verbose=False, **kw)
                for s, r in zip(stems[k:k + bs], res):
                    bx = r.boxes
                    out[s] = {"boxes": bx.xyxy.float().cpu().numpy().astype(np.float32),
                              "scores": bx.conf.float().cpu().numpy().astype(np.float32),
                              "cls": bx.cls.cpu().numpy().astype(np.int16)}
                    sizes[s] = (r.orig_shape[1], r.orig_shape[0])
            gt = load_gt(f"{root}/labels/{split}", stems, sizes)
            names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
            pickle.dump({"pred": out, "gt": gt, "names": names, "imgsz": imgsz, "mode": mode,
                         "seconds": time.time() - t0}, open(dst, "wb"))
            print(a.run, mode, split, len(out), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
