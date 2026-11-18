"""Supplementary tables S1-S3 of the CAEE manuscript, from saved result files only.

S1  YOLOv12n variants on NEU-DET: components, complexity, test accuracy, escape at the default threshold and review
    of RC-Inspect (results/metrics/, results/risk/*__test_summary.csv, results/architectures.json)
S2  device check: seed-0 models trained on a 4-core CPU and on the GPU with the same seed, split and settings. The CPU
    models were trained and evaluated in the cloud workspace; their values are taken from the supplementary file of
    that workspace (branch claude/focused-maxwell-qd3g0o, commit d4f7267) and the GPU values are checked here
    against the local result files.
S3  escape among the images decided without review (results/risk_extra/auto_escape.csv, scripts/auto_escape.py)
Writes results/tables/supplementary.md.
"""
import json
import os

import numpy as np
import pandas as pd

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA = "0.1"

DATASETS = {"neu_s0": "NEU-DET", "gc10g_s0": "GC10-DET (grouped)", "gc10r_s0": "GC10-DET (random)",
            "pcbg_s0": "DeepPCB (grouped)", "pcbo": "DeepPCB (official)"}
DETECTORS = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n", "yolo12n": "YOLOv12n", "yolov5nu": "YOLOv5nu",
             "yolov5su": "YOLOv5su", "yolov8s": "YOLOv8s", "yolov10n": "YOLOv10n", "yolo11s": "YOLO11s",
             "rtdetrl": "RT-DETR-L", "frcnn": "Faster R-CNN", "secfull": "YOLOv12n-M",
             "secnosofl": "YOLOv12n-M (default loss)", "v12nbb": "YOLOv12n (backbone-only init.)",
             "hcsa": "Attention only", "bifpn": "BiFPN neck only", "ewbifpn": "Weighted BiFPN neck only",
             "v12nbbsofl": "Modified loss only"}
ORDER = list(DETECTORS)

# S1: (key, attention, neck, loss, initialisation, architecture key)
VARIANTS = [
    ("yolo12n", "-", "PAN", "default", "COCO, all layers", "v12n_pan"),
    ("v12nbb", "-", "PAN", "default", "COCO, backbone only", "v12n_pan"),
    ("hcsa", "yes", "PAN", "default", "COCO, backbone only", "sec_hcsa"),
    ("bifpn", "-", "BiFPN (96 ch.)", "default", "COCO, backbone only", "sec_bifpn"),
    ("ewbifpn", "-", "weighted BiFPN (96 ch.)", "default", "COCO, backbone only", "sec_ewbifpn"),
    ("v12nbbsofl", "-", "PAN", "modified", "COCO, backbone only", "v12n_pan"),
    ("secnosofl", "yes", "weighted BiFPN (96 ch.)", "default", "COCO, backbone only", "sec_full"),
    ("secfull", "yes", "weighted BiFPN (96 ch.)", "modified", "COCO, all matching layers", "sec_full"),
]

# S2: values of the cloud supplementary (CPU -> GPU); GPU values are re-checked below
DEVICE = [
    ("Magnetic Tile", "YOLO11n", "gpu_mt_s0_yolo11n_320_seed0", (38.8, 42.7), (26.8, 27.6), (100.0, 100.0)),
    ("NEU-DET", "YOLOv12n-M", "gpu_neu_s0_secfull_256_seed0", (31.3, 31.5), (37.4, 43.1), (69.2, 62.7)),
    ("NEU-DET", "YOLOv12n-M (default loss)", "gpu_neu_s0_secnosofl_256_seed0", (32.9, 33.7), (36.4, 34.9), (55.6, 60.7)),
    ("NEU-DET", "YOLOv12n (backbone-only init.)", "gpu_neu_s0_v12nbb_256_seed0", (30.1, 30.9), (41.6, 38.9), (61.1, 60.6)),
    ("NEU-DET", "Modified loss only", "gpu_neu_s0_v12nbbsofl_256_seed0", (21.0, 22.1), (59.2, 54.5), (73.0, 69.5)),
    ("NEU-DET", "YOLO11n", "gpu_neu_s0_yolo11n_256_seed0", (40.5, 39.7), (27.5, 26.9), (58.1, 53.8)),
    ("NEU-DET", "YOLOv12n", "gpu_neu_s0_yolo12n_256_seed0", (40.8, 39.5), (28.7, 28.3), (53.8, 51.2)),
]


def summ(run):
    S = pd.read_csv(f"{RES}/risk/{run}__test_summary.csv", dtype={"alpha": str})
    S["alpha"] = S["alpha"].fillna("")
    rows = {(m, a): r for m, a, r in zip(S.method, S.alpha, S.to_dict("records"))}
    return rows[("default_0.25", "")], rows[("ltt_review_union@fp0.5", ALPHA)]


def met(run):
    return json.load(open(f"{RES}/metrics/{run}_test.json"))


def pm(x, d=1):
    x = np.asarray(x, float)
    return f"{x.mean():.{d}f} ± {x.std(ddof=1):.{d}f}" if len(x) > 1 else f"{x.mean():.{d}f}"


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out)


def table_s1():
    # complexity of the fused model, from the same file as Table 5 of the paper (latency_d9.json); a variant without
    # an entry takes it from another variant with the same architecture (the loss does not change the network)
    lat = json.load(open(f"{RES}/latency_d9.json"))
    cx, by_arch = {}, {}
    for key, *_, ak in VARIANTS:
        e = lat.get(f"gpu_neu_s0_{key}_256_seed0")
        if e:
            cx[key] = (e["params"], e["gflops_at_imgsz"])
            by_arch.setdefault(ak, cx[key])
    same_build = {"v12nbbsofl": "v12nbb"}  # backbone-only initialisation, only the loss differs
    arch = {key: cx.get(key) or cx.get(same_build.get(key)) or by_arch[ak] for key, *_, ak in VARIANTS}
    rows = []
    for key, att, neck, loss, init, ak in VARIANTS:
        runs = [f"gpu_neu_s0_{key}_256_seed{s}" for s in range(3) if os.path.exists(f"{RES}/metrics/gpu_neu_s0_{key}_256_seed{s}_test.json")]
        M = [met(r) for r in runs]
        Sm = [summ(r) for r in runs]
        rows.append({"Variant": DETECTORS[key], "Attention": att, "Neck": neck, "Loss": loss, "Initialisation": init,
                     "Seeds": len(runs), "Params (M)": f"{arch[key][0] / 1e6:.2f}",
                     "GFLOPs (256 px)": f"{arch[key][1]:.2f}",
                     "mAP@0.5 (%)": pm([100 * m["map50"] for m in M]),
                     "mAP@0.5:0.95 (%)": pm([100 * m["map50_95"] for m in M]),
                     "Escape at 0.25 (%)": pm([100 * d["inst_escape"] for d, _ in Sm]),
                     "RC-Inspect: review (%)": pm([100 * u["review"] for _, u in Sm])})
    return pd.DataFrame(rows)


def table_s2():
    rows, checks = [], []
    for data, name, run, mp, esc, rev in DEVICE:
        m = 100 * met(run)["map50_95"]
        d, u = summ(run)
        checks.append((run, round(m, 1), mp[1], round(100 * d["inst_escape"], 1), esc[1], round(100 * u["review"], 1), rev[1]))
        rows.append({"Dataset": data, "Detector (seed 0)": name,
                     "mAP@0.5:0.95, CPU → GPU (%)": f"{mp[0]:.1f} → {mp[1]:.1f}",
                     "Escape at 0.25, CPU → GPU (%)": f"{esc[0]:.1f} → {esc[1]:.1f}",
                     "RC-Inspect review, CPU → GPU (%)": f"{rev[0]:.1f} → {rev[1]:.1f}"})
    return pd.DataFrame(rows), checks


def table_s3():
    A = pd.read_csv(f"{RES}/risk_extra/auto_escape.csv")
    A["ds"] = A.run.str.extract(r"^gpu_(neu_s0|gc10g_s0|gc10r_s0|pcbg_s0|pcbo)_")[0]
    A["det"] = A.run.str.extract(r"_([a-z0-9]+)_\d+_seed\d$")[0]
    rows = []
    for ds in DATASETS:
        for det in ORDER:
            g = A[(A.ds == ds) & (A.det == det)]
            if not len(g):
                continue
            # escape among non-reviewed images exists only in feasible partitions: weight the seeds by their number
            w = g.feasible.to_numpy()
            wm = lambda c: np.average(g[c].to_numpy(), weights=w)
            rows.append({"Dataset": DATASETS[ds], "Detector": DETECTORS[det], "Seeds": len(g),
                         "Escape, all defective images (%)": f"{100 * g.escape_all.mean():.1f}",
                         "Review (%)": f"{100 * g.review.mean():.1f}",
                         "Escape among non-reviewed images (%)": f"{100 * wm('escape_auto'):.1f}",
                         "90th percentile (%)": f"{100 * wm('escape_auto_q90'):.1f}",
                         "Partitions above 10 % (%)": f"{100 * wm('auto_above_alpha'):.0f}",
                         "Defective images accepted (%)": f"{100 * g.defective_accepted.mean():.2f}"})
    return pd.DataFrame(rows)


def main():
    s1, (s2, checks), s3 = table_s1(), table_s2(), table_s3()
    print("S2 GPU check (run, mAP local vs table, escape local vs table, review local vs table):")
    for c in checks:
        print(" ", c)
    txt = f"""# Supplementary material

RC-Inspect: A lightweight risk-controlled deep learning framework for real-time defect inspection in steel and
electronic manufacturing

Code, split lists, cached predictions and the result files behind these tables: https://doi.org/10.5281/zenodo.24269095

## Table S1. YOLOv12n variants on NEU-DET

All variants share the YOLOv12n backbone and the training protocol of Table 3. Attention: channel–spatial attention on
the three backbone outputs. Weighted BiFPN: bidirectional feature pyramid with edge-weighted fusion nodes. Modified
loss: modified box-regression terms with a focal classification loss. Values are means ± standard deviations over
seeds (test partition); escape at the default threshold of 0.25 and review of RC-Inspect ($\\alpha=0.10$, $\\delta=0.1$,
false-alarm budget 0.5) are means over 500 calibration/evaluation partitions per seed. GFLOPs at the 256-pixel input
used for NEU-DET; Params and GFLOPs of the fused model, as in Table 5 of the paper.

{md(s1)}

## Table S2. Device check

Seed-0 models trained on a 4-core CPU and on the GPU with the same seed, split and settings. Training on a different
device is not bit-identical; all results in the paper use the GPU-trained models. On Magnetic Tile, RC-Inspect sends
every image to review (Section 4.5.3).

{md(s2)}

## Table S3. Escape among the images decided without review

RC-Inspect ($\\alpha=0.10$, $\\delta=0.1$, false-alarm budget 0.5), at the operating points of the main analysis (500
partitions per seed; means over seeds). The guarantee of Eq. (4) concerns the escape over all defective images, with
reviewed images counting zero. This table reports the escape among the defective images that are accepted or rejected
without review, which is not guaranteed by the method, and the share of defective images that are accepted.
Partitions without a feasible operating point (full review) are excluded from the escape among non-reviewed images,
its 90th percentile and the share of partitions above 10 %, and the seeds are weighted by their number of feasible
partitions in these three columns, so that the 90th percentile is a weighted mean of the per-seed percentiles (Faster R-CNN on NEU-DET has a feasible point in 79 %, 0.6 % and 100 % of the
partitions of its three seeds).

{md(s3)}
"""
    os.makedirs(f"{RES}/tables", exist_ok=True)
    open(f"{RES}/tables/supplementary.md", "w", encoding="utf-8").write(txt)
    print(md(s1)); print(md(s3))


if __name__ == "__main__":
    main()
