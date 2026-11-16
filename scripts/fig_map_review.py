"""Test mAP@0.5:0.95 against (a) escape at the default threshold and (b) review rate of the proposed module
(alpha = 0.10, delta = 0.1, false-alarm budget 0.5), for every NEU-DET model and seed trained on one device.

Redraws the earlier mAP-versus-review figure with the variant names used in the manuscript.
Reads results/metrics/gpu_neu_s0_*_test.json and results/risk/gpu_neu_s0_*__test_summary.csv.
Writes results/figures/fig_map_vs_review.{png,pdf}.
"""
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from make_figures import C, INK, save

RES = os.path.join(os.path.dirname(__file__), "..", "results")
# key in run name -> (label in the manuscript, colour, marker)
M = {"yolov8n": ("YOLOv8n", C[0], "o"), "yolo11n": ("YOLO11n", C[1], "o"), "yolo12n": ("YOLOv12n", C[2], "o"),
     "secfull": ("YOLOv12n-M", C[3], "s"), "secnosofl": ("YOLOv12n-M (default loss)", C[4], "s"),
     "v12nbb": ("YOLOv12n (backbone-only init.)", C[5], "s"),
     "hcsa": ("Attention only", "#E6C700", "^"), "bifpn": ("BiFPN neck only", "#8a8a8a", "^"),
     "ewbifpn": ("Weighted BiFPN neck only", "#000000", "^"), "v12nbbsofl": ("Modified loss only", "#A0522D", "^"),
     # additional detectors of Section 4.4.1 (hollow markers)
     "yolov5nu": ("YOLOv5nu", C[0], "D"), "yolov5su": ("YOLOv5su", C[1], "D"), "yolov8s": ("YOLOv8s", C[2], "D"),
     "yolov10n": ("YOLOv10n", C[3], "D"), "yolo11s": ("YOLO11s", C[4], "D"), "rtdetrl": ("RT-DETR-L", C[5], "P"),
     "frcnn": ("Faster R-CNN", "#000000", "X")}
HOLLOW = {"yolov5nu", "yolov5su", "yolov8s", "yolov10n", "yolo11s", "rtdetrl", "frcnn"}


def main():
    rows = []
    for f in sorted(glob.glob(f"{RES}/metrics/gpu_neu_s0_*_256_seed*_test.json")):
        run = json.load(open(f))["run"]
        key = re.match(r"gpu_neu_s0_(.+)_256_seed\d", run).group(1)
        S = pd.read_csv(f"{RES}/risk/{run}__test_summary.csv")
        esc = S[S.method == "default_0.25"].inst_escape.iloc[0]
        rev = S[(S.method == "ltt_review_union@fp0.5") & (S.alpha == 0.1)].review.iloc[0]
        rows.append({"run": run, "key": key, "map": 100 * json.load(open(f))["map50_95"], "esc": 100 * esc,
                     "rev": 100 * rev})
    D = pd.DataFrame(rows)
    assert set(D.key) == set(M), set(D.key) ^ set(M)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.5))
    for key, (label, col, mk) in M.items():
        d = D[D.key == key]
        for k, y in enumerate(("esc", "rev")):
            ax[k].plot(d["map"], d[y], mk, color=col, markersize=5, label=label if k == 0 else None, linestyle="none",
                       markerfacecolor="none" if key in HOLLOW else col, markeredgewidth=1.1)
    ax[0].set(xlabel="Test mAP@0.5:0.95 (%)", ylabel="Escape at the default threshold (%)")
    ax[1].set(xlabel="Test mAP@0.5:0.95 (%)", ylabel="Images sent to review (%)")
    ax[0].set_title("(a)", loc="left", fontsize=9, color=INK)
    ax[1].set_title("(b)", loc="left", fontsize=9, color=INK)
    fig.legend(loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    save(fig, "fig_map_vs_review")
    print(D.groupby("key")[["map", "esc", "rev"]].agg(["min", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
