"""Paired image bootstrap (2,000 resamples) of the mAP difference between the three stock detectors on NEU-DET,
per seed, from the cached deployment predictions. Same procedure as scripts/analyze_detection.py, restricted to the
pairs reported in Section 4.10 (G3, R1 W12). Output: results/tables/detection_pairs_stock.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import evalmap  # noqa: E402
from sdd.paths import DATA  # noqa: E402

NB = 2000
DETS = ["yolov8n", "yolo11n", "yolo12n"]


def main():
    rows = []
    for seed in (0, 1, 2):
        P = {}
        for d in DETS:
            D = pickle.load(open(f"{DATA}/preds/gpu_neu_s0_{d}_256_seed{seed}/test.pkl", "rb"))
            P[d] = evalmap.prepare(D["pred"], D["gt"], len(D["names"]))
        for i, a in enumerate(DETS):
            for b in DETS[i + 1:]:
                n = P[a]["n_img"]
                rng = np.random.default_rng(seed + 7)
                d50, d5095 = [], []
                for _ in range(NB):
                    w = rng.multinomial(n, np.full(n, 1 / n))
                    sa, sb = evalmap.summary(P[a], w), evalmap.summary(P[b], w)
                    d50.append(sb["map50"] - sa["map50"])
                    d5095.append(sb["map50_95"] - sa["map50_95"])
                sa, sb = evalmap.summary(P[a]), evalmap.summary(P[b])
                rows.append({"seed": seed, "a": a, "b": b, "diff_map50_95": sb["map50_95"] - sa["map50_95"],
                             "ci_lo": np.quantile(d5095, .025), "ci_hi": np.quantile(d5095, .975),
                             "diff_map50": sb["map50"] - sa["map50"], "ci50_lo": np.quantile(d50, .025),
                             "ci50_hi": np.quantile(d50, .975)})
                print(rows[-1], flush=True)
    os.makedirs("results/tables", exist_ok=True)
    pd.DataFrame(rows).to_csv("results/tables/detection_pairs_stock.csv", index=False)


if __name__ == "__main__":
    main()
