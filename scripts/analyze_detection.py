"""Detection accuracy across detectors and seeds, with paired image-bootstrap intervals.

Reads results/metrics/<run>_test.json (Ultralytics validator, multi-label NMS; comparable with the literature)
and the cached deployment predictions /home/user/obj1data/preds/<run>/test.pkl (single-label NMS; used for
paired differences). Output: results/tables/detection_runs.csv, detection_summary.csv, detection_pairs.csv,
per_class.csv, escape_default.csv
"""
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import evalmap, risk  # noqa: E402
from sdd.paths import DATA  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MET, TAB = os.path.join(ROOT, "results", "metrics"), os.path.join(ROOT, "results", "tables")
PREDS = f"{DATA}/preds"
PAT = re.compile(r"^(?P<ds>neu_s0|mt_s0|gc10g_s0|gc10r_s0)_(?P<det>[a-z0-9]+)_(?P<sz>\d+)_seed(?P<seed>\d+)$")
NB = 2000


def runs():
    out = []
    for f in sorted(glob.glob(f"{MET}/*_test.json")):
        r = os.path.basename(f)[:-10]
        m = PAT.match(r)
        if m and os.path.exists(f"{PREDS}/{r}/test.pkl"):
            out.append({"run": r, **m.groupdict()})
    return pd.DataFrame(out)


def main():
    os.makedirs(TAB, exist_ok=True)
    R = runs()
    rows, pcs, esc, prepared = [], [], [], {}
    for _, r in R.iterrows():
        u = json.load(open(f"{MET}/{r.run}_test.json"))
        v = json.load(open(f"{MET}/{r.run}_val.json"))
        D = pickle.load(open(f"{PREDS}/{r.run}/test.pkl", "rb"))
        nc = len(D["names"])
        P = evalmap.prepare(D["pred"], D["gt"], nc)
        prepared[r.run] = P
        s = evalmap.summary(P)
        rows.append({**r.to_dict(), "test_map50": u["map50"], "test_map50_95": u["map50_95"], "test_P": u["precision"],
                     "test_R": u["recall"], "val_map50": v["map50"], "val_map50_95": v["map50_95"],
                     "dep_map50": s["map50"], "dep_map50_95": s["map50_95"]})
        for k, (name, d) in enumerate(u["per_class"].items()):
            pcs.append({**r.to_dict(), "class": name, "ap50": d["ap50"], "ap50_95": d["ap50_95"]})
        rec = risk.summarise(D["pred"], D["gt"])
        lstar = np.concatenate([x["lstar"] for x in rec])
        cls = np.concatenate([x["gt_cls"] for x in rec])
        for c in range(nc):
            esc.append({**r.to_dict(), "class": D["names"][c], "escape_at_0.25": float((lstar[cls == c] < 0.25).mean())
                        if (cls == c).any() else np.nan})
        esc.append({**r.to_dict(), "class": "all", "escape_at_0.25": float((lstar < 0.25).mean())})
    T = pd.DataFrame(rows)
    T.to_csv(f"{TAB}/detection_runs.csv", index=False)
    pd.DataFrame(pcs).to_csv(f"{TAB}/per_class.csv", index=False)
    pd.DataFrame(esc).to_csv(f"{TAB}/escape_default.csv", index=False)
    S = T.groupby(["ds", "det", "sz"]).agg(n_seeds=("seed", "count"),
                                           **{f"{m}_{f}": (m, f) for m in ["test_map50", "test_map50_95", "val_map50_95",
                                                                            "dep_map50", "dep_map50_95"] for f in ("mean", "std")}).reset_index()
    S.to_csv(f"{TAB}/detection_summary.csv", index=False)
    # paired differences (same dataset, same seed): image bootstrap of deployment-prediction mAP
    pairs = []
    for ds, G in T.groupby("ds"):
        dets = sorted(G.det.unique())
        for i, a in enumerate(dets):
            for b in dets[i + 1:]:
                for seed in sorted(set(G[G.det == a].seed) & set(G[G.det == b].seed)):
                    ra = G[(G.det == a) & (G.seed == seed)].run.iloc[0]
                    rb = G[(G.det == b) & (G.seed == seed)].run.iloc[0]
                    Pa, Pb = prepared[ra], prepared[rb]
                    n = Pa["n_img"]
                    rng = np.random.default_rng(int(seed) + 7)
                    d50, d5095 = [], []
                    for _ in range(NB):
                        w = rng.multinomial(n, np.full(n, 1 / n))
                        sa, sb = evalmap.summary(Pa, w), evalmap.summary(Pb, w)
                        d50.append(sb["map50"] - sa["map50"])
                        d5095.append(sb["map50_95"] - sa["map50_95"])
                    sa, sb = evalmap.summary(Pa), evalmap.summary(Pb)
                    pairs.append({"ds": ds, "a": a, "b": b, "seed": seed,
                                  "diff_map50": sb["map50"] - sa["map50"], "ci50_lo": np.quantile(d50, .025), "ci50_hi": np.quantile(d50, .975),
                                  "diff_map50_95": sb["map50_95"] - sa["map50_95"], "ci5095_lo": np.quantile(d5095, .025),
                                  "ci5095_hi": np.quantile(d5095, .975)})
    pd.DataFrame(pairs).to_csv(f"{TAB}/detection_pairs.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(S.round(4).to_string())
        print(pd.DataFrame(pairs).round(4).to_string())


if __name__ == "__main__":
    main()
