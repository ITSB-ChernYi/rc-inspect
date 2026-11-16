"""Escape among the images decided without review, at the operating points of RC-Inspect (Learn-then-Test, union
review rule, alpha 0.10, delta 0.1, false-alarm budget 0.5) already selected by scripts/analyze_risk.py.

For every GPU-trained run (Magnetic Tile excluded: no operating point without full review) the partitions of the
main analysis are rebuilt with the same permutation stream (seed 2026, as scripts/analyze_review_extra.py) and the
selected (lambda, rho, rule) of each partition is read from results/risk/<run>__test.csv. Per partition:
  escape_all   mean over defective evaluation images of the missed share, reviewed images counting zero (Eq. 4)
  escape_auto  the same mean over the defective images that are accepted or rejected without review
  accepted     share of defective evaluation images accepted (no box at the threshold) without review
Partitions without a feasible operating point (full review) have no automatic decisions and are skipped for
escape_auto. Output: results/risk_extra/auto_escape.csv (one row per run).

usage: python scripts/auto_escape.py
"""
import glob
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from sdd import risk  # noqa: E402
from sdd.paths import DATA  # noqa: E402
from analyze_risk import LAMS, size_edges_from_train  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ALPHA, METHOD = 0.10, "ltt_review_union@fp0.5"


def one(run):
    D = pickle.load(open(f"{DATA}/preds/{run}/test.pkl", "rb"))
    rec = risk.summarise(D["pred"], D["gt"], class_aware=False, fragment=True)
    tab = risk.Table(rec, LAMS, size_edges=size_edges_from_train(run))
    n = len(rec)
    R = pd.read_csv(f"{RES}/risk/{run}__test.csv")
    U = R[(R.method == METHOD) & np.isclose(R.alpha, ALPHA)].set_index("rep")
    rng = np.random.default_rng(2026)
    e_all, e_auto, acc, rev, feas = [], [], [], [], []
    for rep in range(len(U)):
        perm = rng.permutation(n)
        ev = perm[int(round(0.5 * n)):]
        row = U.loc[rep]
        d = tab.defective[ev]
        if not bool(row.feasible):
            e_all.append(0.0); acc.append(0.0); rev.append(1.0); feas.append(False)
            continue
        j = int(np.argmin(np.abs(LAMS - row.lam)))
        r = tab.review(ev, float(row.rho), row.rule)[:, j]
        auto = ~r
        miss = tab.miss_frac[ev, j]
        e_all.append(float((auto * miss)[d].mean()))
        if (auto & d).any():
            e_auto.append(float(miss[auto & d].mean()))
        acc.append(float((auto & tab.empty[ev, j] & d).sum() / max(d.sum(), 1)))
        rev.append(float(r.mean())); feas.append(True)
    e_auto = np.array(e_auto)
    return {"run": run, "reps": len(U), "feasible": np.mean(feas), "escape_all": np.mean(e_all), "review": np.mean(rev),
            "escape_auto": e_auto.mean() if len(e_auto) else np.nan,
            "escape_auto_q90": np.quantile(e_auto, 0.9) if len(e_auto) else np.nan,
            "auto_above_alpha": np.mean(e_auto > ALPHA) if len(e_auto) else np.nan,
            "defective_accepted": np.mean(acc)}


def main():
    runs = sorted(re.sub(r"__test\.csv$", "", os.path.basename(f)) for f in glob.glob(f"{RES}/risk/gpu_*__test.csv"))
    runs = [r for r in runs if "_mt_" not in r]
    rows = []
    for r in runs:
        rows.append(one(r))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(f"{RES}/risk_extra/auto_escape.csv", index=False)


if __name__ == "__main__":
    main()
