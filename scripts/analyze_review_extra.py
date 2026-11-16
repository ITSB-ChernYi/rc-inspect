"""Additional analyses of the guaranteed operating point (RC-Inspect: Learn-then-Test, union review rule,
false-alarm budget 0.5), from cached predictions only. Written for the simulated review (G3).

For one run and alpha = 0.10:
 (a) imperfect review: station escape = escape of automatic decisions + m * (missed share of the defects in reviewed
     images), for an operator who misses each defect independently with probability m; evaluated at the operating
     points already selected by scripts/analyze_risk.py (same partitions, read from results/risk/<run>__test.csv);
 (b) per-class escape at those operating points (pooled over defects of a class);
 (c) operator-aware calibration: Learn-then-Test with the per-image loss auto * miss_frac + review * m, so that the
     guarantee covers the station escape for an assumed operator miss rate m;
 (d) grouped partitions: calibration and evaluation halves drawn by acquisition sequence (GC10-DET), tile group
     (Magnetic Tile) or board design (DeepPCB) instead of by image.
(c) and (d) use --reps partitions (default 200). Output: results/risk_extra/<run>.json

usage: python scripts/analyze_review_extra.py <run> [--reps 200] [--m 0.05 0.1 0.2]
"""
import argparse
import json
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
from analyze_risk import LAMS, RHO_RESID, RHO_MAXCONF, size_edges_from_train  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
UNION = [("resid", r) for r in RHO_RESID] + [("maxconf", r) for r in RHO_MAXCONF]
ALPHA, DELTA, BUDGET = 0.10, 0.1, 0.5


class OpTable(risk.Table):
    """Table whose escape loss also charges reviewed defective images with an operator miss rate m."""
    m = 0.0

    def losses(self, idx, rho=None, kind="resid"):
        Lo = super().losses(idx, rho, kind)
        Lo["inst"] = Lo["inst"] + self.m * Lo["review"] * self.defective[idx, None]
        return Lo


def groups_of(run, stems):
    if "_gc10" in run:
        return [re.match(r"img_(\d+_[A-Za-z]*\d+)_\d+$", s).group(1) for s in stems]
    if "_pcb" in run:
        return [s[:5] for s in stems]
    if "_mt_" in run:
        g = json.load(open(f"{RES}/audit/mt_tile_groups.json"))["groups"]
        return [g.get(s, s) for s in stems]
    return None


def pick(tab, cal):
    _, valid = risk.ltt_select(tab, cal, {"inst": ALPHA}, UNION, delta=DELTA, kind="resid")
    cand = [v for v in valid if v[3] <= BUDGET]
    return min(cand, key=lambda t: (t[2], t[3])) if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--m", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2])   # (a) operator miss rates
    ap.add_argument("--aware", type=float, nargs="*", default=[0.05])   # (c) operator miss rates for calibration
    ap.add_argument("--no-ltt", action="store_true")   # skip (c) and (d)
    a = ap.parse_args()
    D = pickle.load(open(f"{DATA}/preds/{a.run}/test.pkl", "rb"))
    rec = risk.summarise(D["pred"], D["gt"], class_aware=False, fragment=True)
    tab = OpTable(rec, LAMS, size_edges=size_edges_from_train(a.run))
    n, names = len(rec), D["names"]
    nc = len(names)
    # per-image, per-class defect counts and misses at every lam
    ncls = np.zeros((n, nc))
    mcls = np.zeros((n, nc, len(LAMS)))
    for i, r in enumerate(rec):
        for c in range(nc):
            sel = r["gt_cls"] == c
            ncls[i, c] = sel.sum()
            if sel.any():
                mcls[i, c] = (r["lstar"][sel][None, :] < LAMS[:, None]).sum(1)
    out = {"run": a.run, "alpha": ALPHA, "delta": DELTA, "budget": BUDGET, "class_names": [names[c] for c in range(nc)]}

    # (a), (b): operating points of the main analysis (500 partitions, same permutation stream)
    R = pd.read_csv(f"{RES}/risk/{a.run}__test.csv")
    U = R[(R.method == "ltt_review_union@fp0.5") & (R.alpha == ALPHA)].set_index("rep")
    rng = np.random.default_rng(2026)
    reps = len(U)
    st = {m: [] for m in [0.0] + a.m}
    rev_def, rev_all, cls_miss, cls_n, acc_def = [], [], np.zeros(nc), np.zeros(nc), []
    rev_free, fp_auto, pooled, feas_main = [], [], [], []
    for rep in range(reps):
        perm = rng.permutation(n)
        ev = perm[int(round(0.5 * n)):]
        row = U.loc[rep]
        d = tab.defective[ev]
        if not bool(row.feasible):
            for m in st:
                st[m].append(m * 1.0)   # every image reviewed: the station escape is the operator's
            rev_def.append(1.0); rev_all.append(1.0); acc_def.append(0.0); feas_main.append(False)
            if (~d).any():
                rev_free.append(1.0)
            pooled.append(0.0)
            cls_n += ncls[ev].sum(0)
            continue
        feas_main.append(True)
        j = int(np.argmin(np.abs(LAMS - row.lam)))
        r = tab.review(ev, float(row.rho), row.rule)[:, j]
        auto = ~r
        for m in st:
            st[m].append(float((auto * tab.miss_frac[ev, j] + m * r)[d].mean()))
        rev_def.append(float(r[d].mean())); rev_all.append(float(r.mean()))
        acc_def.append(float((auto & tab.empty[ev, j] & d).sum() / max(d.sum(), 1)))
        if (~d).any():
            rev_free.append(float(r[~d].mean()))
        fp_auto.append(float((auto * tab.fp[ev, j]).sum() / max(auto.sum(), 1)))
        pooled.append(float((auto * tab.missed[ev, j]).sum() / max(tab.n_gt[ev].sum(), 1)))
        cls_miss += (auto[:, None] * mcls[ev, :, j]).sum(0)
        cls_n += ncls[ev].sum(0)
    out["imperfect_review"] = {f"{m}": {"escape": float(np.mean(v)), "violations": float(np.mean(np.array(v) > ALPHA))}
                               for m, v in st.items()}
    out["review_share_defective"] = float(np.mean(rev_def))
    out["review_share_all"] = float(np.mean(rev_all))
    out["defective_images_accepted"] = float(np.mean(acc_def))
    out["review_share_defect_free"] = float(np.mean(rev_free)) if rev_free else None
    out["false_alarms_per_auto_image"] = float(np.mean(fp_auto)) if fp_auto else None
    out["pooled_escape"] = float(np.mean(pooled))
    out["feasible_main"] = float(np.mean(feas_main))
    out["class_escape"] ={names[c]: float(cls_miss[c] / max(cls_n[c], 1)) for c in range(nc)}
    out["reps_main"] = reps

    # (c) operator-aware calibration
    out["operator_aware"] = {}
    for m in ([] if a.no_ltt else a.aware):
        tab.m = m
        rng = np.random.default_rng(2026)
        esc, viol, rev, feas = [], [], [], []
        for rep in range(a.reps):
            perm = rng.permutation(n)
            cal, ev = perm[:int(round(0.5 * n))], perm[int(round(0.5 * n)):]
            b = pick(tab, cal)
            d = tab.defective[ev]
            if b is None:
                esc.append(m); rev.append(1.0); feas.append(False)
            else:
                (k, rho), lam = b[0], b[1]
                j = int(np.argmin(np.abs(LAMS - lam)))
                Lo = tab.losses(ev, rho, k)
                esc.append(float(Lo["inst"][d, j].mean())); rev.append(float(Lo["review"][:, j].mean())); feas.append(True)
            viol.append(esc[-1] > ALPHA)
        out["operator_aware"][f"{m}"] = {"escape": float(np.mean(esc)), "violations": float(np.mean(viol)),
                                         "review": float(np.mean(rev)), "feasible": float(np.mean(feas))}
    tab.m = 0.0

    # (d) grouped partitions
    stems = [r["stem"] for r in rec]
    G = groups_of(a.run, stems)
    if G is not None and not a.no_ltt:
        ug = sorted(set(G))
        gi = np.array([ug.index(g) for g in G])
        rng = np.random.default_rng(2026)
        esc, viol, rev, feas, ncal = [], [], [], [], []
        for rep in range(a.reps):
            order = rng.permutation(len(ug))
            sizes = np.bincount(gi, minlength=len(ug))[order]
            k = int(np.searchsorted(np.cumsum(sizes), n / 2)) + 1
            calg = set(order[:k].tolist())
            cal = np.array([i for i in range(n) if gi[i] in calg])
            ev = np.array([i for i in range(n) if gi[i] not in calg])
            ncal.append(len(cal))
            b = pick(tab, cal)
            d = tab.defective[ev]
            if b is None:
                esc.append(0.0); rev.append(1.0); feas.append(False)
            else:
                (kk, rho), lam = b[0], b[1]
                j = int(np.argmin(np.abs(LAMS - lam)))
                Lo = tab.losses(ev, rho, kk)
                esc.append(float(Lo["inst"][d, j].mean())); rev.append(float(Lo["review"][:, j].mean())); feas.append(True)
            viol.append(esc[-1] > ALPHA)
        out["grouped_partitions"] = {"n_groups": len(ug), "escape": float(np.mean(esc)), "violations": float(np.mean(viol)),
                                     "review": float(np.mean(rev)), "feasible": float(np.mean(feas)),
                                     "cal_images_mean": float(np.mean(ncal))}
    os.makedirs(f"{RES}/risk_extra", exist_ok=True)
    json.dump(out, open(f"{RES}/risk_extra/{a.run}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("class_names",)}, indent=1))


if __name__ == "__main__":
    main()
