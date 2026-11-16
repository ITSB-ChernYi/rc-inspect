"""Risk-controlled inspection: evaluation over repeated calibration/evaluation partitions.

usage: python scripts/analyze_risk.py <run> [--pool test] [--reps 500] [--alphas 0.05 0.1] [--delta 0.1]
       [--risks inst] [--fa 0.05] [--corrupt noise3,...]

The held-out pool (default: the untouched test partition) is split at random into two halves `reps` times:
calibration (choose thresholds) and evaluation (measure realised risks). The model and all settings below
are fixed before the test partition is opened; method development used --pool val.
Outputs results/risk/<run>__<pool>.csv (one row per repetition x method x alpha) and a summary JSON.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import risk  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

PREDS = f"{DATA}/preds"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "risk")
LAMS = np.round(np.concatenate([[0.001, 0.002, 0.003, 0.004], np.arange(0.005, 0.951, 0.005)]), 3)
# when a target cannot be certified the station must review every image (Section 4.2)
REVIEW_ALL = {"inst_escape": 0.0, "inst_escape_pooled": 0.0, "part_escape": 0.0, "false_reject": 0.0,
              "review_rate": 1.0, "fp_per_image": 0.0, "lam": np.nan, "rho": None, "size_escape": None}
RHO_RESID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RHO_MAXCONF = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
RHO_RANDOM = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def size_edges_from_train(run):
    """Defect-size strata: tertiles of ground-truth box area in the TRAINING labels (never test)."""
    rec = json.load(open(f"{DATA}/runs/{run}/run_record.json"))
    import yaml
    root = dataset_root(rec)
    from PIL import Image
    areas = []
    lab = os.path.join(root, "labels", "train")
    for f in sorted(os.listdir(lab))[:2000]:
        a = np.loadtxt(os.path.join(lab, f), ndmin=2)
        if not len(a):
            continue
        W, H = Image.open(os.path.join(root, "images", "train", f[:-4] + ".jpg")).size
        areas += list(a[:, 3] * W * a[:, 4] * H)
    return list(np.quantile(areas, [1 / 3, 2 / 3]))


def f1_threshold(tab, idx):
    """Threshold maximising instance F1 (found / (found + FP) vs found / n_gt) on the given images."""
    Lo = tab.losses(idx)
    found = (tab.n_gt[idx].sum() - Lo["missed"].sum(0))
    prec = found / np.maximum(found + Lo["fp"].sum(0), 1)
    rec = found / max(tab.n_gt[idx].sum(), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    return float(tab.lams[int(np.argmax(f1))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pool", default="test")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--fa", type=float, default=None)     # also control false-reject rate on defect-free images
    ap.add_argument("--class-aware", action="store_true")
    ap.add_argument("--strict", action="store_true")     # criterion ablation: IoU match only, no fragment covering
    ap.add_argument("--fp-budgets", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    ap.add_argument("--cal-frac", type=float, default=0.5)   # share of the pool used for calibration
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    D = pickle.load(open(f"{PREDS}/{a.run}/{a.pool}.pkl", "rb"))
    V = pickle.load(open(f"{PREDS}/{a.run}/val.pkl", "rb"))
    edges = size_edges_from_train(a.run)
    rec = risk.summarise(D["pred"], D["gt"], class_aware=a.class_aware, fragment=not a.strict)
    tab = risk.Table(rec, LAMS, size_edges=edges)
    vtab = risk.Table(risk.summarise(V["pred"], V["gt"], class_aware=a.class_aware, fragment=not a.strict), LAMS)
    lam_f1 = f1_threshold(vtab, np.arange(len(vtab.n_gt)))
    nc = len(D["names"])
    ctab = risk.ClassTable(risk.summarise(D["pred"], D["gt"], class_aware=True, fragment=not a.strict), LAMS, nc)   # chosen on validation predictions, as practitioners do
    n = len(rec)
    rng = np.random.default_rng(2026)
    rows, frontier = [], []
    has_free = bool((~tab.defective).any())
    order = "desc" if (a.fa is not None and has_free) else "asc"
    for rep in range(a.reps):
        perm = rng.permutation(n)
        nc_ = int(round(a.cal_frac * n))
        cal, ev = perm[:nc_], perm[nc_:]
        base = {"rep": rep}
        rows.append({**base, "method": "default_0.25", "alpha": np.nan, **risk.evaluate(tab, ev, 0.25)})
        rows.append({**base, "method": "f1_val", "alpha": np.nan, **risk.evaluate(tab, ev, lam_f1)})
        for alpha in a.alphas:
            Lc = tab.losses(cal)
            d = tab.defective[cal]
            # naive: largest lam whose empirical calibration escape <= alpha (no finite-sample correction)
            emp = Lc["inst"][d].mean(0) <= alpha
            j = np.nonzero(emp)[0]
            j = j[j == np.arange(len(j))]
            lam_emp = LAMS[j[-1]] if len(j) else LAMS[0]
            rows.append({**base, "method": "empirical", "alpha": alpha, **risk.evaluate(tab, ev, lam_emp)})
            # conformal risk control, marginal guarantee E[escape] <= alpha, no review
            lam_crc = risk.crc_threshold(Lc["inst"][d], LAMS, alpha, return_none=True)
            rows.append({**base, "method": "crc", "alpha": alpha, "feasible": lam_crc is not None,
                         **(risk.evaluate(tab, ev, lam_crc) if lam_crc is not None else REVIEW_ALL)})
            # size-conditional CRC: per-stratum escape (pooled within stratum, bounded per image), min threshold
            lams_k = []
            for mk, nk in tab.strata:
                sel = nk[cal] > 0
                lk = mk[cal][sel] / nk[cal][sel, None]
                lams_k.append(risk.crc_threshold(lk, LAMS, alpha, return_none=True))
            ok_k = all(v is not None for v in lams_k)
            rows.append({**base, "method": "crc_size", "alpha": alpha, "feasible": ok_k,
                         **(risk.evaluate(tab, ev, min(lams_k)) if ok_k else REVIEW_ALL)})
            # class-specific thresholds, each class guaranteed (class-aware criterion)
            lc = ctab.crc_thresholds(cal, alpha)
            if any(v is None for v in lc):
                rows.append({**base, "method": "crc_class", "alpha": alpha, "feasible": False, **REVIEW_ALL})
            else:
                ce = ctab.evaluate(ev, lc)
                rows.append({**base, "method": "crc_class", "alpha": alpha, "feasible": True, "inst_escape": ce["inst_escape"],
                             "inst_escape_pooled": ce["inst_escape_pooled"], "fp_per_image": ce["fp_per_image"],
                             "review_rate": 0.0, "class_escape": ce["class_escape_img"], "lam_c": ce["lam_c"]})
            # Learn-then-Test, high-probability guarantee P(escape <= alpha) >= 1 - delta
            risks = {"inst": alpha}
            if a.fa is not None and has_free:
                risks["fa"] = a.fa
            UNION = [("resid", r) for r in RHO_RESID] + [("maxconf", r) for r in RHO_MAXCONF]
            for kind, grid in (("none", [None]), ("resid", RHO_RESID), ("maxconf", RHO_MAXCONF), ("random", RHO_RANDOM),
                               ("union", UNION)):
                k2 = "resid" if kind in ("none", "union") else kind
                _, valid = risk.ltt_select(tab, cal, risks, grid, delta=a.delta, kind=k2, order=order)
                name = "ltt" if kind == "none" else f"ltt_review_{kind}"
                if kind == "none":
                    # no review: the valid threshold with the fewest calibration false alarms (largest lam)
                    budgets = [None]
                else:
                    budgets = a.fp_budgets
                for fb in budgets:
                    cand = [v for v in valid if fb is None or v[3] <= fb]
                    tagm = name if fb is None else f"{name}@fp{fb}"
                    if not cand:
                        rows.append({**base, "method": tagm, "alpha": alpha, "feasible": False, **REVIEW_ALL})
                        continue
                    best = min(cand, key=lambda t: (t[2], t[3]))
                    rows.append({**base, "method": tagm, "alpha": alpha, "feasible": True,
                                 "rule": best[0][0] if isinstance(best[0], tuple) else kind,
                                 **risk.evaluate(tab, ev, best[1], best[0], kind=k2)})
                if kind != "none" and rep < 50:
                    for v in valid:   # frontier data (first 50 repetitions)
                        frontier.append({"rep": rep, "alpha": alpha, "kind": kind, "rho": str(v[0]), "lam": v[1],
                                         "review_cal": v[2], "fp_cal": v[3]})
        if rep % 100 == 0:
            print(a.run, rep, flush=True)
    R = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    tag = f"__{a.pool}{a.tag}{'_ca' if a.class_aware else ''}{'_strict' if a.strict else ''}" + (f"_cal{a.cal_frac}" if a.cal_frac != 0.5 else "")
    pd.DataFrame(frontier).to_csv(f"{OUT}/{a.run}{tag}_frontier.csv", index=False)
    R.drop(columns=["size_escape", "class_escape", "lam_c"], errors="ignore").to_csv(f"{OUT}/{a.run}{tag}.csv", index=False)
    S = R.copy()
    for k in range(len(edges) + 1):
        S[f"size{k}_escape"] = S.size_escape.map(lambda v, k=k: v[k] if isinstance(v, list) and len(v) > k else np.nan)
    CE = S[(S.method == "crc_class")][["alpha", "class_escape"]].dropna().copy() if "class_escape" in S else pd.DataFrame()
    if len(CE):
        ce_mean = CE.groupby("alpha").class_escape.apply(lambda v: np.nanmean(np.stack(v.values), 0).round(4).tolist()).to_dict()
    else:
        ce_mean = {}
    S = S.drop(columns=["size_escape", "class_escape", "lam_c"], errors="ignore")
    agg = S.groupby(["method", "alpha"], dropna=False).agg(
        inst_escape=("inst_escape", "mean"), inst_escape_q90=("inst_escape", lambda x: x.quantile(0.9)),
        viol_rate=("inst_escape", lambda x: np.nan), part_escape=("part_escape", "mean"),
        false_reject=("false_reject", "mean"), review=("review_rate", "mean"), fp_img=("fp_per_image", "mean"),
        lam=("lam", "mean"), **{f"size{k}_escape": (f"size{k}_escape", "mean") for k in range(len(edges) + 1)}).reset_index()
    # violation rate: fraction of repetitions with realised escape above the target
    vr = S.dropna(subset=["alpha"]).assign(v=lambda d: d.inst_escape > d.alpha).groupby(["method", "alpha"]).v.mean()
    agg["viol_rate"] = [vr.get((m, al), np.nan) for m, al in zip(agg.method, agg.alpha)]
    if "feasible" in S:
        fe = S.dropna(subset=["alpha"]).groupby(["method", "alpha"]).feasible.mean()
        agg["feasible"] = [fe.get((m, al), np.nan) for m, al in zip(agg.method, agg.alpha)]
    agg.to_csv(f"{OUT}/{a.run}{tag}_summary.csv", index=False)
    json.dump({"run": a.run, "pool": a.pool, "n_images": n, "n_defective": int(tab.defective.sum()),
               "n_gt": int(tab.n_gt.sum()), "size_edges_px2": edges, "lam_f1_val": lam_f1, "reps": a.reps,
               "delta": a.delta, "order": order, "class_escape_crc_class": {str(k): v for k, v in ce_mean.items()},
               "class_names": [D["names"][i] for i in range(nc)]}, open(f"{OUT}/{a.run}{tag}_meta.json", "w"), indent=1)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
