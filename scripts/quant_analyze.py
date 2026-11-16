"""Does a guaranteed operating point survive a change of numerical precision?

usage: python scripts/quant_analyze.py <run> [<run> ...] [--reps 500] [--alphas 0.1] [--delta 0.1] [--fp-budget 0.5]

For every pair (reference precision -> deployed precision) and every random calibration/evaluation
partition of the test pool (the same partitions as scripts/analyze_risk.py):
  ref     calibrate on the reference predictions, evaluate on the reference predictions
  transfer   calibrate on the reference predictions, evaluate on the DEPLOYED predictions of the other half
  recal   calibrate on the deployed predictions, evaluate on the deployed predictions
Methods: conformal risk control (no review) and Learn-then-Test over the union of the two review rules
under the false-alarm budget (the proposed module). Escape control only.

Reads  $OBJ1DATA/preds_quant/<run>/<mode>_test.pkl (scripts/quant_predict.py)
Writes results/quant/<run>.csv (one row per repetition), results/quant/summary.csv, results/quant/models.csv
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import evalmap, risk  # noqa: E402
from sdd.paths import DATA  # noqa: E402

PREDS = f"{DATA}/preds_quant"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "quant")
LAMS = np.round(np.concatenate([[0.001, 0.002, 0.003, 0.004], np.arange(0.005, 0.951, 0.005)]), 3)
UNION = [("resid", r) for r in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)] + \
        [("maxconf", r) for r in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]
PAIRS = [("pt32", "pt16"), ("pt32", "onnx32"), ("onnx32", "int8")]
INT8_MODES = ("int8", "int8e", "int8p", "int8c", "int8h")
REVIEW_ALL = {"inst_escape": 0.0, "review_rate": 1.0, "fp_per_image": 0.0}


def choose(tab, cal, alpha, delta, budget):
    """-> {'crc': lam or None, 'ltt': (rho, lam) or None} chosen on the calibration images of one table."""
    d = tab.defective[cal]
    lam = risk.crc_threshold(tab.losses(cal)["inst"][d], LAMS, alpha, return_none=True)
    _, valid = risk.ltt_select(tab, cal, {"inst": alpha}, UNION, delta=delta, kind="resid", order="asc")
    cand = [v for v in valid if v[3] <= budget]
    best = min(cand, key=lambda t: (t[2], t[3])) if cand else None
    return {"crc": lam, "ltt": (best[0], best[1]) if best else None}


def apply(tab, ev, method, par):
    if par is None:
        return dict(REVIEW_ALL, feasible=False)
    e = risk.evaluate(tab, ev, par) if method == "crc" else risk.evaluate(tab, ev, par[1], par[0])
    return {"inst_escape": e["inst_escape"], "review_rate": e["review_rate"], "fp_per_image": e["fp_per_image"],
            "feasible": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.10])
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--fp-budget", type=float, default=0.5)
    ap.add_argument("--int8", default="int8", choices=INT8_MODES)   # which INT8 calibration to analyse
    ap.add_argument("--tag", default="")                            # suffix of the output files
    a = ap.parse_args()
    pairs = [(r, a.int8 if d == "int8" else d) for r, d in PAIRS]
    os.makedirs(OUT, exist_ok=True)
    models, summ = [], []
    for run in a.runs:
        D = {m: pickle.load(open(f"{PREDS}/{run}/{m}_test.pkl", "rb")) for m in ("pt32", "pt16", "onnx32", a.int8)}
        nc = len(D["pt32"]["names"])
        rec = {m: risk.summarise(d["pred"], d["gt"]) for m, d in D.items()}
        tab = {m: risk.Table(r, LAMS) for m, r in rec.items()}
        n = len(rec["pt32"])
        allidx = np.arange(n)
        for m in D:
            mp = evalmap.summary(evalmap.prepare(D[m]["pred"], D[m]["gt"], nc))
            ref = "onnx32" if m in INT8_MODES else "pt32"
            mx = np.array([r["scores"].max(initial=0.0) for r in rec[m]])
            mr = np.array([r["scores"].max(initial=0.0) for r in rec[ref]])
            ls = np.concatenate([r["lstar"] for r in rec[m]])
            lr = np.concatenate([r["lstar"] for r in rec[ref]])
            models.append({"run": run, "mode": m, "map50": mp["map50"], "map50_95": mp["map50_95"],
                           "escape_default": risk.evaluate(tab[m], allidx, 0.25)["inst_escape"],
                           "fp_default": risk.evaluate(tab[m], allidx, 0.25)["fp_per_image"],
                           "ref": ref, "maxscore_mean_abs_diff": float(np.abs(mx - mr).mean()),
                           "maxscore_mean_diff": float((mx - mr).mean()),
                           "lstar_mean_abs_diff": float(np.abs(ls - lr).mean()),
                           "lstar_mean_diff": float((ls - lr).mean()),
                           "defects_lost": int(((lr > 0) & (ls == 0)).sum()), "n_defects": int(len(lr))})
        rng = np.random.default_rng(2026)
        rows = []
        for rep in range(a.reps):
            perm = rng.permutation(n)
            cal, ev = perm[:int(round(0.5 * n))], perm[int(round(0.5 * n)):]
            for alpha in a.alphas:
                par = {m: choose(tab[m], cal, alpha, a.delta, a.fp_budget) for m in D}
                for ref, dep in pairs:
                    for method in ("crc", "ltt"):
                        for scen, pm, tm in (("ref", ref, ref), ("transfer", ref, dep), ("recal", dep, dep)):
                            rows.append({"rep": rep, "alpha": alpha, "ref": ref, "deployed": dep, "method": method,
                                         "scenario": scen, **apply(tab[tm], ev, method, par[pm][method])})
            if rep % 100 == 0:
                print(run, rep, flush=True)
        R = pd.DataFrame(rows)
        R.to_csv(f"{OUT}/{run}{a.tag}.csv", index=False)
        g = R.assign(viol=lambda d: d.inst_escape > d.alpha).groupby(["alpha", "ref", "deployed", "method", "scenario"])
        s = g.agg(escape=("inst_escape", "mean"), escape_q90=("inst_escape", lambda x: x.quantile(0.9)),
                  viol_rate=("viol", "mean"), review=("review_rate", "mean"), fp_img=("fp_per_image", "mean"),
                  feasible=("feasible", "mean")).reset_index()
        s.insert(0, "run", run)
        summ.append(s)
    M, S = pd.DataFrame(models), pd.concat(summ)
    for f, df in ((f"models{a.tag}.csv", M), (f"summary{a.tag}.csv", S)):
        p = f"{OUT}/{f}"
        if os.path.exists(p):                      # keep rows of runs analysed earlier
            old = pd.read_csv(p)
            df = pd.concat([old[~old.run.isin(a.runs)], df])
        df.to_csv(p, index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(M.round(4).to_string(index=False))
        print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
