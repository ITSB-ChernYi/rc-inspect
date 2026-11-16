"""Guarantees under acquisition shift: calibrate on clean calibration halves, evaluate on corrupted evaluation halves.

usage: python scripts/analyze_shift.py <run> [--reps 200]
Requires test.pkl and test_<kind><sev>.pkl from scripts/predict.py --corrupt kind:sev.
For every partition the calibration half is clean; the evaluation half is the same images after corruption.
Output: results/risk/<run>__shift.csv (+ _summary.csv); with --gate-size N: <run>__shift_gateN.csv
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import corrupt, quality, risk  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

PREDS = f"{DATA}/preds"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "risk")
GATES = f"{DATA}/gates"
LAMS = np.round(np.concatenate([[0.001, 0.002, 0.003, 0.004], np.arange(0.005, 0.951, 0.005)]), 3)
UNION = [("resid", r) for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] + [("maxconf", r) for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]


def evaluate_gated(tab, idx, lam, rho, g):
    """risk.evaluate with an extra review indicator g (per image in idx)."""
    kind, r_ = rho if isinstance(rho, tuple) else ("resid", rho)
    j = int(np.argmin(np.abs(tab.lams - lam)))
    rev = tab.review(idx, r_, kind)[:, j] | g
    auto = ~rev
    d = tab.defective[idx]
    return {"inst_escape": float((auto * tab.miss_frac[idx, j])[d].mean()),
            "review_rate": float(rev.mean()), "fp_per_image": float((auto * tab.fp[idx, j]).mean()), "lam": float(lam)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--fp", type=float, default=0.5)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--gate-size", type=int, default=0)   # >0: gate statistics at this input size (sdd/quality.py)
    a = ap.parse_args()
    clean = pickle.load(open(f"{PREDS}/{a.run}/test.pkl", "rb"))
    tab = risk.Table(risk.summarise(clean["pred"], clean["gt"]), LAMS)
    shifted = {}
    for f in sorted(glob.glob(f"{PREDS}/{a.run}/test_*.pkl")):
        tag = os.path.basename(f)[5:-4]
        S = pickle.load(open(f, "rb"))
        shifted[tag] = risk.Table(risk.summarise(S["pred"], S["gt"]), LAMS)
    shifted["clean"] = tab
    n = len(tab.n_gt)
    # quality gate: support from training images, gate decisions for clean and corrupted test images.
    # Gates depend only on the images (not on the detector), so they are cached per dataset and per condition;
    # every run on the same split reuses them, and an interrupted computation resumes where it stopped.
    import json
    import yaml
    from PIL import Image
    rec = json.load(open(f"{DATA}/runs/{a.run}/run_record.json"))
    root = dataset_root(rec)
    gs = a.gate_size or None
    cache = f"{GATES}/{os.path.basename(root.rstrip('/'))}" + (f"_in{gs}" if gs else "")
    os.makedirs(cache, exist_ok=True)
    if os.path.exists(f"{cache}/support.npy"):
        support = tuple(np.load(f"{cache}/support.npy"))
    else:
        tr = sorted(os.listdir(f"{root}/images/train"))
        support = quality.fit((np.asarray(Image.open(f"{root}/images/train/{f}").convert("L")) for f in tr), gs)
        np.save(f"{cache}/support.npy", np.stack(support))
    files = sorted(os.listdir(f"{root}/images/test"))
    stems = sorted(clean["gt"])
    assert [f[:-4] for f in files] == stems
    gates = {}
    for tag in shifted:
        c = f"{cache}/{tag}.npy"
        if not os.path.exists(c):
            if tag == "clean":
                g = [quality.gate(np.asarray(Image.open(f"{root}/images/test/{f}").convert("L")), support, gs) for f in files]
            else:
                kind, sev = tag[:-1], int(tag[-1])
                g = [quality.gate(corrupt.apply(f"{root}/images/test/{f}", kind, sev, seed=k)[:, :, 0], support, gs)
                     for k, f in enumerate(files)]
            np.save(c, np.array(g))
        gates[tag] = np.load(c)
    rng = np.random.default_rng(2026)
    rows = []
    for rep in range(a.reps):
        p = rng.permutation(n)
        cal, ev = p[: n // 2], p[n // 2:]
        Lc = tab.losses(cal)
        d = tab.defective[cal]
        lam_crc = risk.crc_threshold(Lc["inst"][d], LAMS, a.alpha, return_none=True)
        _, valid = risk.ltt_select(tab, cal, {"inst": a.alpha}, UNION, delta=a.delta)
        cand = [v for v in valid if v[3] <= a.fp]
        best = min(cand, key=lambda t: (t[2], t[3])) if cand else None
        for tag, T in shifted.items():
            rows.append({"rep": rep, "shift": tag, "method": "default_0.25", **risk.evaluate(T, ev, 0.25)})
            if lam_crc is not None:
                rows.append({"rep": rep, "shift": tag, "method": "crc", **risk.evaluate(T, ev, lam_crc)})
            if best is not None:
                rows.append({"rep": rep, "shift": tag, "method": f"ltt_review_union@fp{a.fp}",
                             **risk.evaluate(T, ev, best[1], best[0])})
                # same operating point plus the quality gate (gated images are reviewed)
                rows.append({"rep": rep, "shift": tag, "method": f"ltt_review_union@fp{a.fp}+gate",
                             **evaluate_gated(T, ev, best[1], best[0], gates[tag][ev])})
            rows.append({"rep": rep, "shift": tag, "method": "gate_rate", "review_rate": float(gates[tag][ev].mean())})
    R = pd.DataFrame(rows).drop(columns=["size_escape"], errors="ignore")
    suf = f"_gate{gs}" if gs else ""
    R.to_csv(f"{OUT}/{a.run}__shift{suf}.csv", index=False)
    agg = R.groupby(["shift", "method"]).agg(inst_escape=("inst_escape", "mean"),
                                             viol_rate=("inst_escape", lambda x: (x > a.alpha).mean()),
                                             review=("review_rate", "mean"), fp_img=("fp_per_image", "mean")).reset_index()
    agg.to_csv(f"{OUT}/{a.run}__shift{suf}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(agg.round(4).to_string())


if __name__ == "__main__":
    main()
