"""Table 12 with infeasible calibration draws counted as full review (Section 4.3).
analyze_shift.py writes a row only when CRC or RC-Inspect returns an operating point; here a draw without one counts
as review of every image (zero escape, review 1) and as no violation. Reads results/risk/*__shift{,_gate640}.csv.
usage (from objective1/): python scripts/shift_fallback.py   -> results/tables/shift_fallback.csv
"""
import glob, os, pandas as pd
U = "ltt_review_union@fp0.5"
out = []
for f in sorted(glob.glob("results/risk/gpu_*_s0_*_seed0__shift*.csv")):
    if f.endswith("_summary.csv"):
        continue
    run, suf = os.path.basename(f)[:-4].split("__shift")
    R = pd.read_csv(f); reps = sorted(R.rep.unique()); n = len(reps)
    for shift, S in R.groupby("shift"):
        rec = {"run": run, "gate": suf.lstrip("_") or "full", "shift": shift, "draws": n}
        for m, key in [("default_0.25", "default"), ("crc", "crc"), (U, "rci_nogate"), (U + "+gate", "rci")]:
            M = S[S.method == m].set_index("rep")
            rec[key + "_feasible"] = len(M) / n
            esc = M.inst_escape.reindex(reps).fillna(0.0)          # infeasible draw: full review, zero escape
            rev = M.review_rate.reindex(reps).fillna(1.0)
            rec[key + "_escape"] = esc.mean(); rec[key + "_review"] = rev.mean()
            rec[key + "_viol"] = (esc > 0.10).mean()
            rec[key + "_escape_feasible_only"] = M.inst_escape.mean()
        rec["gate_rate"] = S[S.method == "gate_rate"].review_rate.mean()
        out.append(rec)
D = pd.DataFrame(out)
os.makedirs("results/tables", exist_ok=True)
D.to_csv("results/tables/shift_fallback.csv", index=False)
print(len(D), "rows; feasibility below 1:")
print(D[(D.crc_feasible < 1) | (D.rci_feasible < 1)].groupby(["run", "gate"])[["crc_feasible", "rci_feasible"]].mean())
