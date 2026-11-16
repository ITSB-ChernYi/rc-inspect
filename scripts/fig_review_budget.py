"""Fig. 5: review rate needed for a guaranteed escape <= alpha against the false-alarm budget b.

Reads the frontier files of scripts/analyze_risk.py (valid LTT configurations of the first 50 partitions, with their
calibration review rate and calibration false alarms per image, reviewed images counting zero; the same quantity
that the budget b bounds). For each budget the lowest review rate among valid configurations within the budget is
taken per partition and averaged over partitions. Writes results/figures/fig_review_budget.{png,pdf}.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import C, INK, save  # style and palette of the other result figures

R = os.path.join(os.path.dirname(__file__), "..", "results", "risk")
PANELS = [("gpu_neu_s0_yolo11n_256_seed0", "(a) NEU-DET, YOLO11n"),
          ("gpu_gc10g_s0_yolo11n_640_seed0", "(b) GC10-DET grouped, YOLO11n")]
KINDS = [("union", "union (proposed)"), ("resid", "residual evidence"), ("maxconf", "low max score"), ("random", "random")]
COL = {"union": C[0], "resid": C[1], "maxconf": C[2], "random": C[3]}


def curve(g, budgets):
    out = []
    for _, G in g.groupby("rep"):
        fp, rv = G.fp_cal.values, G.review_cal.values
        out.append([rv[fp <= b].min() if (fp <= b).any() else np.nan for b in budgets])
    a = np.array(out, float)
    ok = np.isfinite(a).mean(0) >= 0.5          # draw only where the rule is feasible in most partitions
    m = np.nanmean(np.where(np.isfinite(a), a, np.nan), 0)
    return np.where(ok, m, np.nan)


def main(alpha=0.10):
    budgets = np.linspace(0, 2, 81)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    for k, (run, title) in enumerate(PANELS):
        f = pd.read_csv(f"{R}/{run}__test_frontier.csv")
        f = f[np.isclose(f.alpha, alpha)]
        for kind, lab in KINDS:
            g = f[f.kind == kind]
            if len(g):
                ax[k].plot(budgets, 100 * curve(g, budgets), color=COL[kind], lw=2.2 if kind == "union" else 1.4, label=lab,
                           zorder=3 if kind == "union" else 2)
        ax[k].set_title(title, loc="left", fontsize=9, color=INK)
        ax[k].set_xlabel("false-alarm budget b (false alarms per image)")
        ax[k].set_xlim(-0.05, 2.05)
    ax[0].set_ylabel("Images sent to review (%)")
    ax[0].set_ylim(0, 100)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig_review_budget")


if __name__ == "__main__":
    main()
