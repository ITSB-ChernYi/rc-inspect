"""Operating curves of the decision module: review rate and realised escape against the escape target.

Reads results/risk/_curve_all.csv (built from the *_curve_summary.csv files of scripts/analyze_risk.py run with
--alphas 0.05 0.10 0.15 0.20 --fp-budgets 0.5 --tag _curve). Writes results/figures/fig_operating_curves.{png,pdf}.
"""
import os

import matplotlib.pyplot as plt
import pandas as pd

from make_figures import C, INK, MUTED, save  # style and palette of the other result figures

R = os.path.join(os.path.dirname(__file__), "..", "results", "risk")
DS = [("NEU", "NEU-DET (3 detectors × 3 seeds)", C[0], "o"), ("GC10", "GC10-DET (YOLO11n, 2 seeds)", C[1], "s"),
      ("PCB", "DeepPCB (3 detectors × 3 seeds)", C[2], "^")]


def main():
    A = pd.read_csv(f"{R}/_curve_all.csv")
    A = A[A.method == "ltt_review_union"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for ds, label, col, mk in DS:
        d = A[A.ds == ds]
        g = d.groupby("alpha")
        m, lo, hi, feas = g.rev.mean(), g.rev.min(), g.rev.max(), g.feas.mean()
        x = m.index.values * 100
        ax[0].fill_between(x, lo.values, hi.values, color=col, alpha=0.15, linewidth=0)
        ax[0].plot(x, m.values, color=col, linewidth=1.6, label=label)
        for xi, yi, fi in zip(x, m.values, feas.values):   # hollow marker: operating point not always feasible
            ax[0].plot(xi, yi, mk, color=col, markersize=5.5, markerfacecolor=col if fi > 99.9 else "white")
        e = g.esc.mean()
        ax[1].plot(x, e.values, color=col, linewidth=1.6, marker=mk, markersize=5.5)
    ax[0].set(xlabel="Escape target α (%)", ylabel="Images sent to review (%)", xticks=[5, 10, 15, 20], ylim=(-3, 103))
    ax[0].legend(loc="upper right", fontsize=7.5)
    ax[0].set_title("(a)", loc="left", fontsize=9, color=INK)
    ax[1].plot([5, 20], [5, 20], color=MUTED, linewidth=1, linestyle="--")
    ax[1].annotate("target", (16.2, 17.4), color=MUTED, fontsize=7.5, rotation=31)
    ax[1].set(xlabel="Escape target α (%)", ylabel="Realised escape (%)", xticks=[5, 10, 15, 20], ylim=(-0.6, 21))
    ax[1].set_title("(b)", loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig_operating_curves")


if __name__ == "__main__":
    main()
