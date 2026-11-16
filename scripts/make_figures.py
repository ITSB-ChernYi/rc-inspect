"""Result figures, generated from saved result files only."""
import glob
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import risk  # noqa: E402
from sdd.paths import DATA  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
R, FIG, TAB = os.path.join(ROOT, "results", "risk"), os.path.join(ROOT, "results", "figures"), os.path.join(ROOT, "results", "tables")
PREDS = f"{DATA}/preds"
INK, MUTED, GRID = "#1f2a37", "#6b7280", "#e5e7eb"
# categorical palette (colour-blind safe, Okabe-Ito based)
C = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442", "#000000"]
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
NAMES = {"default_0.25": "Default 0.25", "f1_val": "F1-optimal (val)", "empirical": "Empirical quantile",
         "crc": "Conformal (CRC)", "crc_size": "CRC, size strata", "crc_class": "CRC, per class", "ltt": "LTT, no review",
         "ltt_review_union@fp0.5": "LTT + review (union), FA ≤ 0.5", "ltt_review_resid@fp0.5": "LTT + review (residual), FA ≤ 0.5",
         "ltt_review_maxconf@fp0.5": "LTT + review (max score), FA ≤ 0.5", "ltt_review_random@fp0.5": "LTT + random review, FA ≤ 0.5"}


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_escape_curves(run="neu_s0_yolo11n_256_seed0"):
    """Escape rate vs threshold per class (test pool), default threshold and CRC thresholds marked."""
    D = pickle.load(open(f"{PREDS}/{run}/test.pkl", "rb"))
    rec = risk.summarise(D["pred"], D["gt"])
    ls = np.concatenate([r["lstar"] for r in rec])
    cl = np.concatenate([r["gt_cls"] for r in rec])
    lam = np.linspace(0.001, 0.9, 400)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for c in range(len(D["names"])):
        e = [(ls[cl == c] < l).mean() for l in lam]
        ax.plot(lam, e, color=C[c % len(C)], lw=1.4, label=D["names"][c].replace("_", " "))
    ax.plot(lam, [(ls < l).mean() for l in lam], color=INK, lw=2.0, ls="--", label="all defects")
    ax.axvline(0.25, color=MUTED, lw=0.8, ls=":")
    ax.text(0.255, 0.95, "default 0.25", color=MUTED, fontsize=7.5, va="top")
    for a_, y in ((0.05, 0.05), (0.10, 0.10)):
        ax.axhline(y, color=GRID, lw=0.8)
    ax.set_xlabel("confidence threshold λ")
    ax.set_ylabel("share of defects that escape")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig2_escape_vs_threshold")


def fig_realised(run="neu_s0_yolo11n_256_seed0", alpha=0.10):
    """Distribution of realised escape over 500 calibration/evaluation partitions."""
    d = pd.read_csv(f"{R}/{run}__test.csv")
    ms = ["default_0.25", "f1_val", "empirical", "crc", "crc_size", "ltt", "ltt_review_union@fp0.5"]
    data, labels = [], []
    for m in ms:
        x = d[(d.method == m) & ((d.alpha == alpha) | d.alpha.isna())].inst_escape.dropna()
        if len(x):
            data.append(x.values)
            labels.append(NAMES.get(m, m))
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    bp = ax.boxplot(data, orientation="horizontal", widths=0.6, showfliers=False, patch_artist=True)
    for b in bp["boxes"]:
        b.set(facecolor="#dbe7f7", edgecolor=C[0])
    for k in ("whiskers", "caps", "medians"):
        for l in bp[k]:
            l.set(color=INK, lw=0.9)
    ax.axvline(alpha, color=C[5], lw=1.2, ls="--")
    ax.text(alpha, len(data) + 0.6, f"target α = {alpha:.2f}", color=C[5], fontsize=7.5, ha="center")
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.set_xlabel("realised defect escape on evaluation halves")
    save(fig, "fig4_realised_escape")


def fig_frontier(run="neu_s0_yolo11n_256_seed0", alpha=0.10):
    """Certified operating points: review rate vs false alarms per image (calibration halves, first 50 partitions)."""
    f = pd.read_csv(f"{R}/{run}__test_frontier.csv")
    f = f[f.alpha == alpha]
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    budgets = np.linspace(0, 3, 61)
    for k, (kind, lab) in enumerate((("union", "union (proposed)"), ("resid", "residual evidence"),
                                     ("maxconf", "low max score"), ("random", "random"))):
        g = f[f.kind == kind]
        curves = []
        for rep, G in g.groupby("rep"):
            curves.append([G[G.fp_cal <= b].review_cal.min() if (G.fp_cal <= b).any() else np.nan for b in budgets])
        if curves:
            m = np.nanmean(np.array(curves, float), 0)
            ax.plot(budgets, m, color=C[k], lw=1.6, label=lab)
    ax.set_xlabel("false alarms per auto-processed image (budget)")
    ax.set_ylabel("review rate needed")
    ax.set_ylim(0, 1)
    ax.set_title(f"certified escape ≤ {alpha:.2f} (δ = 0.1)", fontsize=8.5, color=INK)
    ax.legend(fontsize=7.5)
    save(fig, "fig3_review_frontier")


if __name__ == "__main__":
    for f in (fig_escape_curves, fig_realised, fig_frontier):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
