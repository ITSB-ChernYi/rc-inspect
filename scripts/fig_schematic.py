"""Fig. 1: risk-controlled inspection, calibration and deployment (schematic, no data)."""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
INK, MUTED, ACC, ACC2, OK, BAD, REV = "#1f2a37", "#6b7280", "#2b59c3", "#e3ecfb", "#2e7d4f", "#b3261e", "#b26a00"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.0))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, wrap=True,
            fontweight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, c=INK):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.0, shrinkA=0, shrinkB=0))


def main():
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.6)
    ax.axis("off")
    ax.text(0.1, 4.4, "(a) Calibration (once, on held-out images)", fontsize=10, fontweight="bold", color=INK, va="top")
    box(ax, 0.1, 2.9, 1.55, 1.0, "Efficient detector\n(trained, frozen)\nscore floor 0.001")
    box(ax, 2.0, 2.9, 1.9, 1.0, r"Per-defect detection" "\n" r"threshold $\lambda^*(g)$, Eq. (2)" "\n" "fragment-tolerant, monotone", fc=ACC2, ec=ACC)
    box(ax, 4.25, 2.9, 2.05, 1.0, "Exact risk curves\nescape, false reject,\nreview, false alarms vs (λ, ρ)")
    box(ax, 6.65, 2.9, 1.75, 1.0, "Conformal risk control\n/ Learn-then-Test\n(fixed sequence, HB)", fc=ACC2, ec=ACC)
    box(ax, 8.75, 2.9, 1.65, 1.0, r"Certified operating" "\n" r"point $\hat\theta=(\hat\lambda,\hat\rho)$" "\n" "min review s.t. budgets")
    for x0, x1 in ((1.65, 2.0), (3.9, 4.25), (6.3, 6.65), (8.4, 8.75)):
        arrow(ax, x0, 3.4, x1, 3.4)
    ax.text(5.2, 2.6, "targets: escape ≤ α, false reject ≤ β (with probability ≥ 1 − δ); false alarms per image ≤ budget",
            ha="center", fontsize=8, color=MUTED)
    ax.text(0.1, 2.15, "(b) Deployment (per part, microseconds after the detector)", fontsize=10, fontweight="bold", color=INK, va="top")
    box(ax, 0.1, 0.45, 1.55, 1.1, r"Image of part" "\n" "→ detector" "\n" r"scores $s_j$")
    box(ax, 2.0, 0.45, 2.3, 1.1, "residual evidence\n" + r"$u=\max\{s_j: s_j<\hat\lambda\}$" + "\n(or low max score)")
    arrow(ax, 1.65, 1.0, 2.0, 1.0)
    box(ax, 4.65, 1.25, 1.6, 0.6, r"$u\geq\hat\rho\hat\lambda$ → REVIEW", fc="#fff4e5", ec=REV, fs=8.5, bold=True)
    box(ax, 4.65, 0.55, 1.6, 0.6, r"else: boxes $\geq\hat\lambda$ ?", fs=8.5)
    arrow(ax, 4.3, 1.1, 4.65, 1.5)
    arrow(ax, 4.3, 0.95, 4.65, 0.85)
    box(ax, 6.6, 0.85, 1.6, 0.5, "yes → REJECT\n(report boxes)", fc="#fdecea", ec=BAD, fs=8.5, bold=True)
    box(ax, 6.6, 0.2, 1.6, 0.5, "no → ACCEPT", fc="#e8f5ec", ec=OK, fs=8.5, bold=True)
    arrow(ax, 6.25, 0.85, 6.6, 1.1)
    arrow(ax, 6.25, 0.8, 6.6, 0.45)
    ax.text(8.45, 1.2, "Guaranteed on future parts:\n• defect escape ≤ α\n• good parts rejected ≤ β\nMinimised: review workload",
            fontsize=8.5, color=INK, va="center")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig1_overview.{ext}"), dpi=300)
    print("fig1 written")


if __name__ == "__main__":
    main()
