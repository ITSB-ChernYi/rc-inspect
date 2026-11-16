"""Fig. 1: overall architecture of the proposed framework (schematic, no data). Wording follows manuscript Section 3.1.

Writes results/figures/fig_architecture.{png,pdf}. The PNG (300 dpi, 7.2 in wide = 2160 px) also serves as the
graphical abstract (journal minimum 1328 px wide).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from make_figures import INK, MUTED, save

BLUE, BLUE_L = "#0072B2", "#e4eff8"
GREEN, RED, AMBER = "#009E73", "#D55E00", "#B26A00"
GREY_L = "#f3f4f6"
FS = 7


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=FS, lw=0.9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, c=INK, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.9, shrinkA=0,
                                                                  shrinkB=0, linestyle=ls, mutation_scale=8))


def main():
    W, H = 10.0, 5.5
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * H / W))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # ---------------- (a) online inspection ----------------
    ax.text(0.1, 5.45, "(a) Online inspection of each image", fontsize=7.5, fontweight="bold", color=INK, va="top")
    yc, h = 3.85, 0.9
    ax.add_patch(FancyBboxPatch((1.45, 2.95), 5.95, 1.9, boxstyle="round,pad=0,rounding_size=0.08", fc=GREY_L,
                                ec=MUTED, lw=0.8, ls="--"))
    ax.text(1.55, 4.78, "Inspection computer", fontsize=6.5, color=MUTED, va="top", style="italic")
    box(ax, 0.1, yc - h / 2, 1.05, h, "Industrial\ncamera")
    box(ax, 1.55, yc - h / 2, 1.6, h, "Acquisition-\nquality gate\n(training ranges)")
    box(ax, 3.6, yc - h / 2, 1.4, h, "Lightweight\ndetector\n(unchanged)")
    box(ax, 5.5, yc - h / 2, 1.75, h, "Risk-controlled\ndecision module\n(λ, ρ)", fc=BLUE_L, ec=BLUE, lw=1.2)
    arrow(ax, 1.15, yc, 1.55, yc)
    arrow(ax, 3.15, yc, 3.6, yc)
    ax.text(3.375, yc + 0.06, "in\nrange", fontsize=5.8, color=INK, ha="center", va="bottom", linespacing=1.1)
    arrow(ax, 5.0, yc, 5.5, yc)
    ax.text(5.25, yc + 0.06, "boxes,\nscores", fontsize=5.8, color=INK, ha="center", va="bottom", linespacing=1.1)
    outs = [(4.5, "Accept", GREEN), (3.85, "Reject, boxes reported", RED), (3.2, "Review by operator", AMBER)]
    for y, t, c in outs:
        box(ax, 7.85, y - 0.22, 2.12, 0.44, t, ec=c, lw=1.1)
        arrow(ax, 7.25, yc, 7.85, y)
    # gate: image outside the training ranges -> review
    ax.plot([2.35, 2.35, 7.6, 7.6], [yc - h / 2, 2.72, 2.72, 3.2], color=AMBER, lw=0.9)
    arrow(ax, 7.6, 3.2, 7.85, 3.2, c=AMBER)
    ax.text(3.9, 2.69, "out of range", fontsize=5.8, color=AMBER, ha="center", va="top")

    # ---------------- (b) offline calibration ----------------
    ax.text(0.1, 2.3, "(b) Offline calibration, once per detector", fontsize=7.5, fontweight="bold", color=INK, va="top")
    y0, h2 = 0.45, 1.05
    yc2 = y0 + h2 / 2
    box(ax, 0.1, y0, 1.55, h2, "Labelled\ncalibration\nimages")
    box(ax, 2.05, y0, 1.6, h2, "Per-defect\ndetection\nthresholds")
    box(ax, 4.05, y0, 1.6, h2, "Risk curves\nover (λ, ρ)")
    box(ax, 6.05, y0, 2.25, h2, "Select (λ, ρ):\nP(escape ≤ α) ≥ 1 − δ,\nfalse alarms ≤ budget,\nleast review", fc=BLUE_L,
        ec=BLUE, lw=1.2)
    for x0, x1 in ((1.65, 2.05), (3.65, 4.05), (5.65, 6.05)):
        arrow(ax, x0, yc2, x1, yc2)
    # parameters are stored with the deployed model and used by the decision module
    ax.plot([8.3, 8.75, 8.75, 6.375], [yc2, yc2, 2.5, 2.5], color=BLUE, lw=0.9, ls="--")
    arrow(ax, 6.375, 2.5, 6.375, yc - h / 2, c=BLUE, ls="--")
    ax.text(8.85, 1.6, "stored with the\ndeployed model", fontsize=6, color=BLUE, va="center")
    save(fig, "fig_architecture")


if __name__ == "__main__":
    main()
