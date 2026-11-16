"""Graphical abstract: 13 x 5 cm (journal preferred size), pipeline plus the headline numbers of the abstract.

All numbers are copied from the abstract (no new results). Writes results/figures/graphical_abstract.{png,pdf};
the PNG is 300 dpi, i.e. 1535 x 591 px (journal minimum 1328 x 531 px).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from fig_architecture import AMBER, BLUE, BLUE_L, GREEN, RED, arrow
from make_figures import INK, MUTED, save

CM = 1 / 2.54
FS = 6.8


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=FS, lw=0.8, weight="normal", ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fc, ec=ec, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.15
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, color=INK, linespacing=1.25, fontweight=weight)


def main():
    W, H = 13.0, 5.0
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # pipeline
    yc, h = 3.7, 1.05
    box(ax, 0.15, yc - h / 2, 1.55, h, "Camera\nimage")
    box(ax, 2.05, yc - h / 2, 1.85, h, "Acquisition-\nquality gate")
    box(ax, 4.25, yc - h / 2, 1.85, h, "Lightweight\ndetector\n(unchanged)")
    box(ax, 6.45, yc - h / 2, 2.55, h, "RC-Inspect\ndecision module\n(plug-in, +0.1 ms)", fc=BLUE_L, ec=BLUE, lw=1.1)
    for x0, x1 in ((1.7, 2.05), (3.9, 4.25), (6.1, 6.45)):
        arrow(ax, x0, yc, x1, yc)
    outs = [(4.55, "Accept", GREEN), (3.7, "Reject", RED), (2.85, "Review by operator", AMBER)]
    for y, t, c in outs:
        box(ax, 9.75, y - 0.3, 3.1, 0.6, t, ec=c, lw=1.0)
        arrow(ax, 9.0, yc, 9.75, y)

    # results: default threshold vs RC-Inspect (numbers as in the abstract)
    y0, h2 = 0.15, 2.3
    box(ax, 0.15, y0, 5.0, h2, "", ec=MUTED, lw=0.8)
    ax.text(0.35, y0 + h2 - 0.22, "Default threshold 0.25", fontsize=7.0, fontweight="bold", color=INK, va="top")
    ax.text(0.35, y0 + h2 - 0.75, "Defects that escape:\n11–42 % on steel and ferrite\n3–5 % on PCBs\nno guarantee",
            fontsize=FS, color=INK, va="top", linespacing=1.3)
    box(ax, 5.45, y0, 7.4, h2, "", fc=BLUE_L, ec=BLUE, lw=1.1)
    ax.text(5.65, y0 + h2 - 0.22, "RC-Inspect, 10 % escape target (error-free review)", fontsize=7.0,
            fontweight="bold", color=INK, va="top")
    ax.text(5.65, y0 + h2 - 0.75,
            "Escape 2.2–4.7 %, above the target in ≤ 0.5 % of draws\n"
            "Images reviewed: 50–69 % steel, 1–17 % PCB\n"
            "68 detectors on steel, ferrite and PCB datasets\n"
            "INT8 models: recalibrate on the quantised model",
            fontsize=FS, color=INK, va="top", linespacing=1.3)
    save(fig, "graphical_abstract")


if __name__ == "__main__":
    main()
