"""Vector redraws of Figs. 3, 4 and 8 of the CAEE manuscript from stored results (the original plotting scripts are
not on this PC). Content is unchanged: same runs, methods, partitions and averaging as the published raster files.

Fig. 3  escape vs confidence threshold per class   preds/gpu_neu_s0_yolo11n_256_seed0/test.pkl (criterion of Eq. 1)
Fig. 4  realised escape over partitions             results/risk/gpu_{neu,gc10g}_s0_yolo11n_*__test.csv, alpha 0.10
Fig. 8  acquisition shift on NEU-DET                results/risk/gpu_neu_s0_*_256_seed0__shift_summary.csv,
                                                     nine detectors: YOLOv8n, YOLOv12n and seven YOLOv12n variants
Writes results/figures/fig3_escape_threshold.{png,pdf}, fig4_realised_escape_v2.{png,pdf}, fig8_shift.{png,pdf}.
"""
import glob
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from make_figures import C, GRID, INK, MUTED, PREDS, R, save

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import risk  # noqa: E402

ORANGE, BLUE, BLUE_L, ORANGE_L = "#D55E00", "#0072B2", "#dbe7f7", "#fbe3d0"


def fig3(run="gpu_neu_s0_yolo11n_256_seed0"):
    D = pickle.load(open(f"{PREDS}/{run}/test.pkl", "rb"))
    rec = risk.summarise(D["pred"], D["gt"])
    ls = np.concatenate([r["lstar"] for r in rec])
    cl = np.concatenate([r["gt_cls"] for r in rec])
    lam = np.linspace(0.0, 0.9, 451)
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    names = D["names"]
    names = dict(names) if isinstance(names, dict) else dict(enumerate(names))
    for c, name in sorted(names.items()):
        e = [100 * (ls[cl == c] < l).mean() for l in lam]
        ax.plot(lam, e, color=C[c % len(C)], lw=1.4, label=name.replace("_", " "))
    ax.plot(lam, [100 * (ls < l).mean() for l in lam], color=INK, lw=2.0, ls=(0, (3, 1.5)), label="all defects")
    ax.axvline(0.25, color=MUTED, lw=0.9, ls=":")
    ax.text(0.258, 97, "default 0.25", color=MUTED, fontsize=7.5, va="top")
    ax.axhline(10, color=ORANGE, lw=0.9, ls="--")
    ax.text(0.885, 11.5, "10 % target", color=ORANGE, fontsize=7.5, ha="right", va="bottom")
    ax.set(xlabel="confidence threshold λ", ylabel="defects that escape (%)", xlim=(0, 0.9), ylim=(0, 100))
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "fig3_escape_threshold")
    return {n: 100 * (ls[cl == c] < 0.25).mean() for c, n in sorted(names.items())}


def fig4(alpha=0.10):
    panels = [("(a) NEU-DET, YOLO11n, 3 seeds", [f"gpu_neu_s0_yolo11n_256_seed{s}" for s in range(3)]),
              ("(b) GC10-DET grouped, YOLO11n, 2 seeds", [f"gpu_gc10g_s0_yolo11n_640_seed{s}" for s in range(2)])]
    ms = [("ltt_review_union@fp0.5", "LTT + union review (proposed)"), ("ltt", "Learn-then-Test, no review"),
          ("crc_size", "CRC, size strata"), ("crc", "Conformal risk control"), ("empirical", "Empirical quantile"),
          ("f1_val", "F1-optimal (validation)"), ("default_0.25", "Default threshold 0.25")]
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    out = {}
    for ax, (title, runs) in zip(axs, panels):
        d = pd.concat([pd.read_csv(f"{R}/{r}__test.csv") for r in runs])
        d = d[np.isclose(d.alpha, alpha) | d.alpha.isna()]
        data = [100 * d[d.method == m].inst_escape.dropna().values for m, _ in ms][::-1]
        bp = ax.boxplot(data, orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True, whis=1.5)
        for k, b in enumerate(bp["boxes"]):
            prop = k == len(data) - 1
            b.set(facecolor=ORANGE_L if prop else BLUE_L, edgecolor=ORANGE if prop else BLUE, lw=1.1)
        for key in ("whiskers", "caps", "medians"):
            for l in bp[key]:
                l.set(color=INK, lw=0.9)
        ax.axvline(100 * alpha, color=ORANGE, lw=1.2, ls="--")
        ax.text(100 * alpha + 1, len(data) + 0.45, f"target {100 * alpha:.0f} %", color=ORANGE, fontsize=7.5)
        ax.set_xlim(0, 50)
        ax.set_ylim(0.4, len(data) + 0.8)
        ax.set_title(title, fontsize=8.5, color=INK)
        out[title] = {lab: float(np.median(x)) for (m, lab), x in zip(ms, data[::-1])}
    axs[0].set_yticks(range(1, len(ms) + 1), [lab for _, lab in ms][::-1])
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    x0, x1 = axs[0].get_position().x0, axs[1].get_position().x1
    fig.text((x0 + x1) / 2, 0.015, "realised defect escape on the evaluation half (%), over calibration/evaluation "
             "partitions", ha="center", va="bottom", fontsize=8.5, color=INK)
    save(fig, "fig4_realised_escape_v2")
    return out


def fig8(dets=("yolov8n", "yolo12n", "v12nbb", "v12nbbsofl", "secfull", "secnosofl", "hcsa", "bifpn", "ewbifpn")):
    fs = [f"{R}/gpu_neu_s0_{d}_256_seed0__shift_summary.csv" for d in dets]
    D = pd.concat([pd.read_csv(f) for f in fs])
    rows = ["clean"] + [f"{k}{i}" for k in ("noise", "blur", "motion", "contrast", "dark", "bright", "jpeg")
                        for i in (1, 2, 3)]
    cols = ["default_0.25", "crc", "ltt_review_union@fp0.5", "ltt_review_union@fp0.5+gate"]
    labels = ["Default\n0.25", "CRC", "Proposed", "Proposed\n+ gate"]
    P = D.groupby(["shift", "method"])[["inst_escape", "review"]].mean()
    E = np.array([[100 * P.loc[(r, c), "inst_escape"] for c in cols] for r in rows])
    V = np.array([[100 * P.loc[(r, c), "review"] for c in cols] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 6.0))
    for ax, M, cmap, vmax, title in ((axs[0], E, "Reds", 60, "(a) realised escape (%)"),
                                     (axs[1], V, "Blues", 100, "(b) review rate (%)")):
        ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 * vmax else INK)
        for y in (0.5, 3.5, 6.5, 9.5, 12.5, 15.5, 18.5):   # separators between disturbance types
            ax.axhline(y, color="white", lw=2.5)
        ax.set_xticks(range(len(cols)), labels, fontsize=8)
        ax.set_yticks(range(len(rows)), [r[:-1] + " " + r[-1] if r != "clean" else r for r in rows])
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
    axs[1].set_yticklabels([])
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig8_shift")
    return E, V


if __name__ == "__main__":
    print("fig3 escape at 0.25 per class:", {k: round(v, 1) for k, v in fig3().items()})
    for k, v in fig4().items():
        print("fig4", k, {a: round(b, 1) for a, b in v.items()})
    E, V = fig8()
    print("fig8 clean escape", E[0].round(0), "review", V[0].round(0))
