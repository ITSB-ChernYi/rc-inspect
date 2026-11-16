"""Example decisions of the proposed module on each dataset (manuscript Section 4.9).

One detector per dataset (YOLO11n, seed 0), the first calibration/evaluation partition of
scripts/analyze_risk.py (rng 2026), escape target 0.10, delta 0.1, false-alarm budget 0.5 per image,
union of review rules. Evaluation images are sorted into five categories and the FIRST image of each
category in file-name order is shown, so the examples are not hand-picked.
The quality gate uses the support and clean-image decisions cached by scripts/analyze_shift.py (computed
here, in the same way, if the cache does not exist).
Writes results/figures/fig_examples.{png,pdf} and results/figures/fig_examples.json (chosen images, counts).
"""
import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import quality, risk  # noqa: E402
from sdd.paths import DATA, dataset_root  # noqa: E402

from make_figures import FIG, INK, MUTED, save  # noqa: E402

PREDS, GATES = f"{DATA}/preds", f"{DATA}/gates"
LAMS = np.round(np.concatenate([[0.001, 0.002, 0.003, 0.004], np.arange(0.005, 0.951, 0.005)]), 3)
UNION = [("resid", r) for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]] + \
        [("maxconf", r) for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
RUNS = [("NEU-DET", "gpu_neu_s0_yolo11n_256_seed0"), ("GC10-DET", "gpu_gc10g_s0_yolo11n_640_seed0"),
        ("DeepPCB", "gpu_pcbg_s0_yolo11n_640_seed0")]
COLS = ["Rejected,\nall defects found", "Review:\nselected review rule", "Review:\nquality gate",
        "Missed defect\n(accepted or rejected)"]
GT, KEPT, DISM, MISS = "#009E73", "#0072B2", "#E69F00", "#D55E00"
ALPHA, DELTA, FB = 0.10, 0.1, 0.5


def clean_gate(root, stems):
    cache = f"{GATES}/{os.path.basename(root.rstrip('/'))}"
    os.makedirs(cache, exist_ok=True)
    if os.path.exists(f"{cache}/support.npy"):
        support = tuple(np.load(f"{cache}/support.npy"))
    else:   # same support as analyze_shift.py; images streamed to keep memory low
        tr = sorted(os.listdir(f"{root}/images/train"))
        support = quality.fit(np.asarray(Image.open(f"{root}/images/train/{f}").convert("L")) for f in tr)
        np.save(f"{cache}/support.npy", np.stack(support))
    files = sorted(os.listdir(f"{root}/images/test"))
    assert [f[:-4] for f in files] == stems
    if not os.path.exists(f"{cache}/clean.npy"):
        np.save(f"{cache}/clean.npy", np.array([quality.gate(np.asarray(Image.open(f"{root}/images/test/{f}").convert("L")),
                                                             support) for f in files]))
    return np.load(f"{cache}/clean.npy"), files


def decide(run):
    D = pickle.load(open(f"{PREDS}/{run}/test.pkl", "rb"))
    rec = risk.summarise(D["pred"], D["gt"])
    tab = risk.Table(rec, LAMS)
    n = len(rec)
    perm = np.random.default_rng(2026).permutation(n)      # first partition of analyze_risk.py
    cal, ev = perm[:int(round(0.5 * n))], perm[int(round(0.5 * n)):]
    _, valid = risk.ltt_select(tab, cal, {"inst": ALPHA}, UNION, delta=DELTA, kind="resid", order="asc")
    cand = [v for v in valid if v[3] <= FB]
    (kind, rho), lam = min(cand, key=lambda t: (t[2], t[3]))[:2]
    j = int(np.argmin(np.abs(LAMS - lam)))
    root = dataset_root(json.load(open(f"{DATA}/runs/{run}/run_record.json")))
    gate, files = clean_gate(root, [r["stem"] for r in rec])
    cats = {c: [] for c in range(4)}
    for i in sorted(ev, key=lambda i: rec[i]["stem"]):
        if gate[i]:
            cats[2].append(i)
        elif tab.review(np.array([i]), rho, kind)[0, j]:
            cats[1].append(i)
        elif tab.missed[i, j] > 0:
            cats[3].append(i)
        elif tab.defective[i] and not tab.empty[i, j]:
            cats[0].append(i)
    return {"D": D, "rec": rec, "tab": tab, "lam": float(lam), "kind": kind, "rho": float(rho), "j": j,
            "root": root, "files": files, "cats": cats, "n_ev": len(ev)}


def tile(ax, S, i, col):
    r, D = S["rec"][i], S["D"]
    stem = r["stem"]
    img = Image.open(f"{S['root']}/images/test/{S['files'][i]}").convert("L")
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    p, g = D["pred"].get(stem, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0)}), D["gt"][stem]
    lam, lstar = S["lam"], r["lstar"]
    W = img.size[0]
    lw = 1.1
    for b, ls in zip(g["boxes"], lstar):
        miss = ls < lam
        ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=MISS if miss else GT,
                               lw=lw * (1.6 if miss else 1), ls="--"))
    sc = p["scores"]
    below = np.nonzero(sc < lam)[0]
    shown = list(np.nonzero(sc >= lam)[0])
    if col == 1 and S["kind"] == "resid" and len(below):     # the dismissed candidate that triggered review
        shown.append(int(below[np.argmax(sc[below])]))
    for k in shown:
        b = p["boxes"][k]
        c = KEPT if sc[k] >= lam else DISM
        ax.add_patch(Rectangle(b[:2], b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw))
        ax.text(b[0], b[1] - 0.01 * W, f"{sc[k]:.2f}", color="white", fontsize=5.2, va="bottom",
                bbox=dict(boxstyle="square,pad=0.08", fc=c, ec="none"))
    note = {1: (f"dismissed {sc[below].max():.2f} ≥ ρλ = {S['rho'] * lam:.3f}" if S["kind"] == "resid" and len(below)
                else f"max score {sc.max(initial=0):.2f} < ρ = {S['rho']:.1f}"),
            2: "image statistics outside\nthe training ranges"}.get(col)
    if note:
        ax.set_xlabel(note, fontsize=6, color=INK, labelpad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(MUTED)
        s.set_linewidth(0.5)


def main():
    out = {}
    fig, axs = plt.subplots(len(RUNS), 4, figsize=(7.0, 6.2), gridspec_kw={"wspace": 0.08, "hspace": 0.38})
    for ri, (ds, run) in enumerate(RUNS):
        S = decide(run)
        out[ds] = {"run": run, "lam": S["lam"], "rule": S["kind"], "rho": S["rho"], "n_eval": S["n_ev"],
                   "counts": {COLS[c]: len(v) for c, v in S["cats"].items()},
                   "shown": {COLS[c]: (S["rec"][v[0]]["stem"] if v else None) for c, v in S["cats"].items()}}
        print(ds, json.dumps(out[ds]), flush=True)
        for c in range(4):
            ax = axs[ri, c]
            if S["cats"][c]:
                tile(ax, S, S["cats"][c][0], c)
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "no image in this\ncategory", ha="center", va="center", fontsize=7, color=MUTED,
                        transform=ax.transAxes)
            if ri == 0:
                ax.set_title(f"({'abcd'[c]}) {COLS[c]}", fontsize=7, color=INK, pad=8)
        rule = "residual evidence" if S["kind"] == "resid" else "low maximum score"
        axs[ri, 0].set_ylabel(f"{ds}\nλ = {S['lam']:.3f}\n{rule}, ρ = {S['rho']:.1f}", fontsize=6.6, color=INK)
    hs = [Rectangle((0, 0), 1, 1, fill=False, ec=GT, ls="--"), Rectangle((0, 0), 1, 1, fill=False, ec=MISS, ls="--", lw=1.6),
          Rectangle((0, 0), 1, 1, fill=False, ec=KEPT), Rectangle((0, 0), 1, 1, fill=False, ec=DISM)]
    fig.legend(hs, ["ground truth, found", "ground truth, missed", "detection kept (score ≥ λ)",
                    "strongest dismissed candidate (score < λ)"], loc="lower center", ncol=4, fontsize=6.6,
               bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_examples")
    json.dump(out, open(os.path.join(FIG, "fig_examples.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
