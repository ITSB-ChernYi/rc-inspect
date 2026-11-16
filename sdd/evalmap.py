"""AP from cached predictions, with image weights (for the paired image bootstrap).
Default integration reproduces the Ultralytics validator (used for all reported mAP values);
the pycocotools-style step integration is available as mode='coco'.

Matching per image, class and IoU threshold: predictions in descending score order are matched greedily to
the unmatched ground truth of the same class with the highest IoU >= t (COCO). AP uses 101-point
interpolation of the monotone precision envelope (COCO / Ultralytics). Image weights w_i (bootstrap counts)
enter as multiplicities of that image's detections and ground truths, so a resample needs no re-matching.
"""
import numpy as np

from .risk import iou_matrix

IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def prepare(pred, gt, nc, ious=IOUS):
    """-> dict with, per class: score (sorted desc), image index, tp flags (len(ious), n_det); n_gt per image."""
    stems = sorted(gt)
    per = {c: {"s": [], "img": [], "tp": []} for c in range(nc)}
    ngt = np.zeros((len(stems), nc))
    for i, s in enumerate(stems):
        g = gt[s]
        p = pred.get(s, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0)})
        for c in range(nc):
            gb = g["boxes"][g["cls"] == c]
            ngt[i, c] = len(gb)
            sel = p["cls"] == c
            pb, ps = p["boxes"][sel], p["scores"][sel]
            if not len(ps):
                continue
            o = np.argsort(-ps)
            pb, ps = pb[o], ps[o]
            I = iou_matrix(pb, gb) if len(gb) else np.zeros((len(pb), 0))
            tp = np.zeros((len(ious), len(ps)), bool)
            for ti, t in enumerate(ious):
                used = np.zeros(len(gb), bool)
                for j in range(len(ps)):
                    if not len(gb):
                        break
                    cand = np.where(used, -1.0, I[j])
                    k = int(np.argmax(cand))
                    if cand[k] >= t:
                        used[k] = True
                        tp[ti, j] = True
            per[c]["s"].append(ps)
            per[c]["img"].append(np.full(len(ps), i))
            per[c]["tp"].append(tp)
    out = {"ngt": ngt, "n_img": len(stems), "cls": {}}
    for c in range(nc):
        if per[c]["s"]:
            s = np.concatenate(per[c]["s"])
            o = np.argsort(-s, kind="stable")
            out["cls"][c] = {"s": s[o], "img": np.concatenate(per[c]["img"])[o], "tp": np.concatenate(per[c]["tp"], 1)[:, o]}
        else:
            out["cls"][c] = None
    return out


def ap_101(tp_w, fp_w, n_pos, mode="ultra"):
    """mode 'coco': 101-point step sampling of the precision envelope (pycocotools);
    mode 'ultra': Ultralytics compute_ap (linear interpolation of the envelope at 101 recall points, trapezoid rule)."""
    if n_pos <= 0:
        return np.nan
    ctp, cfp = np.cumsum(tp_w), np.cumsum(fp_w)
    rec = ctp / n_pos
    prec = ctp / np.maximum(ctp + cfp, 1e-12)
    if mode == "ultra":
        mrec = np.concatenate(([0.0], rec, [rec[-1] if len(rec) else 1.0], [1.0]))
        mpre = np.concatenate(([1.0], prec, [0.0], [0.0]))
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
        x = np.linspace(0, 1, 101)
        return float(np.trapezoid(np.interp(x, mrec, mpre), x))
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    q = np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0)
    return float(q.mean())


def ap_table(P, w=None):
    """AP per class (rows) and IoU threshold (cols) for image weights w (default all ones)."""
    nc = P["ngt"].shape[1]
    w = np.ones(P["n_img"]) if w is None else np.asarray(w, float)
    A = np.full((nc, len(IOUS)), np.nan)
    for c in range(nc):
        n_pos = float((P["ngt"][:, c] * w).sum())
        d = P["cls"][c]
        if d is None:
            A[c] = 0.0 if n_pos > 0 else np.nan
            continue
        wd = w[d["img"]]
        for ti in range(len(IOUS)):
            t = d["tp"][ti]
            A[c, ti] = ap_101(wd * t, wd * (~t), n_pos)
    return A


def summary(P, w=None):
    A = ap_table(P, w)
    return {"map50": float(np.nanmean(A[:, 0])), "map50_95": float(np.nanmean(A)),
            "ap50_per_class": A[:, 0].tolist(), "ap50_95_per_class": np.nanmean(A, 1).tolist()}
