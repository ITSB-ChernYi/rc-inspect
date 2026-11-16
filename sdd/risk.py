"""Risk-controlled inspection on top of a trained defect detector.

Notation (manuscript Section 3):
  image x_i with ground-truth defects G_i (boxes, classes) and detector output
  P_i = {(b_j, s_j, c_j)} with scores s_j in [0, 1];  P_i(lam) = {p in P_i : s >= lam}.

Detection criterion (monotone, fragmentation tolerant).  A ground-truth box g is *found* by a set of
predicted boxes Q if either
  (a) some q in Q has IoU(q, g) >= tau, or
  (b) the boxes q in Q that lie mostly inside g (|q ∩ g| / |q| >= ioa) jointly cover at least
      kappa of the area of g.
Both conditions can only become true when boxes are added to Q, so "found" is monotone in Q and hence
in the score threshold. The *detection threshold* of g is lam*(g) = max{lam : g found by P_i(lam)}
(0 if never found); g is missed at threshold lam iff lam > lam*(g). All risks below are therefore
computed for every threshold from one pass over the predictions.

Decisions per image at parameters theta = (lam, rho):
  review   r_i = 1[u_i(lam) >= rho], with u_i(lam) = max{s : s < lam} (strongest dismissed candidate,
           "residual evidence"; 0 if none) -- the image goes to a human inspector;
  otherwise automatic: reject if P_i(lam) is non-empty, accept if it is empty.

Risks (all are means of per-image losses bounded in [0, 1]):
  R_inst  = E_defective[(1 - r_i) * |{g missed}| / |G_i|]       defect (instance) escape rate
  R_part  = E_defective[(1 - r_i) * 1[P_i(lam) empty]]           defective part accepted automatically
  R_fa    = E_free[(1 - r_i) * 1[P_i(lam) non-empty]]            defect-free part rejected automatically
  review  = E[r_i]
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats

# --------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------


def iou_matrix(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def inter_matrix(a, b):
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def union_coverage(g, qs, res=32):
    """Fraction of box g covered by the union of boxes qs (rasterised on a res x res grid inside g)."""
    if len(qs) == 0:
        return 0.0
    xs = g[0] + (np.arange(res) + 0.5) * (g[2] - g[0]) / res
    ys = g[1] + (np.arange(res) + 0.5) * (g[3] - g[1]) / res
    X, Y = np.meshgrid(xs, ys)
    cov = np.zeros_like(X, dtype=bool)
    for q in qs:
        cov |= (X >= q[0]) & (X <= q[2]) & (Y >= q[1]) & (Y <= q[3])
    return float(cov.mean())


# --------------------------------------------------------------------------------------------
# per-ground-truth detection thresholds
# --------------------------------------------------------------------------------------------


def detection_thresholds(gt_boxes, gt_cls, pb, ps, pc, class_aware=False, tau=0.5, ioa=0.5, kappa=0.5,
                         fragment=True):
    """lam*(g) for every ground-truth box (monotone criterion above); 0.0 if never found.

    fragment=False keeps only condition (a) (any-overlap IoU criterion, still monotone).
    """
    n = len(gt_boxes)
    out = np.zeros(n)
    if n == 0 or len(pb) == 0:
        return out
    order = np.argsort(-ps)
    pb, ps, pc = pb[order], ps[order], pc[order]
    I = iou_matrix(gt_boxes, pb)
    A = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
    inside = inter_matrix(gt_boxes, pb) / np.maximum(A[None, :], 1e-9) >= ioa
    for k in range(n):
        ok = np.ones(len(pb), bool) if not class_aware else (pc == gt_cls[k])
        inside_seen = []
        for j in np.nonzero(ok)[0]:          # predictions in descending score order
            if I[k, j] >= tau:               # condition (a)
                out[k] = ps[j]
                break
            if fragment and inside[k, j]:    # condition (b), re-evaluated only when a candidate is added
                inside_seen.append(j)
                if len(inside_seen) >= 2 and union_coverage(gt_boxes[k], pb[inside_seen]) >= kappa:
                    out[k] = ps[j]
                    break
    return out


def greedy_recall_curve(gt_boxes, pb, ps, lams, tau=0.5):
    """Number of GT matched under COCO-style one-to-one greedy matching at each threshold (class agnostic).
    Greedy matching handles boxes in score order, so this count is monotone in the threshold; it differs
    from the criterion above in that fragments of one defect are not credited."""
    out = []
    order = np.argsort(-ps)
    pb, ps = pb[order], ps[order]
    I = iou_matrix(pb, gt_boxes) if len(pb) and len(gt_boxes) else np.zeros((len(pb), len(gt_boxes)))
    for lam in lams:
        used = np.zeros(len(gt_boxes), bool)
        m = 0
        for j in np.nonzero(ps >= lam)[0]:
            if len(gt_boxes) == 0:
                break
            cand = np.where(~used, I[j], -1)
            k = int(np.argmax(cand))
            if cand[k] >= tau:
                used[k] = True
                m += 1
        out.append(m)
    return np.array(out)


# --------------------------------------------------------------------------------------------
# per-image summaries from cached predictions
# --------------------------------------------------------------------------------------------


def summarise(pred, gt, class_aware=False, fragment=True, tau=0.5):
    """pred/gt: dict stem -> arrays. Returns per-image records used by all risk computations.

    pred[stem] = {"boxes": (m,4), "scores": (m,), "cls": (m,)}; gt[stem] = {"boxes": (n,4), "cls": (n,)}
    """
    rec = []
    for stem in sorted(gt):
        g, p = gt[stem], pred.get(stem, {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0)})
        lstar = detection_thresholds(g["boxes"], g["cls"], p["boxes"], p["scores"], p["cls"],
                                     class_aware=class_aware, fragment=fragment, tau=tau)
        # predictions that explain no ground truth (false alarms): IoU < tau and not mostly inside any GT
        if len(p["boxes"]) and len(g["boxes"]):
            I = iou_matrix(p["boxes"], g["boxes"]).max(1)
            A = (p["boxes"][:, 2] - p["boxes"][:, 0]) * (p["boxes"][:, 3] - p["boxes"][:, 1])
            ins = (inter_matrix(p["boxes"], g["boxes"]) / np.maximum(A[:, None], 1e-9)).max(1)
            fp = (I < tau) & (ins < 0.5)
        else:
            fp = np.ones(len(p["boxes"]), bool)
        area = (g["boxes"][:, 2] - g["boxes"][:, 0]) * (g["boxes"][:, 3] - g["boxes"][:, 1]) if len(g["boxes"]) else np.zeros(0)
        rec.append({"stem": stem, "n_gt": len(g["boxes"]), "lstar": lstar, "gt_cls": np.asarray(g["cls"]),
                    "gt_area": area, "scores": np.sort(np.asarray(p["scores"]))[::-1],
                    "fp_scores": np.sort(np.asarray(p["scores"])[fp])[::-1],
                    "fp_cls": np.asarray(p["cls"])[fp][np.argsort(-np.asarray(p["scores"])[fp])]})
    return rec


class Table:
    """Vectorised per-image losses on a threshold grid (rows: images, cols: lam grid)."""

    def __init__(self, rec, lams, size_edges=None):
        self.lams = np.asarray(lams)
        L = len(self.lams)
        n = len(rec)
        self.n_gt = np.array([r["n_gt"] for r in rec])
        self.defective = self.n_gt > 0
        self.missed = np.zeros((n, L))
        self.empty = np.zeros((n, L), bool)
        self.fp = np.zeros((n, L))
        self.u = np.zeros((n, L))           # residual evidence: max score strictly below lam
        self.maxs = np.array([r["scores"].max(initial=0.0) for r in rec])
        self.rand = np.random.default_rng(12345).uniform(size=n)
        for i, r in enumerate(rec):
            s = r["scores"]
            if r["n_gt"]:
                self.missed[i] = (r["lstar"][None, :] < self.lams[:, None]).sum(1)
            self.empty[i] = ~(s.max(initial=-1) >= self.lams) if len(s) else True
            self.fp[i] = (r["fp_scores"][None, :] >= self.lams[:, None]).sum(1) if len(r["fp_scores"]) else 0
            if len(s):
                below = np.where(s[None, :] < self.lams[:, None], s[None, :], 0.0)
                self.u[i] = below.max(1)
        self.miss_frac = np.where(self.n_gt[:, None] > 0, self.missed / np.maximum(self.n_gt[:, None], 1), 0.0)
        # size strata (by GT box area, in pixels^2 of the original image)
        self.strata = None
        if size_edges is not None:
            K = len(size_edges) + 1
            self.strata = []
            for k in range(K):
                lo = -np.inf if k == 0 else size_edges[k - 1]
                hi = np.inf if k == K - 1 else size_edges[k]
                mk, nk = np.zeros((n, L)), np.zeros(n)
                for i, r in enumerate(rec):
                    sel = (r["gt_area"] >= lo) & (r["gt_area"] < hi)
                    nk[i] = sel.sum()
                    if nk[i]:
                        mk[i] = (r["lstar"][sel][None, :] < self.lams[:, None]).sum(1)
                self.strata.append((mk, nk))

    def review(self, idx, rho, kind="resid"):
        """Review indicator (n_idx, L).
        resid  : review if the strongest dismissed candidate u(lam) >= rho * lam (rho relative, in (0, 1))
        maxconf: review if the image's maximum score is below rho (absolute) -- 'uncertain image'
        random : review a random fraction rho of images (fixed per-image uniform draw)
        None rho means no review."""
        L = len(self.lams)
        if rho is None:
            return np.zeros((len(idx), L), bool)
        if kind == "resid":
            return (self.u[idx] >= rho * self.lams[None, :]) & (self.u[idx] > 0)
        if kind == "maxconf":
            return np.repeat((self.maxs[idx] < rho)[:, None], L, axis=1)
        if kind == "random":
            return np.repeat((self.rand[idx] < rho)[:, None], L, axis=1)
        raise ValueError(kind)

    def losses(self, idx, rho=None, kind="resid"):
        """Per-image losses for images idx at every lam."""
        r = self.review(idx, rho, kind)
        auto = ~r
        return {"inst": auto * self.miss_frac[idx], "part": auto * (self.empty[idx] & self.defective[idx, None]),
                "fa": auto * (~self.empty[idx] & ~self.defective[idx, None]), "review": r.astype(float),
                "fp": auto * self.fp[idx], "missed": auto * self.missed[idx], "n_gt": self.n_gt[idx]}


# --------------------------------------------------------------------------------------------
# calibration procedures
# --------------------------------------------------------------------------------------------


class ClassTable:
    """Class-wise quantities for class-specific thresholds lam_c (class-aware criterion: a defect of class c is
    found only by predictions of class c, so the per-class risks decouple and each lam_c is calibrated alone)."""

    def __init__(self, rec, lams, nc):
        self.lams, self.nc = np.asarray(lams), nc
        n, L = len(rec), len(lams)
        self.missed = np.zeros((nc, n, L))
        self.n = np.zeros((nc, n))
        self.fp = np.zeros((nc, n, L))
        for i, r in enumerate(rec):
            for c in range(nc):
                sel = r["gt_cls"] == c
                self.n[c, i] = sel.sum()
                if sel.any():
                    self.missed[c, i] = (r["lstar"][sel][None, :] < self.lams[:, None]).sum(1)
                fs = r["fp_scores"][r["fp_cls"] == c]
                if len(fs):
                    self.fp[c, i] = (fs[None, :] >= self.lams[:, None]).sum(1)

    def crc_thresholds(self, idx, alpha):
        out = []
        for c in range(self.nc):
            sel = self.n[c, idx] > 0
            if not sel.any():
                out.append(float(self.lams[0]))
                continue
            loss = self.missed[c, idx][sel] / self.n[c, idx][sel][:, None]
            t = crc_threshold(loss, self.lams, alpha, return_none=True)
            out.append(None if t is None else float(t))
        return out

    def evaluate(self, idx, lam_c):
        j = [int(np.argmin(np.abs(self.lams - l))) for l in lam_c]
        per_class = []
        num = den = 0.0
        img_loss = np.zeros(len(idx))
        img_n = np.zeros(len(idx))
        fp = np.zeros(len(idx))
        for c in range(self.nc):
            m = self.missed[c, idx, j[c]]
            nn = self.n[c, idx]
            per_class.append(float(m.sum() / max(nn.sum(), 1)) if nn.sum() else np.nan)
            sel = nn > 0
            per_class_img = (m[sel] / nn[sel]).mean() if sel.any() else np.nan
            per_class[-1] = (per_class[-1], float(per_class_img))
            img_loss += m
            img_n += nn
            fp += self.fp[c, idx, j[c]]
        d = img_n > 0
        return {"inst_escape": float((img_loss[d] / img_n[d]).mean()), "inst_escape_pooled": float(img_loss.sum() / img_n.sum()),
                "fp_per_image": float(fp.mean()), "class_escape_pooled": [p[0] for p in per_class],
                "class_escape_img": [p[1] for p in per_class], "lam_c": list(map(float, lam_c))}


def crc_threshold(loss_cal, lams, alpha, B=1.0, return_none=False):
    """Conformal risk control (Angelopoulos et al., 2024): largest lam with (n*Rhat(lam) + B)/(n+1) <= alpha.
    loss_cal (n, L) must be non-decreasing in lam (columns sorted ascending in lam). If no lam on the grid
    satisfies the bound the target cannot be certified: returns None when return_none, else the smallest lam."""
    n = loss_cal.shape[0]
    ok = (loss_cal.sum(0) + B) / (n + 1) <= alpha
    j = np.nonzero(ok)[0]
    j = j[j == np.arange(len(j))]
    if not len(j):
        return None if return_none else lams[0]
    return lams[j[-1]]


def hb_pvalue(rhat, n, alpha):
    """Hoeffding-Bentkus p-value for H0: R > alpha, given empirical mean rhat of n losses in [0, 1]
    (Bates et al., 2021; Angelopoulos et al., 2025)."""
    if n == 0:
        return 1.0
    rhat = min(rhat, alpha)
    h = rhat * math.log(rhat / alpha) + (1 - rhat) * math.log((1 - rhat) / (1 - alpha)) if 0 < rhat < alpha else (
        math.log(1 / (1 - alpha)) if rhat == 0 else 0.0)
    p_hoeff = math.exp(-n * h)
    p_bentkus = math.e * stats.binom.cdf(math.ceil(n * rhat), n, alpha)
    return min(p_hoeff, p_bentkus, 1.0)


def ltt_select(tab, idx, risks, rho_grid, delta=0.1, kind="resid", order="asc"):
    """Learn-then-Test over (lam, rho) (Angelopoulos et al., 2025).

    For each rho in rho_grid, the hypotheses H_lam: max_k R_k(lam, rho) > alpha_k are tested by fixed-sequence
    testing over the lam grid in a pre-specified order ('asc': from the smallest lam, conservative for escape
    risks; 'desc': from the largest lam, conservative for false-reject risks); Bonferroni over the rho levels
    gives family-wise error <= delta. Each risk uses Hoeffding-Bentkus p-values on its own population
    (defective images for 'inst'/'part', defect-free images for 'fa').
    Returns (best, valid): best = valid configuration with minimal (review rate, false alarms per image)."""
    valid = []
    js = range(len(tab.lams)) if order == "asc" else range(len(tab.lams) - 1, -1, -1)
    for rho in rho_grid:
        # grid entries may be (kind, rho) pairs: Learn-then-Test over a union of review rules
        k_, r_ = (rho if isinstance(rho, tuple) else (kind, rho))
        Lo = tab.losses(idx, r_, k_)
        for j in js:
            ps = []
            for name, alpha in risks.items():
                pop = tab.defective[idx] if name in ("inst", "part") else ~tab.defective[idx]
                x = Lo[name][pop, j]
                ps.append(hb_pvalue(float(x.mean()) if len(x) else 0.0, int(len(x)), alpha))
            if max(ps) > delta / len(rho_grid):
                break
            valid.append((rho, float(tab.lams[j]), float(Lo["review"][:, j].mean()), float(Lo["fp"][:, j].mean())))
    best = min(valid, key=lambda t: (t[2], t[3])) if valid else None
    return best, valid


def evaluate(tab, idx, lam, rho=None, kind="resid"):
    if isinstance(rho, tuple):
        kind, rho = rho
    j = int(np.argmin(np.abs(tab.lams - lam)))
    Lo = tab.losses(idx, rho, kind)
    d, f = tab.defective[idx], ~tab.defective[idx]
    return {"inst_escape": float(Lo["inst"][d, j].mean()) if d.any() else np.nan,
            "inst_escape_pooled": float(Lo["missed"][d, j].sum() / max(Lo["n_gt"][d].sum(), 1)) if d.any() else np.nan,
            "part_escape": float(Lo["part"][d, j].mean()) if d.any() else np.nan,
            "false_reject": float(Lo["fa"][f, j].mean()) if f.any() else np.nan,
            "review_rate": float(Lo["review"][:, j].mean()), "fp_per_image": float(Lo["fp"][:, j].mean()),
            "lam": float(tab.lams[j]), "rho": rho,
            "size_escape": [float(((1 - Lo["review"][:, j]) * mk[idx, j]).sum() / max(nk[idx].sum(), 1))
                            for mk, nk in (tab.strata or [])]}
