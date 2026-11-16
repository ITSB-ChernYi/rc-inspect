"""Property tests for the risk layer: monotonicity of the detection criterion, a counterexample for
greedy one-to-one matching, CRC validity on synthetic data."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import risk  # noqa: E402


def brute_found(g, pb, ps, lam, tau=0.5, kappa=0.5):
    sel = ps >= lam
    q = pb[sel]
    if len(q) == 0:
        return False
    if (risk.iou_matrix(g[None], q)[0] >= tau).any():
        return True
    A = (q[:, 2] - q[:, 0]) * (q[:, 3] - q[:, 1])
    ins = risk.inter_matrix(g[None], q)[0] / A >= 0.5
    return ins.sum() >= 2 and risk.union_coverage(g, q[ins]) >= kappa


def test_threshold_matches_bruteforce_and_is_monotone():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n, m = rng.integers(1, 4), rng.integers(0, 12)
        g = np.sort(rng.uniform(0, 100, (n, 2, 2)), axis=1).reshape(n, 4)[:, [0, 2, 1, 3]]
        g[:, 2:] += 5
        pb = np.sort(rng.uniform(0, 100, (m, 2, 2)), axis=1).reshape(m, 4)[:, [0, 2, 1, 3]]
        pb[:, 2:] += 2
        ps = rng.uniform(0, 1, m)
        lstar = risk.detection_thresholds(g, np.zeros(n), pb, ps, np.zeros(m))
        for k in range(n):
            prev = True
            for lam in np.linspace(0.001, 1, 60):
                f = brute_found(g[k], pb, ps, lam)
                assert f == (lam <= lstar[k] and lstar[k] > 0), (k, lam, lstar[k])
                assert prev or not f  # once missed (increasing lam), never found again
                prev = f


def test_greedy_matching_is_monotone_and_dominated_by_criterion():
    """COCO-style greedy matching processes boxes in score order, so its recall cannot drop when the
    threshold is lowered (checked on random cases); criterion (a) finds every GT that greedy matching finds."""
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n, m = rng.integers(1, 4), rng.integers(1, 6)
        g = np.column_stack([rng.uniform(0, 50, n), rng.uniform(0, 50, n)])
        g = np.column_stack([g, g + rng.uniform(5, 30, (n, 2))])
        pb = np.column_stack([rng.uniform(0, 50, m), rng.uniform(0, 50, m)])
        pb = np.column_stack([pb, pb + rng.uniform(5, 30, (m, 2))])
        ps = rng.uniform(0, 1, m)
        lams = np.linspace(0.99, 0.01, 30)
        c = risk.greedy_recall_curve(g, pb, ps, lams)
        assert (np.diff(c) >= 0).all()
        lstar = risk.detection_thresholds(g, np.zeros(n), pb, ps, np.zeros(m), fragment=False)
        assert all((lstar >= lam).sum() >= cc for lam, cc in zip(lams, c))


def test_crc_controls_mean_risk():
    rng = np.random.default_rng(1)
    lams = np.linspace(0.01, 0.99, 99)
    viol = []
    for _ in range(400):
        t = rng.beta(2, 5, 400)                       # per-image detection thresholds lam*
        L = (t[:, None] < lams[None, :]).astype(float)  # miss indicator, monotone in lam
        cal, test = L[:200], L[200:]
        lam = risk.crc_threshold(cal, lams, 0.1)
        viol.append(test[:, np.argmin(np.abs(lams - lam))].mean())
    assert np.mean(viol) <= 0.1 + 0.01
