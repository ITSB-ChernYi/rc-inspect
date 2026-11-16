"""Label-free acquisition-quality gate.

Per image, four grey-level statistics: mean intensity, RMS contrast, sharpness (log variance of the Laplacian)
and noise level (1.4826 x median absolute deviation of the residual after a 3 x 3 median filter). The support
is the [0.5 %, 99.5 %] quantile range of each statistic over the TRAINING images; an image with any statistic
outside its range is routed to review. The gate needs no labels and no detector output.

size: if given, the grey image is first reduced so that its longer side is at most `size` pixels (bilinear with
antialiasing, aspect ratio kept), i.e. the statistics are computed at the detector input resolution. Images that
are already this small are left unchanged. The support must be fitted with the same `size` as the gate uses.
"""
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

Q = (0.005, 0.995)


def to_input(g, size=None):
    g = np.asarray(g)
    if not size or max(g.shape[:2]) <= size:
        return g
    h, w = g.shape[:2]
    s = size / max(h, w)
    im = Image.fromarray(g.astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return np.asarray(im)


def features(g, size=None):
    g = np.asarray(to_input(g, size), np.float32)
    lap = ndimage.laplace(g)
    med = np.asarray(Image.fromarray(g.clip(0, 255).astype(np.uint8)).filter(ImageFilter.MedianFilter(3)), np.float32)
    r = g - med
    noise = 1.4826 * np.median(np.abs(r - np.median(r)))
    return np.array([g.mean(), g.std(), np.log(lap.var() + 1e-6), np.log(noise + 1e-3)])


def fit(images, size=None):
    F = np.stack([features(g, size) for g in images])
    return np.quantile(F, Q[0], axis=0), np.quantile(F, Q[1], axis=0)


def gate(g, support, size=None):
    f = features(g, size)
    lo, hi = support
    return bool(((f < lo) | (f > hi)).any())
