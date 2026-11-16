"""Synthetic acquisition corruptions for robustness / shift tests (test-time only, never in training).

Each corruption has three severities. Images are grey-level steel/tile images stored as 3-channel JPEG;
the corruption is applied to the grey image and replicated to three channels (BGR order, as expected by
Ultralytics for numpy inputs). Randomness is seeded per image.
  noise     additive Gaussian noise, sigma = 8, 16, 32 grey levels
  blur      Gaussian (defocus-like) blur, sigma = 1, 2, 3 px at native resolution
  motion    horizontal motion blur, kernel length 5, 9, 15 px
  contrast  contrast reduced around the mean to 0.6, 0.4, 0.25 of the original
  dark      under-exposure, gamma 1.5, 2.0, 3.0 (applied to intensities scaled to [0, 1])
  bright    over-exposure, gain 1.4, 1.8, 2.4 with clipping (saturation)
  jpeg      re-compression at quality 30, 15, 7
"""
import io

import numpy as np
from PIL import Image, ImageFilter

PARAMS = {"noise": [8, 16, 32], "blur": [1, 2, 3], "motion": [5, 9, 15], "contrast": [0.6, 0.4, 0.25],
          "dark": [1.5, 2.0, 3.0], "bright": [1.4, 1.8, 2.4], "jpeg": [30, 15, 7]}


def corrupt_gray(g, kind, sev, seed=0):
    p = PARAMS[kind][sev - 1]
    x = g.astype(np.float32)
    if kind == "noise":
        x = x + np.random.default_rng(seed).normal(0, p, x.shape)
    elif kind == "blur":
        x = np.asarray(Image.fromarray(g).filter(ImageFilter.GaussianBlur(p)), np.float32)
    elif kind == "motion":
        k = np.ones(p) / p
        x = np.apply_along_axis(lambda r: np.convolve(np.pad(r, p // 2, mode="edge"), k, mode="valid"), 1, x)
    elif kind == "contrast":
        x = x.mean() + p * (x - x.mean())
    elif kind == "dark":
        x = 255.0 * (x / 255.0) ** p
    elif kind == "bright":
        x = x * p
    elif kind == "jpeg":
        buf = io.BytesIO()
        Image.fromarray(g).save(buf, format="JPEG", quality=p)
        x = np.asarray(Image.open(buf).convert("L"), np.float32)
    return np.clip(x, 0, 255).astype(np.uint8)


def apply(path, kind, sev, seed=0):
    g = np.asarray(Image.open(path).convert("L"))
    c = corrupt_gray(g, kind, sev, seed)
    return np.repeat(c[:, :, None], 3, axis=2)
