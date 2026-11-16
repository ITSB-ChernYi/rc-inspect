"""CPU time of the acquisition-quality gate at full image resolution and at the detector input resolution.

Same protocol as the gate column of scripts/bench_latency.py: grey image already decoded, median over the first
100 test images (after 5 warm-up images), one process. Writes results/latency_gate.json.
usage: python scripts/bench_gate.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import quality  # noqa: E402
from sdd.paths import DATA  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "latency_gate.json")
SETS = [("NEU-DET", "neu_s0", 256), ("Magnetic Tile", "mt_s0", 320), ("GC10-DET", "gc10_grouped_s0", 640),
        ("DeepPCB", "deeppcb_grouped_s0", 640)]


def main():
    torch.set_num_threads(4)
    res = {}
    for name, ds, size in SETS:
        d = f"{DATA}/{ds}/images/test"
        imgs = [np.asarray(Image.open(f"{d}/{f}").convert("L")) for f in sorted(os.listdir(d))[:105]]
        row = {"input_size": size, "n_images": 100}
        for key, sz in (("full_ms", None), ("input_ms", size)):
            t = []
            for k, g in enumerate(imgs):
                t0 = time.perf_counter()
                quality.features(g, sz)
                if k >= 5:
                    t.append((time.perf_counter() - t0) * 1e3)
            row[key] = float(np.median(t))
        res[name] = row
        print(name, {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    json.dump(res, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
