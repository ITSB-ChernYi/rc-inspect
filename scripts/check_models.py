"""Build every architecture used in the study; report parameters and GFLOPs (Ultralytics counter)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sdd import yolosec as Y  # noqa: E402
from ultralytics.nn.tasks import DetectionModel  # noqa: E402
from ultralytics.utils.torch_utils import get_flops  # noqa: E402

Y.register()
torch.set_num_threads(1)
VARIANTS = {"v12n_pan": {}, "sec_hcsa": dict(hcsa=True), "sec_hcsa_nosobel": dict(hcsa=True, sobel=False),
            "sec_bifpn": dict(neck="bifpn"), "sec_ewbifpn": dict(neck="ewbifpn"),
            "sec_hcsa_bifpn": dict(hcsa=True, neck="bifpn"), "sec_full": dict(hcsa=True, neck="ewbifpn")}
out = {}
for name, kw in VARIANTS.items():
    m = DetectionModel(Y.make_yaml(name, **kw), nc=6, verbose=False)
    m(torch.zeros(1, 3, 256, 256))
    out[name] = {"params": sum(x.numel() for x in m.parameters()),
                 "gflops_640": round(get_flops(m, 640), 3), "gflops_256": round(get_flops(m, 256), 3)}
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "architectures.json"), "w"), indent=1)
