"""Re-implementation of the YOLO-SEC components inside Ultralytics, for a controlled re-test.

The original YOLO-SEC training code, checkpoints and logs are not available, so the
components are rebuilt from the manuscript equations (HCSA Eq. 7-12, EW-BiFPN Eq. 13-17,
SOFL Eq. 18-23) on top of the Ultralytics YOLOv12n definition. Module code follows
objective3/edgeopt/models/yolosec.py, which was written from the same equations.

Registration: Ultralytics' parse_model resolves module names from the globals of
ultralytics.nn.tasks. HCSA is shape preserving and is registered by name. The edge-
weighted fusion node takes a list of inputs; parse_model only infers the output width
of list-input modules for its own CBFuse, so the fusion node is registered under that
name (CBFuse is not used by any model in this study). All inputs of a fusion node are
aligned to the same width beforehand, so the inferred width ch[f[-1]] is correct.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn import tasks
from ultralytics.utils import loss as uloss
from ultralytics.utils.metrics import bbox_iou

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "configs")


class SobelEdge(nn.Module):
    """Channel-averaged Sobel magnitude (Eq. 10)."""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("weight", torch.stack([kx, kx.t()]).unsqueeze(1), persistent=False)

    def forward(self, x):
        g = F.conv2d(x.mean(1, keepdim=True), self.weight.to(x.dtype), padding=1)
        return torch.sqrt(g[:, :1] ** 2 + g[:, 1:] ** 2 + 1e-12)


class HCSA(nn.Module):
    """Hierarchical channel-spatial attention (Eq. 7-12); shape preserving."""

    def __init__(self, c, reduction=8, use_sobel=True):
        super().__init__()
        h = max(1, c // reduction)
        self.fc1, self.fc2 = nn.Conv2d(c, h, 1), nn.Conv2d(h, c, 1)
        self.dw3 = nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)
        self.dw7 = nn.Conv2d(c, c, 7, padding=3, groups=c, bias=False)
        self.proj3, self.proj7 = nn.Conv2d(c, 1, 1, bias=False), nn.Conv2d(c, 1, 1, bias=False)
        self.use_sobel = use_sobel
        self.sobel = SobelEdge()
        self.eta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f_ch = x * torch.sigmoid(self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)))))
        s = self.proj3(self.dw3(x)) + self.proj7(self.dw7(x))
        if self.use_sobel:
            s = s + self.eta * self.sobel(x)
        return f_ch + x * torch.sigmoid(s)


class EWFuse(nn.Module):
    """Edge-weighted BiFPN fusion node (Eq. 14-17). edge=False gives plain weighted BiFPN (Eq. 14)."""

    def __init__(self, c, n=2, edge=True, eps=1e-4):
        super().__init__()
        self.n, self.edge, self.eps = n, edge, eps
        self.w = nn.Parameter(torch.ones(n))
        if edge:
            self.lam = nn.Parameter(torch.zeros(1))
            self.edge_dw = nn.ModuleList(nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False) for _ in range(n))
            self.edge_pw = nn.ModuleList(nn.Conv2d(c, 1, 1) for _ in range(n))
        # post-fusion 3x3 Conv-BN-SiLU; with this choice the full model has 2.33 M parameters, matching the
        # 2,323,378 reported in the manuscript (the manuscript does not name the post-fusion block)
        self.out = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.SiLU())

    def forward(self, xs):
        num, den = 0.0, self.eps
        for i, f in enumerate(xs):
            w = F.relu(self.w[i])
            if self.edge:
                w = w * (1.0 + F.softplus(self.lam) * torch.sigmoid(self.edge_pw[i](self.edge_dw[i](f))))
            num, den = num + w * f, den + w
        return self.out(num / den)


def register():
    tasks.HCSA = HCSA
    tasks.CBFuse = EWFuse  # list-input registration shim, see module docstring


# ----------------------------------------------------------------------------------------
# YAML generation: YOLOv12n backbone + optional HCSA + PAN (baseline) or (EW-)BiFPN neck
# ----------------------------------------------------------------------------------------
BACKBONE = """backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]
"""
PAN = """  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, {p4}], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, {p3}], 1, Concat, [1]]
  - [-1, 2, A2C2f, [256, False, -1]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, {n11}], 1, Concat, [1]]
  - [-1, 2, A2C2f, [512, False, -1]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, {p5}], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[{o3}, {o4}, {o5}], 1, Detect, [nc]]
"""


def bifpn(p3, p4, p5, i0, width, edge):
    """Width-uniform BiFPN (one top-down, one bottom-up pass with P4 skip), layer indices from i0."""
    e = "True" if edge else "False"
    W = width * 4  # yaml widths are scaled by 0.25 for scale n
    L = [f"  - [{p3}, 1, Conv, [{W}, 1, 1]]", f"  - [{p4}, 1, Conv, [{W}, 1, 1]]", f"  - [{p5}, 1, Conv, [{W}, 1, 1]]"]
    a3, a4, a5 = i0, i0 + 1, i0 + 2
    L += ["  - [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",               # i0+3  up(P5)
          f"  - [[-1, {a4}], 1, CBFuse, [{width}, 2, {e}]]",                  # i0+4  P4td
          "  - [-1, 1, nn.Upsample, [None, 2, \"nearest\"]]",                # i0+5
          f"  - [[-1, {a3}], 1, CBFuse, [{width}, 2, {e}]]",                  # i0+6  P3out
          f"  - [-1, 1, Conv, [{W}, 3, 2]]",                                  # i0+7  down(P3out)
          f"  - [[-1, {i0 + 4}, {a4}], 1, CBFuse, [{width}, 3, {e}]]",        # i0+8  P4out
          f"  - [-1, 1, Conv, [{W}, 3, 2]]",                                  # i0+9  down(P4out)
          f"  - [[-1, {a5}], 1, CBFuse, [{width}, 2, {e}]]",                  # i0+10 P5out
          f"  - [[{i0 + 6}, {i0 + 8}, {i0 + 10}], 1, Detect, [nc]]"]
    return "\n".join(L) + "\n"


def make_yaml(name, hcsa=False, neck="pan", width=96, sobel=True):
    """neck in {'pan', 'bifpn', 'ewbifpn'}."""
    head = "head:\n"
    if hcsa:
        head += f"  - [4, 1, HCSA, [128, 8, {sobel}]]\n  - [6, 1, HCSA, [128, 8, {sobel}]]\n  - [8, 1, HCSA, [256, 8, {sobel}]]\n"
        p3, p4, p5, i0 = 9, 10, 11, 12
    else:
        p3, p4, p5, i0 = 4, 6, 8, 9
    if neck == "pan":
        head += PAN.format(p3=p3, p4=p4, p5=p5, n11=i0 + 2, o3=i0 + 5, o4=i0 + 8, o5=i0 + 11)
    else:
        head += bifpn(p3, p4, p5, i0, width, edge=(neck == "ewbifpn"))
    txt = "nc: 6\nscales:\n  n: [0.50, 0.25, 1024]\n" + BACKBONE + head
    os.makedirs(CFG, exist_ok=True)
    path = os.path.join(CFG, f"{name}.yaml")
    open(path, "w").write(txt)
    return path


# HCSA and the fusion nodes are not width-scaled by parse_model, so their channel arguments are the
# actual YOLOv12n widths (P3 128, P4 128, P5 256; fusion width 96); Conv widths are pre-scaled (x4).


# ----------------------------------------------------------------------------------------
# SOFL (Eq. 18-23): 1 - IoU + a1*corner + a2*area, focal classification
# ----------------------------------------------------------------------------------------
class SOFLParams(nn.Module):
    """Learnable non-negative regression weights alpha1, alpha2 (softplus), registered on the model
    so that the optimizer updates them (manuscript Table III: init 0.97 / 0.69 after softplus)."""

    def __init__(self, a1=0.97, a2=0.69):
        super().__init__()
        inv = lambda v: float(torch.log(torch.expm1(torch.tensor(v))))  # noqa: E731
        self.a1 = nn.Parameter(torch.tensor([inv(a1)]))
        self.a2 = nn.Parameter(torch.tensor([inv(a2)]))


def corner_term(p, g):
    ex1, ey1 = torch.min(p[:, 0], g[:, 0]), torch.min(p[:, 1], g[:, 1])
    ex2, ey2 = torch.max(p[:, 2], g[:, 2]), torch.max(p[:, 3], g[:, 3])
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-9
    px, py = p[:, [0, 2, 0, 2]], p[:, [1, 1, 3, 3]]
    gx, gy = g[:, [0, 2, 0, 2]], g[:, [1, 1, 3, 3]]
    return ((px - gx) ** 2 + (py - gy) ** 2).sum(-1) / (4.0 * c2)


def area_term(p, g):
    ap = (p[:, 2] - p[:, 0]).clamp(min=0) * (p[:, 3] - p[:, 1]).clamp(min=0)
    ag = (g[:, 2] - g[:, 0]).clamp(min=0) * (g[:, 3] - g[:, 1]).clamp(min=0)
    return (ap - ag).abs() / torch.maximum(ap, ag).clamp(min=1e-9)


class SOFLBboxLoss(uloss.BboxLoss):
    def __init__(self, reg_max, params):
        super().__init__(reg_max)
        self.params = params

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum,
                fg_mask, imgsz, stride):
        fg = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg].sum(-1, keepdim=True)
        p, g = pred_bboxes[fg], target_bboxes[fg]
        iou = bbox_iou(p, g, xywh=False).squeeze(-1)
        reg = 1.0 - iou + F.softplus(self.params.a1) * corner_term(p, g) + F.softplus(self.params.a2) * area_term(p, g)
        loss_iou = (reg.unsqueeze(-1) * weight).sum() / target_scores_sum
        target_ltrb = uloss.bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg].view(-1, self.dfl_loss.reg_max), target_ltrb[fg]) * weight
        return loss_iou, loss_dfl.sum() / target_scores_sum


class FocalBCE(nn.Module):
    """Focal form of the classification loss (Eq. 22), elementwise, gamma 1.5, alpha 0.25."""

    def __init__(self, gamma=1.5, alpha=0.25):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = logits.sigmoid()
        pt = targets * p + (1 - targets) * (1 - p)
        at = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return bce * at * (1.0 - pt) ** self.gamma


class SOFLLoss(uloss.v8DetectionLoss):
    def __init__(self, model, focal=True, reg=True):
        super().__init__(model)
        if focal:
            self.bce = FocalBCE()
        if reg:
            self.bbox_loss = SOFLBboxLoss(self.reg_max, model.sofl).to(self.device)


class SOFLDetectionModel(tasks.DetectionModel):
    """DetectionModel carrying the learnable SOFL weights and using the SOFL criterion."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sofl = SOFLParams()
        self.sofl_cfg = (True, True)

    def init_criterion(self):
        return SOFLLoss(self, focal=self.sofl_cfg[0], reg=self.sofl_cfg[1])


def make_trainer(sofl=False, focal=True, reg=True):
    """Detection trainer whose models carry SOFLParams and use the SOFL criterion."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class SECTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            register()
            cls = SOFLDetectionModel if sofl else tasks.DetectionModel
            model = cls(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
            if sofl:
                model.sofl_cfg = (focal, reg)
            model = self.set_model_names_for_load(model)
            if weights:
                model.load(weights)
            return model

    return SECTrainer
