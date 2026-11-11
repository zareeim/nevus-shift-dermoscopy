import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


# --------------------------------------------------------------------------
# Backbones split into stages
# --------------------------------------------------------------------------
class ToNCHW(nn.Module):
    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


def build_backbone(name: str, pretrained: bool = True):
    """Return (stem, [stage modules], [stage output channels])."""
    name = name.lower()
    if name.startswith("resnet"):
        weights = "DEFAULT" if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool)
        stages = [m.layer1, m.layer2, m.layer3, m.layer4]
    elif name.startswith("convnext"):
        weights = "DEFAULT" if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        f = m.features  # [stem, s1, ds, s2, ds, s3, ds, s4]
        stem = f[0]
        stages = [f[1], nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]), nn.Sequential(f[6], f[7])]
    elif name.startswith("efficientnet"):
        weights = "DEFAULT" if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        f = m.features  # 0 stem, 1..7 MBConv, 8 head conv
        stem = nn.Sequential(f[0], f[1], f[2])
        stages = [f[3], f[4], nn.Sequential(f[5], f[6]), nn.Sequential(f[7], f[8])]
    elif name.startswith("swin"):
        # vision transformer with shifted windows (Liu et al., ICCV 2021); torchvision works in
        # channels-last (B,H,W,C) inside; the last stage ends with the model's own LayerNorm and a
        # permute to (B,C,H,W) so the rest of the network is unchanged. Contains no BatchNorm.
        weights = "DEFAULT" if pretrained else None
        m = getattr(tvm, name)(weights=weights)
        f = m.features  # 0 patch embed, 1 s1, 2 merge, 3 s2, 4 merge, 5 s3, 6 merge, 7 s4
        stem = f[0]
        stages = [f[1], nn.Sequential(f[2], f[3]), nn.Sequential(f[4], f[5]),
                  nn.Sequential(f[6], f[7], m.norm, ToNCHW())]
        c = m.head.in_features
        return stem, nn.ModuleList(stages), [c // 8, c // 4, c // 2, c]
    else:
        raise ValueError(f"Unsupported backbone: {name}")

    # infer stage channels with a dummy pass in eval mode (does not touch BN running stats)
    stem.eval(); [st.eval() for st in stages]
    with torch.no_grad():
        x = stem(torch.zeros(1, 3, 64, 64))
        chans = []
        for st in stages:
            x = st(x)
            chans.append(x.shape[1])
    stem.train(); [st.train() for st in stages]
    return stem, nn.ModuleList(stages), chans


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class DenseHead(nn.Module):
    """Per-location classifier: features (B,C,H,W) -> logits map (B,K,H,W)."""

    def __init__(self, c_in, num_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, num_classes, 1),
        )

    def forward(self, x):
        return self.net(x)


def lse_pool(logit_map, r=4.0):
    """Log-sum-exp pooling over space, normalised so r->0 gives mean, r->inf max."""
    b, k, h, w = logit_map.shape
    z = logit_map.flatten(2)
    return (torch.logsumexp(r * z, dim=-1) - math.log(h * w)) / r


def hardness_map(logit_map, temperature=1.0):
    """Normalised predictive entropy in [0,1] per location, shape (B,1,H,W). Label-free."""
    p = F.softmax(logit_map / temperature, dim=1)
    ent = -(p * torch.log(p.clamp_min(1e-8))).sum(1, keepdim=True)
    return ent / math.log(logit_map.shape[1])


def normalise_spatial(h):
    """Scale a (B,1,H,W) map so that its spatial mean is 1 (keeps feature energy)."""
    return h / (h.mean(dim=(2, 3), keepdim=True) + 1e-6)


# --------------------------------------------------------------------------
# HGPB model
# --------------------------------------------------------------------------
class HGPBNet(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone="resnet50",
        pretrained=True,
        boost_stages=(1, 2, 3),   # indices of stages that get a dense head (0-based, last must be S-1)
        use_spatial=True,
        use_committee=True,
        lse_r=4.0,
        detach_hardness=True,
        baseline=False,
        main_head="committee",   # "gap": standard GAP+linear head is the output; boosting is auxiliary
    ):
        super().__init__()
        self.stem, self.stages, chans = build_backbone(backbone, pretrained)
        self.num_classes = num_classes
        self.baseline = baseline
        self.main_head = main_head
        self.use_spatial = use_spatial and not baseline
        self.use_committee = use_committee and not baseline
        self.lse_r = lse_r
        self.detach_hardness = detach_hardness
        S = len(self.stages)

        if baseline:
            # ConvNeXt's own classifier normalises pooled features with LayerNorm; keep that
            norm = [nn.LayerNorm(chans[-1])] if backbone.lower().startswith("convnext") else []
            self.fc = nn.Sequential(*norm, nn.Dropout(0.2), nn.Linear(chans[-1], num_classes))
            self.boost_stages = []
            return

        boost_stages = sorted(set(boost_stages) | {S - 1})
        self.boost_stages = boost_stages
        self.heads = nn.ModuleDict({str(s): DenseHead(chans[s], num_classes) for s in boost_stages})
        # gamma_s modulates the input of stage s+1 using hardness of stage s (init 0 = identity)
        self.gammas = nn.ParameterDict(
            {str(s): nn.Parameter(torch.zeros(1)) for s in boost_stages if s < S - 1}
        )
        # committee weights alpha (softplus(0)=0.69 for all at init)
        self.alpha_raw = nn.Parameter(torch.zeros(len(boost_stages)))
        if main_head == "gap":
            self.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(chans[-1], num_classes))

    def alphas(self):
        return F.softplus(self.alpha_raw)

    def forward(self, x, return_maps=False):
        x = self.stem(x)
        if self.baseline:
            for st in self.stages:
                x = st(x)
            feat = x.mean(dim=(2, 3))
            logits = self.fc(feat)
            out = {"logits": logits, "stage_logits": [logits], "cum_logits": [logits]}
            return out

        S = len(self.stages)
        stage_logits, maps, hard_maps = [], [], []
        pending_h = None  # hardness to apply at the input of the next stage
        pending_gamma = None
        for s, st in enumerate(self.stages):
            if pending_h is not None and self.use_spatial:
                h = F.interpolate(pending_h, size=x.shape[-2:], mode="bilinear", align_corners=False)
                x = x * (1.0 + pending_gamma * normalise_spatial(h))
            x = st(x)
            pending_h = None
            if str(s) in self.heads:
                m = self.heads[str(s)](x)
                maps.append(m)
                stage_logits.append(lse_pool(m, self.lse_r))
                if s < S - 1:
                    h = hardness_map(m)
                    hard_maps.append(h)
                    pending_h = h.detach() if self.detach_hardness else h
                    pending_gamma = self.gammas[str(s)]

        if self.use_committee:
            a = self.alphas()
            cum, run = [], 0.0
            for k, f in enumerate(stage_logits):
                run = run + a[k] * f
                cum.append(run)
            logits = cum[-1]
        else:
            cum = [stage_logits[-1]]
            logits = stage_logits[-1]
        if self.main_head == "gap":
            # the output is the standard head on the (hardness-modulated) last-stage features
            logits = self.fc(x.mean(dim=(2, 3)))

        out = {"logits": logits, "stage_logits": stage_logits, "cum_logits": cum}
        if return_maps:
            out["logit_maps"] = maps
            out["hardness_maps"] = hard_maps
        return out


if __name__ == "__main__":
    for bb in ["resnet18", "resnet50"]:
        net = HGPBNet(7, backbone=bb, pretrained=False)
        o = net(torch.randn(2, 3, 224, 224), return_maps=True)
        print(bb, o["logits"].shape, [m.shape for m in o["hardness_maps"]])
