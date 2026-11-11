import torch
import torch.nn.functional as F


def boosting_weights(prev_ce, eta, w_max=5.0):
    w = torch.exp(eta * prev_ce.detach())
    w = w / w.mean()
    w = w.clamp(max=w_max)
    return w / w.mean()


def hgpb_loss(out, y, class_weights=None, label_smoothing=0.0,
              aux_w=0.5, eta=1.0, use_sample_boost=True, use_committee=True,
              main_is_committee=True):
    """main_is_committee=False when the output is a separate GAP head (v2): then every
    committee / stage term is auxiliary, including the last one."""
    def ce(z):
        return F.cross_entropy(z, y, weight=class_weights, reduction="none",
                               label_smoothing=label_smoothing)

    main = ce(out["logits"]).mean()
    logs = {"main": float(main.detach())}
    if len(out["stage_logits"]) <= 1:  # plain baseline
        return main, logs

    aux = torch.zeros((), device=main.device)
    if use_committee:
        seq = out["cum_logits"]
        prev = None
        for s, z in enumerate(seq):
            l = ce(z)
            if prev is not None and use_sample_boost:
                aux = aux + (boosting_weights(prev, eta) * l).mean()
            elif s < len(seq) - 1 or not main_is_committee:  # else: last term == `main`
                aux = aux + l.mean()
            prev = l
    else:
        heads = out["stage_logits"][:-1] if main_is_committee else out["stage_logits"]
        for z in heads:
            aux = aux + ce(z).mean()

    logs["aux"] = float(aux.detach())
    return main + aux_w * aux, logs


def make_class_weights(labels, num_classes, mode="inv_sqrt", device="cpu"):
    if mode == "none":
        return None
    counts = torch.bincount(torch.as_tensor(labels), minlength=num_classes).float().clamp_min(1)
    w = 1.0 / counts if mode == "inv" else 1.0 / counts.sqrt()
    w = w / w.sum() * num_classes
    return w.to(device)
