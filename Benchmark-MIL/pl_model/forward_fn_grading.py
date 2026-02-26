# pl_model/forward_fn_grading.py
import torch
import torch.nn.functional as F

# =========================================================
# Public API
# =========================================================
def get_forward_func_grading(mil_model: str):
    if mil_model in ["meanpooling", "maxpooling", "ABMIL", "GABMIL"]:
        return general_forward_grading
    elif mil_model == "DSMIL":
        return dsmil_forward_grading
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        return clam_forward_grading
    elif mil_model in ["TransMIL", "Transformer", "MambaMIL", "RRTMIL", "ILRA"]:
        return transmil_forward_grading
    elif mil_model == "WiKG":
        return wikg_forward_grading
    elif mil_model == "DTFD-MIL":
        return None
    else:
        raise NotImplementedError(f"Unknown MIL model: {mil_model}")


def get_attention_func_grading(mil_model: str):
    if mil_model in ["ABMIL", "GABMIL"]:
        return general_attention_func_grading
    elif mil_model in ["TransMIL", "Transformer", "ILRA"]:
        return transmil_attention_func_grading
    elif mil_model in ["CLAM-SB", "CLAM-MB"]:
        return clam_attention_func_grading
    elif mil_model == "DSMIL":
        return dsmil_attention_func_grading
    elif mil_model == "WiKG":
        return wikg_attention_func_grading
    elif mil_model == "RRTMIL":
        return rrt_attention_func_grading
    else:
        return None


# =========================================================
# Grading Loss: Cost-sensitive CE
# =========================================================
def _get_alpha_power_from_args(args, default_alpha=1.0, default_power=2.0):
    """
    priority:
      1) args.grading_alpha / args.grading_power (forward_fn 표준)
      2) args.grading_cost_lambda + (cost_type, cost_gamma) (option.py 표준)
    """
    alpha = default_alpha
    power = default_power

    if args is None:
        return alpha, power

    if hasattr(args, "grading_alpha") and getattr(args, "grading_alpha") is not None:
        alpha = float(getattr(args, "grading_alpha"))
    elif hasattr(args, "grading_cost_lambda"):
        alpha = float(getattr(args, "grading_cost_lambda", default_alpha))

    if hasattr(args, "grading_power") and getattr(args, "grading_power") is not None:
        power = float(getattr(args, "grading_power"))
    else:
        # fallback: cost_type + gamma
        cost_type = str(getattr(args, "grading_cost_type", "sq")).lower()
        gamma = float(getattr(args, "grading_cost_gamma", 1.0))
        base_power = 1.0 if cost_type == "abs" else 2.0
        power = float(base_power * gamma)

    return alpha, power


def grading_cost_sensitive_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    power: float = 2.0,
    normalize: bool = False,
    eps: float = 1e-8,
    reduction: str = "mean",
):
    """
    loss = CE + alpha * E_p[ |k - y|^power ]
    where p = softmax(logits), k in {0..K-1}
    """
    if target.ndim != 1:
        target = target.view(-1)
    target = target.long()

    ce = F.cross_entropy(logits, target, reduction="none")
    prob = torch.softmax(logits, dim=1)

    idx = torch.arange(num_classes, device=logits.device, dtype=prob.dtype).view(1, -1)
    y = target.view(-1, 1).to(prob.dtype)

    dist = torch.abs(idx - y) ** float(power)  # (B,K)
    penalty = torch.sum(prob * dist, dim=1)    # (B,)

    if normalize:
        # scale penalty to have mean ~ 1 (stabilize across K/power)
        denom = penalty.detach().mean().clamp_min(float(eps))
        penalty = penalty / denom

    loss = ce + float(alpha) * penalty

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def _compute_grading_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    loss_func,
    args=None,
):
    if label is None:
        return None

    # label -> hard index
    if isinstance(label, torch.Tensor) and label.ndim > 1:
        y = torch.argmax(label, dim=1).long()
    else:
        y = label.view(-1).long()

    grading_loss_type = (getattr(args, "grading_loss", "ce") if args is not None else "ce").lower()

    # loss_func 없으면 기본 CE로 fallback
    if loss_func is None:
        loss_func = lambda lg, yy: F.cross_entropy(lg, yy)

    if grading_loss_type in ["ce", "crossentropy", "cross_entropy"]:
        return loss_func(logits, y)

    if grading_loss_type in ["cost_sensitive_ce", "cost-sensitive-ce", "cost", "costce"]:
        alpha, power = _get_alpha_power_from_args(args, default_alpha=1.0, default_power=2.0)
        normalize = bool(getattr(args, "grading_cost_normalize", False)) if args is not None else False
        eps = float(getattr(args, "grading_cost_eps", 1e-8)) if args is not None else 1e-8

        return grading_cost_sensitive_ce(
            logits=logits,
            target=y,
            num_classes=num_classes,
            alpha=alpha,
            power=power,
            normalize=normalize,
            eps=eps,
            reduction="mean",
        )

    return loss_func(logits, y)


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


# =========================================================
# Forward functions (grading)
# =========================================================
def general_forward_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    logits = classifier(data)
    logits = _ensure_2d(logits)
    if label is None:
        return logits
    loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)
    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob


def dsmil_forward_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    ins_prediction, bag_prediction, _, _ = classifier(data)

    bag_prediction = _ensure_2d(bag_prediction)
    max_prediction, _ = torch.max(ins_prediction, 0)
    max_prediction = _ensure_2d(max_prediction)

    loss = None
    if label is not None:
        lbl = label if label.ndim == 1 else torch.argmax(label, dim=1)
        lbl = lbl.view(-1).long()

        bag_loss = _compute_grading_loss(bag_prediction, lbl, num_classes, loss_func, args=args)
        max_loss = _compute_grading_loss(max_prediction, lbl, num_classes, loss_func, args=args)
        loss = 0.5 * bag_loss + 0.5 * max_loss

    prob = torch.softmax(bag_prediction, dim=1)
    return bag_prediction, loss, prob


def clam_forward_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    logits, prob, _, _, instance_dict = classifier(data, label=label, instance_eval=True)
    logits = _ensure_2d(logits)
    prob = _ensure_2d(prob)

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)
        if isinstance(instance_dict, dict) and "instance_loss" in instance_dict:
            instance_loss = instance_dict["instance_loss"]
            loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_loss

    return logits, loss, prob


def transmil_forward_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    res = classifier(data)
    logits = res[0] if isinstance(res, tuple) else res
    logits = _ensure_2d(logits)

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)

    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob


def wikg_forward_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    res = classifier(data)
    logits = res[0] if isinstance(res, tuple) else res
    logits = _ensure_2d(logits)

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)

    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob


# =========================================================
# Attention funcs (grading)
# =========================================================
def general_attention_func_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    logits, attention_map = classifier.get_attention_maps(data)
    logits = _ensure_2d(logits)

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)

    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob, attention_map


def transmil_attention_func_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    out = classifier.get_attention_maps(data)
    logits = _ensure_2d(out[0])
    attn = out[-1]

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)

    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob, attn


def clam_attention_func_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    logits, prob, _, attn, instance_dict = classifier(data, label=label, instance_eval=True)
    logits = _ensure_2d(logits)
    prob = _ensure_2d(prob)

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)
        if isinstance(instance_dict, dict) and "instance_loss" in instance_dict:
            loss = classifier.bag_weight * loss + (1 - classifier.bag_weight) * instance_dict["instance_loss"]

    return logits, loss, prob, attn


def dsmil_attention_func_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    ins_prediction, bag_prediction, attn, _ = classifier(data)

    bag_prediction = _ensure_2d(bag_prediction)
    loss = None
    if label is not None:
        max_prediction, _ = torch.max(ins_prediction, 0)
        max_prediction = _ensure_2d(max_prediction)

        lbl = label if label.ndim == 1 else torch.argmax(label, dim=1)
        lbl = lbl.view(-1).long()

        bag_loss = _compute_grading_loss(bag_prediction, lbl, num_classes, loss_func, args=args)
        max_loss = _compute_grading_loss(max_prediction, lbl, num_classes, loss_func, args=args)
        loss = 0.5 * bag_loss + 0.5 * max_loss

    prob = torch.softmax(bag_prediction, dim=1)

    pred_label = torch.argmax(bag_prediction, dim=1)
    try:
        attn_out = attn[:, pred_label].squeeze(1)
    except Exception:
        attn_out = attn

    return bag_prediction, loss, prob, attn_out


def wikg_attention_func_grading(data, classifier, loss_func, num_classes, label=None, args=None):
    out = classifier.get_attention_maps(data)
    logits = _ensure_2d(out[0])
    attn = out[-1]

    loss = None
    if label is not None:
        loss = _compute_grading_loss(logits, label, num_classes, loss_func, args=args)

    prob = torch.softmax(logits, dim=1)
    return logits, loss, prob, attn


def rrt_attention_func_grading(*args, **kwargs):
    return None, None, None, None
