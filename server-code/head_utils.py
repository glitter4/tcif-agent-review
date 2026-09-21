import torch
import torch.nn.functional as F


CLS7_CENTERS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
NEUTRAL_ACC7_MAX_ABS = 0.5


def get_cls7_centers(device=None, dtype=torch.float32):
    return torch.tensor(CLS7_CENTERS, device=device, dtype=dtype)


def continuous_to_cls7_hard(y: torch.Tensor) -> torch.Tensor:
    y = torch.as_tensor(y)
    rounded = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5))
    rounded = rounded.clamp(-3, 3).long()
    return rounded + 3


def continuous_to_cls7_soft(y: torch.Tensor, centers: torch.Tensor = None, tau: float = 0.5) -> torch.Tensor:
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    y = torch.as_tensor(y)
    centers = get_cls7_centers(device=y.device, dtype=y.dtype) if centers is None else centers.to(device=y.device, dtype=y.dtype)
    distances = torch.abs(y.reshape(-1, 1) - centers.reshape(1, -1))
    logits = -distances / float(tau)
    return torch.softmax(logits, dim=-1)


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs.to(dtype=log_probs.dtype) * log_probs).sum(dim=-1).mean()


def cls7_expected_value_from_probs(probs: torch.Tensor, centers: torch.Tensor = None) -> torch.Tensor:
    centers = get_cls7_centers(device=probs.device, dtype=probs.dtype) if centers is None else centers.to(device=probs.device, dtype=probs.dtype)
    return (probs * centers.reshape(1, -1)).sum(dim=-1)


def cls7_expected_value(logits: torch.Tensor, centers: torch.Tensor = None) -> torch.Tensor:
    centers = get_cls7_centers(device=logits.device, dtype=logits.dtype) if centers is None else centers.to(device=logits.device, dtype=logits.dtype)
    probs = torch.softmax(logits, dim=-1)
    return cls7_expected_value_from_probs(probs, centers=centers)


def probs_to_logits(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp_min(float(eps))
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    return probs.log()


def hier_sign_mag_to_cls7_probs(sign_logits: torch.Tensor, mag_neg_logits: torch.Tensor, mag_pos_logits: torch.Tensor) -> torch.Tensor:
    sign_probs = torch.softmax(sign_logits, dim=-1)
    neg_mag = torch.softmax(mag_neg_logits, dim=-1)
    pos_mag = torch.softmax(mag_pos_logits, dim=-1)
    neg = sign_probs[:, 0:1]
    zero = sign_probs[:, 1:2]
    pos = sign_probs[:, 2:3]
    return torch.cat(
        [
            neg * neg_mag[:, 2:3],
            neg * neg_mag[:, 1:2],
            neg * neg_mag[:, 0:1],
            zero,
            pos * pos_mag[:, 0:1],
            pos * pos_mag[:, 1:2],
            pos * pos_mag[:, 2:3],
        ],
        dim=-1,
    )


def cumulative_logits_to_cls7_probs(boundary_logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Positive hazards produce monotone P(y > threshold) boundaries without sorting.
    hazards = torch.sigmoid(boundary_logits).clamp(min=float(eps), max=1.0 - float(eps))
    boundaries = torch.cumprod(hazards, dim=-1)
    probs = torch.cat(
        [
            1.0 - boundaries[:, 0:1],
            boundaries[:, 0:1] - boundaries[:, 1:2],
            boundaries[:, 1:2] - boundaries[:, 2:3],
            boundaries[:, 2:3] - boundaries[:, 3:4],
            boundaries[:, 3:4] - boundaries[:, 4:5],
            boundaries[:, 4:5] - boundaries[:, 5:6],
            boundaries[:, 5:6],
        ],
        dim=-1,
    )
    return probs.clamp_min(float(eps)) / probs.sum(dim=-1, keepdim=True).clamp_min(float(eps))


def compute_final_prediction(
    y_reg: torch.Tensor,
    cls7_logits: torch.Tensor,
    eta: float = 0.0,
    centers: torch.Tensor = None,
    sign_logits: torch.Tensor = None,
    sign_beta: float = 0.0,
    neutral_positive_gate_threshold: float = None,
) -> torch.Tensor:
    eta = float(eta)
    y_cls_expected = cls7_expected_value(cls7_logits, centers=centers)
    base = (1.0 - eta) * y_reg + eta * y_cls_expected
    sign_beta = float(sign_beta)
    if sign_logits is None:
        if neutral_positive_gate_threshold is not None:
            raise ValueError("neutral positive gate requires sign_logits")
        return base

    sign_unit = 2.0 * torch.sigmoid(sign_logits.reshape_as(base)) - 1.0
    if sign_beta <= 0.0:
        decoded = base
    else:
        sign_score = sign_unit * base.abs()
        decoded = (1.0 - sign_beta) * base + sign_beta * sign_score

    if neutral_positive_gate_threshold is None:
        return decoded
    threshold = float(neutral_positive_gate_threshold)
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError(
            "neutral_positive_gate_threshold must be in [-1, 1], "
            f"got {neutral_positive_gate_threshold}"
        )
    gate = (
        decoded.lt(0.0)
        & decoded.abs().lt(NEUTRAL_ACC7_MAX_ABS)
        & sign_unit.gt(threshold)
    )
    return torch.where(gate, decoded.abs(), decoded)
