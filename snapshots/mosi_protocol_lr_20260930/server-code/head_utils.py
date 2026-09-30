import torch
import torch.nn.functional as F


CLS7_CENTERS = (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0)
CHSIMS5_EXPECTED_CENTERS = (-0.9, -0.4, 0.0, 0.4, 0.9)
CHSIMS5_CLASS_BOUNDARIES = (-0.7, -0.1, 0.1, 0.7)
NEUTRAL_ACC7_MAX_ABS = 0.5


def _is_chsims5_config(num_classes: int, label_min: float, label_max: float) -> bool:
    return (
        int(num_classes) == 5
        and abs(float(label_min) + 1.0) < 1e-6
        and abs(float(label_max) - 1.0) < 1e-6
    )


def get_cls7_centers(
    device=None,
    dtype=torch.float32,
    num_classes: int = 7,
    label_min: float = -3.0,
    label_max: float = 3.0,
):
    num_classes = int(num_classes)
    label_min = float(label_min)
    label_max = float(label_max)
    if num_classes <= 1:
        raise ValueError(f"num_classes must be > 1, got {num_classes}")
    if label_max <= label_min:
        raise ValueError(f"label_max must be > label_min, got {label_min}..{label_max}")
    if num_classes == 7 and label_min == -3.0 and label_max == 3.0:
        values = CLS7_CENTERS
    elif _is_chsims5_config(num_classes, label_min, label_max):
        values = CHSIMS5_EXPECTED_CENTERS
    else:
        return torch.linspace(
            label_min,
            label_max,
            steps=num_classes,
            device=device,
            dtype=dtype,
        )
    return torch.tensor(values, device=device, dtype=dtype)


def continuous_to_cls7_hard(
    y: torch.Tensor, centers: torch.Tensor = None
) -> torch.Tensor:
    y = torch.as_tensor(y)
    centers = (
        get_cls7_centers(device=y.device, dtype=y.dtype)
        if centers is None
        else centers.to(device=y.device, dtype=y.dtype)
    )
    if centers.ndim != 1 or centers.numel() <= 1:
        raise ValueError(f"centers must be a 1D tensor with at least 2 values, got {tuple(centers.shape)}")
    chsims5_centers = torch.tensor(
        CHSIMS5_EXPECTED_CENTERS, device=centers.device, dtype=centers.dtype
    )
    if centers.numel() == 5 and bool(
        torch.allclose(centers, chsims5_centers, atol=1e-6, rtol=1e-6)
    ):
        boundaries = torch.tensor(
            CHSIMS5_CLASS_BOUNDARIES, device=y.device, dtype=y.dtype
        )
        return torch.bucketize(y.clamp(-1.0, 1.0), boundaries, right=False).long()
    if centers.numel() == 7 and bool(
        torch.allclose(
            centers,
            torch.tensor(CLS7_CENTERS, device=centers.device, dtype=centers.dtype),
            atol=1e-6,
            rtol=1e-6,
        )
    ):
        rounded = torch.where(y >= 0, torch.floor(y + 0.5), torch.ceil(y - 0.5))
        return rounded.clamp(-3, 3).long() + 3
    distances = torch.abs(y.reshape(-1, 1) - centers.reshape(1, -1))
    return torch.argmin(distances, dim=-1).long()


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
    classification_readout: str = "expected",
    sign_logits: torch.Tensor = None,
    sign_beta: float = 0.0,
    neutral_positive_gate_threshold: float = None,
) -> torch.Tensor:
    eta = float(eta)
    centers = (
        get_cls7_centers(device=cls7_logits.device, dtype=cls7_logits.dtype)
        if centers is None
        else centers.to(device=cls7_logits.device, dtype=cls7_logits.dtype)
    )
    classification_readout = str(classification_readout)
    if classification_readout == "expected":
        y_cls = cls7_expected_value(cls7_logits, centers=centers)
    elif classification_readout == "argmax":
        y_cls = centers[torch.argmax(cls7_logits, dim=-1)]
    else:
        raise ValueError(
            f"classification_readout must be expected or argmax, got {classification_readout}"
        )
    base = (1.0 - eta) * y_reg + eta * y_cls
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
