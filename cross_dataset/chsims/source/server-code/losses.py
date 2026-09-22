import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_device(t: Optional[torch.Tensor], device: Optional[torch.device]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    if device is None:
        return t
    return t.to(device)


def compute_inverse_freq_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.reshape(-1)
    valid = labels >= 0
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    if valid.any():
        idx = labels[valid].to(torch.long)
        counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
    weights = torch.zeros(int(num_classes), dtype=torch.float32)
    nonzero = counts > 0
    weights[nonzero] = torch.sqrt(1.0 / counts[nonzero])
    nz = weights[nonzero]
    if nz.numel() > 0:
        m = nz.mean()
        if float(m) > 0:
            weights[nonzero] = weights[nonzero] / m
    return weights


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if alpha is None:
            self.register_buffer("alpha", None, persistent=False)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            self.register_buffer("alpha", a, persistent=False)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.to(torch.long)
        valid = target != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits_v = logits[valid]
        target_v = target[valid]

        log_probs = F.log_softmax(logits_v, dim=1)
        probs = torch.exp(log_probs)
        pt = probs.gather(1, target_v.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        ce = -log_probs.gather(1, target_v.unsqueeze(1)).squeeze(1)
        focal_w = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            alpha = self.alpha.to(logits_v.device, dtype=logits_v.dtype)
            alpha_t = alpha.gather(0, target_v)
            loss = alpha_t * focal_w * ce
        else:
            loss = focal_w * ce

        denom = loss.numel()
        if denom <= 0:
            return logits.sum() * 0.0
        return loss.sum() / denom


def build_criterion(params: Dict[str, Any], train_labels: Optional[torch.Tensor] = None) -> Tuple[nn.Module, Optional[torch.Tensor]]:
    loss_type = str(params.get("loss_type", "ce"))
    num_classes = int(params.get("num_classes", 7))
    device = params.get("device", None)
    ignore_index = int(params.get("ignore_index", -1))

    weight_tensor = None
    if loss_type == "weighted_ce":
        if train_labels is None:
            raise ValueError("train_labels is required for loss_type == 'weighted_ce'")
        weights = compute_inverse_freq_weights(train_labels, num_classes=num_classes)
        weight_tensor = _to_device(weights, device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=ignore_index)
        return criterion, weight_tensor

    if loss_type == "focal":
        alpha = None
        if train_labels is not None:
            weights = compute_inverse_freq_weights(train_labels, num_classes=num_classes)
            alpha = _to_device(weights, device)
        criterion = FocalLoss(alpha=alpha, gamma=float(params.get("focal_gamma", 2.0)), ignore_index=ignore_index)
        return criterion, alpha

    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return criterion, None


def compute_emotion_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    criterion: nn.Module,
    params: Dict[str, Any],
) -> torch.Tensor:
    loss_cls = criterion(logits, labels)
    return loss_cls
