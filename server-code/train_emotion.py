import os
import sys
import argparse
import time
import math
import random
import json
import subprocess
import re
from tcif_ablation_config import TCIF_ABLATIONS, validate_training_ablation, load_training_defaults
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_
from torchvision import transforms
from tqdm import tqdm

try:
    import numpy as np
except Exception:
    np = None

from datasets.emotion_dataset import CMUMOSEIProcessDataset, CMUMOSIProcessDataset, MultimodalEmotionDataset, _valence_to_polarity, _valence_to_7class, _valence_to_binary
from models.models_emotion import EmotionM4OE, ExpertLoadTracker, SoftMoELayerWrapper, SoftMoELayerWrapperMET, softmax
from losses import compute_inverse_freq_weights, FocalLoss, compute_emotion_loss
from head_utils import (
    cls7_expected_value,
    compute_final_prediction,
    continuous_to_cls7_hard,
    continuous_to_cls7_soft,
    get_cls7_centers,
    soft_cross_entropy,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

BEST_ACC7_CHECKPOINT_NAME = "best_acc7_model.pth"
BEST_MAE_CHECKPOINT_NAME = "best_mae_model.pth"
CHECKPOINT_SELECTION_SUMMARY = "checkpoint_selection_summary.json"
LAMBDA_HISTORY_NAME = "lambda_history.jsonl"
STRUCTV7_REMOTE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))

REG_CLS_MAG_CONSISTENCY_MODE = "shrink_reg_to_detached_cls_same_sign_same_acc7_bin"
REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE = (
    "shrink_reg_to_detached_cls_same_sign_same_acc7_bin_sign_balanced"
)
REG_CLS_MAG_CONSISTENCY_BOUNDARIES = (-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5)

def _set_seed(seed, deterministic=False):
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic

def _seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    if np is not None:
        np.random.seed(worker_seed)


class TemporalGroupWindowBatchSampler(Sampler):
    """Batch clips from the same temporal group so temporal positives are present."""

    def __init__(self, dataset, batch_size: int, temporal_window: float, seed: int = 0, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.temporal_window = float(temporal_window)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.temporal_window < 0:
            raise ValueError("temporal_window must be non-negative")
        self._items = self._extract_temporal_items(dataset)
        self._batches = self._build_batches()

    @staticmethod
    def _extract_temporal_items(dataset):
        metadata = getattr(dataset, "temporal_metadata", None)
        if metadata is None and hasattr(dataset, "ids") and hasattr(dataset, "temporal_metadata_by_id"):
            ids = list(getattr(dataset, "ids"))
            by_id = getattr(dataset, "temporal_metadata_by_id")
            metadata = [by_id[sample_id] for sample_id in ids]
        if metadata is None:
            raise RuntimeError(
                "--temporal_batch_mode group_window requires temporal metadata; "
                "refusing to silently fall back to random batching."
            )
        if len(metadata) != len(dataset):
            raise RuntimeError(
                "dataset.temporal_metadata length must match dataset length: "
                f"metadata={len(metadata)} dataset={len(dataset)}"
            )
        items = []
        for idx, row in enumerate(metadata):
            if not isinstance(row, dict) or "temporal_group_id" not in row or "temporal_pos" not in row:
                raise RuntimeError("Each temporal_metadata entry must contain temporal_group_id and temporal_pos")
            items.append((idx, int(row["temporal_group_id"]), float(row["temporal_pos"])))
        return items

    def _build_batches(self):
        groups = {}
        for idx, group_id, pos in self._items:
            groups.setdefault(group_id, []).append((pos, idx))
        batches = []
        singles = []
        for group_id in sorted(groups):
            rows = sorted(groups[group_id], key=lambda item: (item[0], item[1]))
            if len(rows) <= 1:
                singles.extend(idx for _, idx in rows)
                continue
            current = []
            start_pos = None
            for pos, idx in rows:
                exceeds_window = (
                    current
                    and self.temporal_window > 0
                    and start_pos is not None
                    and (pos - start_pos) > self.temporal_window
                )
                if len(current) >= self.batch_size or exceeds_window:
                    if len(current) == 1:
                        singles.extend(current)
                    elif len(current) == self.batch_size or not self.drop_last:
                        batches.append(current)
                    current = []
                    start_pos = None
                if not current:
                    start_pos = pos
                current.append(idx)
            if len(current) == 1:
                singles.extend(current)
            elif current and (len(current) == self.batch_size or not self.drop_last):
                batches.append(current)
        for start in range(0, len(singles), self.batch_size):
            batch = singles[start:start + self.batch_size]
            if len(batch) == self.batch_size or (batch and not self.drop_last):
                batches.append(batch)
        return batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = [list(batch) for batch in self._batches]
        rng.shuffle(batches)
        self.epoch += 1
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self._batches)


def _collect_train_raw_valences(train_dataset):
    values = []
    if isinstance(train_dataset, (CMUMOSEIProcessDataset, CMUMOSIProcessDataset)):
        for sample_id in train_dataset.ids:
            values.append(float(train_dataset.labels[sample_id]["val"]))
    elif isinstance(train_dataset, MultimodalEmotionDataset):
        for row in train_dataset.data:
            raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
            if raw_valence is None or raw_valence == "":
                values.append(0.0)
            else:
                values.append(float(raw_valence))
    else:
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            values.append(float(sample.get("raw_valence", 0.0)))
    return values


def _make_target_sampler(train_dataset, args, generator):
    mode = getattr(args, "target_sampler", "none")
    if mode == "none":
        return None
    if mode != "binary_balance":
        raise ValueError(f"Unsupported target_sampler: {mode}")
    raw_values = _collect_train_raw_valences(train_dataset)
    if len(raw_values) != len(train_dataset):
        raise RuntimeError(
            f"target_sampler expected {len(train_dataset)} labels, got {len(raw_values)}"
        )
    labels = [0 if float(v) < 0.0 else 1 for v in raw_values]
    counts = [sum(1 for y in labels if y == cls) for cls in (0, 1)]
    if min(counts) <= 0:
        raise RuntimeError(f"target_sampler=binary_balance requires both classes, counts={counts}")
    weights = [1.0 / counts[y] for y in labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    print(
        "Target sampler enabled: "
        f"mode={mode} counts=negative:{counts[0]} nonnegative:{counts[1]} "
        f"num_samples={len(weights)} replacement=True"
    )
    return sampler


def _make_train_loader(train_dataset, args, train_loader_generator):
    if args.temporal_batch_mode == "shuffle":
        target_sampler = _make_target_sampler(train_dataset, args, train_loader_generator)
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=target_sampler is None,
            sampler=target_sampler,
            num_workers=args.num_workers,
            worker_init_fn=_seed_worker,
            generator=train_loader_generator,
        )
    if getattr(args, "target_sampler", "none") != "none":
        raise ValueError("--target_sampler is currently supported only with --temporal_batch_mode shuffle")
    if args.temporal_batch_mode == "group_window":
        batch_sampler = TemporalGroupWindowBatchSampler(
            train_dataset,
            batch_size=args.batch_size,
            temporal_window=args.temporal_batch_window,
            seed=args.seed,
            drop_last=False,
        )
        print(
            "Temporal group-window batching enabled: "
            f"batches={len(batch_sampler)} batch_size={args.batch_size} window={args.temporal_batch_window}"
        )
        return DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            worker_init_fn=_seed_worker,
        )
    raise ValueError(f"Unsupported temporal_batch_mode: {args.temporal_batch_mode}")


def compute_intensity_loss(prediction, target, loss_name: str):
    if loss_name == "mse":
        return F.mse_loss(prediction, target, reduction="mean")
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target, reduction="mean")
    raise ValueError(f"Unsupported intensity loss type: {loss_name}")


def compute_regression_loss(prediction, target, loss_name: str):
    return compute_intensity_loss(prediction, target, loss_name)


def _cls7_sample_weights(raw_valence, low_abs_threshold: float = 0.0, low_abs_weight: float = 1.0):
    threshold = float(low_abs_threshold)
    low_weight = float(low_abs_weight)
    if threshold <= 0.0 or abs(low_weight - 1.0) < 1e-12:
        return None
    weights = torch.ones_like(raw_valence, dtype=torch.float32)
    weights = weights.to(device=raw_valence.device)
    weights = torch.where(raw_valence.abs() < threshold, weights.new_full(weights.shape, low_weight), weights)
    return weights


def _weighted_mean_loss(per_sample_loss, weights):
    if weights is None:
        return per_sample_loss.mean()
    return (per_sample_loss * weights.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype)).mean()


def _compute_cls7_loss(
    logits,
    raw_valence,
    loss_type: str,
    soft_tau: float,
    low_abs_threshold: float = 0.0,
    low_abs_weight: float = 1.0,
    class_weights=None,
):
    sample_weights = _cls7_sample_weights(raw_valence, low_abs_threshold, low_abs_weight)
    if loss_type == "hard_ce":
        target = continuous_to_cls7_hard(raw_valence)
        ce_weights = None
        if class_weights is not None:
            ce_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
        loss = F.cross_entropy(logits, target, weight=ce_weights, reduction="none")
        return _weighted_mean_loss(loss, sample_weights), target
    if loss_type == "soft_ce":
        centers = get_cls7_centers(device=logits.device, dtype=logits.dtype)
        target_probs = continuous_to_cls7_soft(raw_valence, centers=centers, tau=soft_tau)
        hard_target = continuous_to_cls7_hard(raw_valence)
        if class_weights is not None:
            target_probs = target_probs * class_weights.to(device=logits.device, dtype=target_probs.dtype).view(1, -1)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_probs.to(dtype=log_probs.dtype) * log_probs).sum(dim=-1)
        return _weighted_mean_loss(loss, sample_weights), hard_target
    raise ValueError(f"Unsupported cls7_loss_type: {loss_type}")


def _compute_tcif_context_aux_loss(model_out, raw_valence, args):
    zero = raw_valence.new_zeros(())
    tcif_payload = model_out.get("extras", {}).get("tcif")
    if not isinstance(tcif_payload, dict):
        return zero, zero, zero, 0
    has_context = tcif_payload.get("has_context")
    context_y_reg = tcif_payload.get("context_y_reg")
    context_cls7_logits = tcif_payload.get("context_cls7_logits")
    if (
        has_context is None
        or context_y_reg is None
        or context_cls7_logits is None
    ):
        raise RuntimeError("Incomplete TCIF context auxiliary payload")
    has_context = has_context.to(device=raw_valence.device, dtype=torch.bool)
    valid_count = int(has_context.sum().item())
    if valid_count <= 0:
        return zero, zero, zero, 0
    context_reg_loss = compute_regression_loss(
        context_y_reg[has_context],
        raw_valence[has_context],
        args.reg_loss_type,
    )
    context_cls7_target = continuous_to_cls7_hard(raw_valence[has_context])
    context_cls7_loss = F.cross_entropy(
        context_cls7_logits[has_context],
        context_cls7_target,
    )
    context_total = (
        context_reg_loss
        + float(args.tcif_context_aux_cls7_weight) * context_cls7_loss
    )
    return context_total, context_reg_loss, context_cls7_loss, valid_count


def _compute_tcif_transition_gate_loss(model_out, batch, raw_valence, args):
    zero = raw_valence.new_zeros(())
    empty_stats = {
        "valid_count": 0,
        "sign_conflict_count": 0,
        "target_mean": 0.0,
        "regression_gate_mean": 0.0,
        "ordinal_gate_mean": 0.0,
        "transition_distance_mean": 0.0,
    }
    tcif_payload = model_out.get("extras", {}).get("tcif")
    if not isinstance(tcif_payload, dict):
        return zero, empty_stats
    context_values = batch.get("tcif_context_raw_valence")
    context_valid_mask = batch.get("tcif_context_valid_mask")
    if context_values is None or context_valid_mask is None:
        raise RuntimeError(
            "TCIF transition-gate supervision requires context raw valence and mask"
        )
    context_values = context_values.to(
        device=raw_valence.device,
        dtype=raw_valence.dtype,
    )
    context_valid_mask = context_valid_mask.to(
        device=raw_valence.device,
        dtype=torch.bool,
    )
    valid_sample = context_valid_mask.any(dim=1)
    valid_count = int(valid_sample.sum().item())
    if valid_count <= 0:
        return zero, empty_stats

    mask_float = context_valid_mask.to(dtype=raw_valence.dtype)
    context_mean = (context_values * mask_float).sum(dim=1) / mask_float.sum(
        dim=1
    ).clamp_min(1.0)
    transition_distance = (raw_valence - context_mean).abs()
    target = torch.exp(
        -transition_distance / float(args.tcif_transition_gate_tau)
    )
    center_nonzero = raw_valence.abs() > 1e-12
    context_nonzero = context_values.abs() > 1e-12
    opposite_sign = (
        torch.sign(context_values) != torch.sign(raw_valence).unsqueeze(1)
    )
    sign_conflict = center_nonzero & (
        context_valid_mask & context_nonzero & opposite_sign
    ).any(dim=1)
    target = torch.where(
        sign_conflict,
        target.new_full(
            target.shape,
            float(args.tcif_transition_gate_conflict_target),
        ),
        target,
    ).detach()

    regression_logits = tcif_payload.get("regression_continuation_gate_logits")
    ordinal_logits = tcif_payload.get("ordinal_continuation_gate_logits")
    regression_gate = tcif_payload.get("regression_continuation_gate")
    ordinal_gate = tcif_payload.get("ordinal_continuation_gate")
    if any(
        value is None
        for value in (
            regression_logits,
            ordinal_logits,
            regression_gate,
            ordinal_gate,
        )
    ):
        raise RuntimeError("Incomplete TCIF transition-gate payload")
    regression_loss = F.binary_cross_entropy_with_logits(
        regression_logits[valid_sample],
        target[valid_sample].to(dtype=regression_logits.dtype),
    )
    ordinal_loss = F.binary_cross_entropy_with_logits(
        ordinal_logits[valid_sample],
        target[valid_sample].to(dtype=ordinal_logits.dtype),
    )
    loss = 0.5 * (regression_loss + ordinal_loss)
    stats = {
        "valid_count": valid_count,
        "sign_conflict_count": int((sign_conflict & valid_sample).sum().item()),
        "target_mean": float(target[valid_sample].mean().detach().item()),
        "regression_gate_mean": float(
            regression_gate[valid_sample].mean().detach().item()
        ),
        "ordinal_gate_mean": float(
            ordinal_gate[valid_sample].mean().detach().item()
        ),
        "transition_distance_mean": float(
            transition_distance[valid_sample].mean().detach().item()
        ),
    }
    return loss, stats


def _sign_struct_scale(args, epoch: int | None = None) -> float:
    if epoch is None:
        return float(getattr(args, "_sign_struct_scale", 1.0))
    ratio = float(getattr(args, "sign_struct_warmup_ratio", 0.0))
    if ratio <= 0.0:
        return 1.0
    warmup_epochs = int(math.ceil(max(0, int(getattr(args, "epochs", 0))) * ratio))
    return 0.0 if int(epoch) < warmup_epochs else 1.0


def _raw_valence_to_sign3(raw_valence):
    return torch.where(
        raw_valence < 0.0,
        torch.zeros_like(raw_valence, dtype=torch.long),
        torch.where(
            raw_valence > 0.0,
            torch.full_like(raw_valence, 2, dtype=torch.long),
            torch.ones_like(raw_valence, dtype=torch.long),
        ),
    )


def _raw_valence_to_magnitude(raw_valence):
    cls7 = continuous_to_cls7_hard(raw_valence)
    mag = torch.zeros_like(cls7)
    mag = torch.where(cls7 < 3, 2 - cls7, mag)
    mag = torch.where(cls7 > 3, cls7 - 4, mag)
    return mag.clamp(0, 2)


def _compute_hier_sign_mag_losses(model_out, raw_valence):
    extras = model_out.get("extras", {}) or {}
    sign_logits = extras.get("hier_sign_logits")
    neg_logits = extras.get("hier_mag_neg_logits")
    pos_logits = extras.get("hier_mag_pos_logits")
    y_reg = model_out["y_reg"]
    zero = y_reg.new_zeros(())
    if sign_logits is None or neg_logits is None or pos_logits is None:
        return zero, zero
    sign_target = _raw_valence_to_sign3(raw_valence)
    sign_loss = F.cross_entropy(sign_logits, sign_target)
    mag_target = _raw_valence_to_magnitude(raw_valence)
    neg_mask = raw_valence < 0.0
    pos_mask = raw_valence > 0.0
    mag_losses = []
    if neg_mask.any():
        mag_losses.append(F.cross_entropy(neg_logits[neg_mask], mag_target[neg_mask]))
    if pos_mask.any():
        mag_losses.append(F.cross_entropy(pos_logits[pos_mask], mag_target[pos_mask]))
    mag_loss = torch.stack(mag_losses).mean() if mag_losses else zero
    return sign_loss, mag_loss


def _compute_sign_marginal_loss(cls7_logits, raw_valence, args):
    mode = str(getattr(args, "sign_marginal_loss_type", "none"))
    if mode == "none":
        return cls7_logits.new_zeros(())
    nonzero = raw_valence.abs() > 1e-12
    if not nonzero.any():
        return cls7_logits.sum() * 0.0
    probs = torch.softmax(cls7_logits[nonzero], dim=-1)
    p_neg = probs[:, :3].sum(dim=-1)
    p_pos = probs[:, 4:].sum(dim=-1)
    denom = (p_neg + p_pos).clamp_min(1e-8)
    p_pos_norm = (p_pos / denom).clamp(min=1e-6, max=1.0 - 1e-6)
    target = (raw_valence[nonzero] > 0.0).to(dtype=p_pos_norm.dtype)
    if mode == "focal":
        pt = torch.where(target > 0.5, p_pos_norm, 1.0 - p_pos_norm)
        ce = F.binary_cross_entropy(p_pos_norm, target, reduction="none")
        return (torch.pow((1.0 - pt).clamp_min(1e-8), float(args.sign_marginal_focal_gamma)) * ce).mean()
    if mode == "soft_f1":
        tp_pos = (p_pos_norm * target).sum()
        fp_pos = (p_pos_norm * (1.0 - target)).sum()
        fn_pos = ((1.0 - p_pos_norm) * target).sum()
        p_neg_norm = 1.0 - p_pos_norm
        target_neg = 1.0 - target
        tp_neg = (p_neg_norm * target_neg).sum()
        fp_neg = (p_neg_norm * (1.0 - target_neg)).sum()
        fn_neg = ((1.0 - p_neg_norm) * target_neg).sum()
        eps = cls7_logits.new_tensor(1e-8)
        f1_pos = 2.0 * tp_pos / (2.0 * tp_pos + fp_pos + fn_pos + eps)
        f1_neg = 2.0 * tp_neg / (2.0 * tp_neg + fp_neg + fn_neg + eps)
        return 1.0 - 0.5 * (f1_pos + f1_neg)
    raise ValueError(f"Unsupported sign_marginal_loss_type: {mode}")


def _compute_signed_neutral_band_loss(model_out, raw_valence, args):
    """Keep tiny non-zero labels inside the Acc7 neutral bin with the right sign."""
    y_reg = model_out["y_reg"]
    extras = model_out.get("extras", {}) or {}
    y_cls_expected = extras.get("y_cls_expected")
    if y_cls_expected is None:
        return y_reg.sum() * 0.0

    mask = (raw_valence.abs() > 1e-12) & (raw_valence.abs() < 0.5)
    if not mask.any():
        return y_reg.sum() * 0.0

    eta = float(getattr(args, "signed_neutral_band_eta", 0.4))
    margin = float(getattr(args, "signed_neutral_band_margin", 1.0 / 6.0))
    upper = float(getattr(args, "signed_neutral_band_upper", 0.45))
    temperature = float(getattr(args, "signed_neutral_band_temperature", 0.05))
    score = (1.0 - eta) * y_reg[mask] + eta * y_cls_expected[mask]
    sign = torch.sign(raw_valence[mask]).to(dtype=score.dtype)
    signed_score = sign * score
    lower_barrier = temperature * F.softplus((margin - signed_score) / temperature)
    upper_barrier = temperature * F.softplus((signed_score - upper) / temperature)
    per_sample = lower_barrier + upper_barrier

    # Balance the two signs so the 325:574 tiny-negative/tiny-positive skew does
    # not turn this into another implicit class-weight experiment.
    zero_loss = per_sample.sum() * 0.0
    sign_losses = []
    for sign_value in (-1.0, 1.0):
        selected = sign == sign_value
        sign_losses.append(per_sample[selected].mean() if selected.any() else zero_loss)
    return 0.5 * (sign_losses[0] + sign_losses[1])


def _compute_zero_sign_margin_loss(model_out, raw_valence, args):
    """Nudge exact-zero labels to the non-negative side at both fusion endpoints.

    MOSEI's binary convention assigns exact-zero labels to the positive class.
    Applying the same one-sided soft margin to ``y_reg`` and the cls7 expected
    value makes the supervision available to every eta in the post-training
    sweep, while leaving non-zero examples entirely untouched.
    """
    y_reg = model_out["y_reg"]
    extras = model_out.get("extras", {}) or {}
    y_cls_expected = extras.get("y_cls_expected")
    if y_cls_expected is None:
        return y_reg.sum() * 0.0

    zero_mask = raw_valence == 0.0
    if not zero_mask.any():
        return (y_reg.sum() + y_cls_expected.sum()) * 0.0

    margin = float(getattr(args, "zero_sign_margin", 0.02))
    temperature = float(getattr(args, "zero_sign_margin_temperature", 0.02))
    endpoints = torch.stack((y_reg[zero_mask], y_cls_expected[zero_mask]), dim=0)
    per_endpoint = temperature * F.softplus((margin - endpoints) / temperature)
    return per_endpoint.mean()


def _empty_reg_cls_mag_consistency_stats(candidate_count=0):
    return {
        "selected_count": 0,
        "positive_selected_count": 0,
        "negative_selected_count": 0,
        "sign_balanced_reduction_applied": 0,
        "single_branch_fallback": 0,
        "candidate_count": int(candidate_count),
        "coverage": 0.0,
        "mean_abs_gap": 0.0,
        "abs_gap_sum": 0.0,
    }


def _apply_detached_sign_beta(base, sign_logits, sign_beta):
    """Apply the production sign decoder without sending gradients to sign_logits."""
    sign_beta = float(sign_beta)
    if sign_logits is None or sign_beta <= 0.0:
        return base
    detached_sign_score = 2.0 * torch.sigmoid(sign_logits.detach().reshape_as(base)) - 1.0
    return (1.0 - sign_beta) * base + sign_beta * detached_sign_score * base.abs()


def _reg_cls_mag_consistency_decoded_endpoints(model_out, args):
    """Return production-valued eta=0/1 endpoints with teacher paths detached."""
    y_reg = model_out["y_reg"]
    extras = model_out.get("extras", {}) or {}
    y_cls_expected = extras.get("y_cls_expected")
    if y_cls_expected is None:
        return None, None
    sign_logits = model_out.get("sign_logits", None)
    sign_beta = float(getattr(args, "final_pred_sign_beta", 0.0))
    clamp_eval = bool(getattr(args, "clamp_regression_eval", True))
    if clamp_eval:
        # Match production's clamped forward value while retaining an inward
        # consistency gradient for saturated regression outputs.
        y_reg_st = y_reg + (y_reg.clamp(-3.0, 3.0) - y_reg).detach()
    else:
        y_reg_st = y_reg
    decoded_reg = _apply_detached_sign_beta(y_reg_st, sign_logits, sign_beta)
    decoded_cls = _apply_detached_sign_beta(y_cls_expected.detach(), sign_logits, sign_beta).detach()
    return decoded_reg, decoded_cls


def _compute_reg_cls_mag_consistency_loss(model_out, args):
    """Shrink the decoded regression magnitude toward a detached cls7 target.

    Eligibility is deliberately detached and conservative: the eta=0 and eta=1
    decoded endpoints must have the same non-zero sign, occupy the same Acc7 bin,
    stay away from both Acc7 and binary-sign boundaries, and the regression
    endpoint must have the larger magnitude.  The loss therefore has a direct
    gradient only through ``y_reg``; cls7 and sign heads act as fixed teachers.
    """
    y_reg = model_out["y_reg"]
    stats = _empty_reg_cls_mag_consistency_stats(y_reg.numel())
    mode = str(getattr(args, "reg_cls_mag_consistency_mode", REG_CLS_MAG_CONSISTENCY_MODE))
    supported_modes = {
        REG_CLS_MAG_CONSISTENCY_MODE,
        REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE,
    }
    if mode not in supported_modes:
        raise ValueError(f"Unsupported reg_cls_mag_consistency_mode: {mode}")

    decoded_reg, decoded_cls = _reg_cls_mag_consistency_decoded_endpoints(model_out, args)
    if decoded_reg is None or decoded_cls is None:
        return y_reg.sum() * 0.0, stats

    margin = float(getattr(args, "reg_cls_mag_consistency_boundary_margin", 0.1))
    smooth_l1_beta = float(getattr(args, "reg_cls_mag_consistency_smooth_l1_beta", 0.1))
    with torch.no_grad():
        reg_endpoint = decoded_reg.detach()
        cls_endpoint = decoded_cls.detach()
        boundaries = reg_endpoint.new_tensor(REG_CLS_MAG_CONSISTENCY_BOUNDARIES).reshape(1, -1)
        reg_boundary_distance = (reg_endpoint.reshape(-1, 1) - boundaries).abs().amin(dim=-1)
        cls_boundary_distance = (cls_endpoint.reshape(-1, 1) - boundaries).abs().amin(dim=-1)
        same_nonzero_sign = reg_endpoint * cls_endpoint > 0.0
        same_acc7_bin = continuous_to_cls7_hard(reg_endpoint) == continuous_to_cls7_hard(cls_endpoint)
        boundary_safe = (reg_boundary_distance >= margin) & (cls_boundary_distance >= margin)
        shrink_only = reg_endpoint.abs() > cls_endpoint.abs()
        selected = same_nonzero_sign & same_acc7_bin & boundary_safe & shrink_only

    selected_count = int(selected.sum().item())
    if selected_count == 0:
        return y_reg.sum() * 0.0, stats

    selected_gap = decoded_reg.detach()[selected].abs() - decoded_cls[selected].abs()
    per_selected_loss = F.smooth_l1_loss(
        decoded_reg[selected].abs(),
        decoded_cls[selected].abs(),
        reduction="none",
        beta=smooth_l1_beta,
    )
    selected_positive = reg_endpoint[selected] > 0.0
    positive_selected_count = int(selected_positive.sum().item())
    negative_selected_count = selected_count - positive_selected_count
    sign_balanced_reduction_applied = (
        mode == REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
        and positive_selected_count > 0
        and negative_selected_count > 0
    )
    single_branch_fallback = (
        mode == REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
        and not sign_balanced_reduction_applied
    )
    if sign_balanced_reduction_applied:
        loss = 0.5 * (
            per_selected_loss[selected_positive].mean()
            + per_selected_loss[~selected_positive].mean()
        )
    else:
        loss = per_selected_loss.mean()
    abs_gap_sum = float(selected_gap.sum().item())
    stats.update(
        {
            "selected_count": selected_count,
            "positive_selected_count": positive_selected_count,
            "negative_selected_count": negative_selected_count,
            "sign_balanced_reduction_applied": int(
                sign_balanced_reduction_applied
            ),
            "single_branch_fallback": int(single_branch_fallback),
            "coverage": selected_count / max(1, int(y_reg.numel())),
            "mean_abs_gap": abs_gap_sum / selected_count,
            "abs_gap_sum": abs_gap_sum,
        }
    )
    return loss, stats


def _compute_cumulative_loss(model_out, raw_valence):
    extras = model_out.get("extras", {}) or {}
    logits = extras.get("cumulative_logits")
    y_reg = model_out["y_reg"]
    if logits is None:
        return y_reg.new_zeros(())
    target_cls = continuous_to_cls7_hard(raw_valence).to(device=logits.device)
    thresholds = torch.arange(6, device=logits.device).reshape(1, -1)
    targets = (target_cls.reshape(-1, 1) > thresholds).to(dtype=logits.dtype)
    hazards = torch.sigmoid(logits).clamp(min=1e-6, max=1.0 - 1e-6)
    boundaries = torch.cumprod(hazards, dim=-1).clamp(min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(boundaries, targets)


def _compute_sign_aux_loss(logits, raw_valence, args):
    sign_target = (raw_valence >= 0).to(dtype=logits.dtype)
    per_sample = F.binary_cross_entropy_with_logits(logits, sign_target, reduction="none")
    if args.sign_aux_loss_type == "focal_bce":
        probs = torch.sigmoid(logits)
        p_t = torch.where(sign_target > 0.5, probs, 1.0 - probs)
        per_sample = per_sample * torch.pow((1.0 - p_t).clamp_min(1e-8), args.sign_aux_focal_gamma)
    elif args.sign_aux_loss_type != "bce":
        raise ValueError(f"Unsupported sign_aux_loss_type: {args.sign_aux_loss_type}")

    weights = torch.ones_like(per_sample)
    low_abs_threshold = float(getattr(args, "sign_aux_low_abs_weight_threshold", 0.0))
    if low_abs_threshold > 0.0 and abs(float(args.sign_aux_low_abs_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() < low_abs_threshold,
            weights.new_full(weights.shape, float(args.sign_aux_low_abs_weight)),
            weights,
        )
    if abs(float(args.sign_aux_zero_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() <= 1e-12,
            weights.new_full(weights.shape, float(args.sign_aux_zero_weight)),
            weights,
        )
    if abs(float(args.sign_aux_nonzero_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() > 1e-12,
            weights.new_full(weights.shape, float(args.sign_aux_nonzero_weight)),
            weights,
        )
    positive_weight = float(getattr(args, "sign_aux_positive_weight", 1.0))
    negative_weight = float(getattr(args, "sign_aux_negative_weight", 1.0))
    if abs(positive_weight - 1.0) > 1e-12 or abs(negative_weight - 1.0) > 1e-12:
        class_weights = torch.where(
            sign_target > 0.5,
            weights.new_full(weights.shape, positive_weight),
            weights.new_full(weights.shape, negative_weight),
        )
        weights = weights * class_weights
    return _weighted_mean_loss(per_sample, weights)


def _compute_score_sign_aux_loss(scores, raw_valence, args):
    temperature = float(args.score_sign_aux_temperature)
    logits = scores / temperature
    sign_target = (raw_valence >= 0).to(dtype=logits.dtype)
    per_sample = F.binary_cross_entropy_with_logits(logits, sign_target, reduction="none")
    if args.score_sign_aux_loss_type == "focal_bce":
        probs = torch.sigmoid(logits)
        p_t = torch.where(sign_target > 0.5, probs, 1.0 - probs)
        per_sample = per_sample * torch.pow((1.0 - p_t).clamp_min(1e-8), args.score_sign_aux_focal_gamma)
    elif args.score_sign_aux_loss_type != "bce":
        raise ValueError(f"Unsupported score_sign_aux_loss_type: {args.score_sign_aux_loss_type}")

    weights = torch.ones_like(per_sample)
    low_abs_threshold = float(getattr(args, "score_sign_aux_low_abs_weight_threshold", 0.0))
    if low_abs_threshold > 0.0 and abs(float(args.score_sign_aux_low_abs_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() < low_abs_threshold,
            weights.new_full(weights.shape, float(args.score_sign_aux_low_abs_weight)),
            weights,
        )
    if abs(float(args.score_sign_aux_zero_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() <= 1e-12,
            weights.new_full(weights.shape, float(args.score_sign_aux_zero_weight)),
            weights,
        )
    if abs(float(args.score_sign_aux_nonzero_weight) - 1.0) > 1e-12:
        weights = torch.where(
            raw_valence.abs() > 1e-12,
            weights.new_full(weights.shape, float(args.score_sign_aux_nonzero_weight)),
            weights,
        )
    positive_weight = float(getattr(args, "score_sign_aux_positive_weight", 1.0))
    negative_weight = float(getattr(args, "score_sign_aux_negative_weight", 1.0))
    if abs(positive_weight - 1.0) > 1e-12 or abs(negative_weight - 1.0) > 1e-12:
        class_weights = torch.where(
            sign_target > 0.5,
            weights.new_full(weights.shape, positive_weight),
            weights.new_full(weights.shape, negative_weight),
        )
        weights = weights * class_weights
    return _weighted_mean_loss(per_sample, weights)


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, input_dim: int, proj_dim: int = 128):
        super().__init__()
        input_dim = int(input_dim)
        proj_dim = int(proj_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, features):
        return F.normalize(self.net(features), dim=-1)


def compute_ordinal_distribution_contrastive_loss(
    projection_head,
    features,
    y,
    cls7,
    tau: float = 0.15,
    sigma_y: float = 0.7,
    alpha_c: float = 0.2,
):
    if tau <= 0:
        raise ValueError("tau must be positive")
    if sigma_y <= 0:
        raise ValueError("sigma_y must be positive")
    z = projection_head(features)
    batch_size = int(z.shape[0])
    zero = z.new_zeros(())
    if batch_size <= 1:
        return zero

    y = y.to(device=z.device, dtype=z.dtype).reshape(-1)
    cls7 = cls7.to(device=z.device, dtype=z.dtype).reshape(-1)
    if y.shape[0] != batch_size or cls7.shape[0] != batch_size:
        raise ValueError(
            "features, y, and cls7 must have matching batch size: "
            f"features={batch_size} y={y.shape[0]} cls7={cls7.shape[0]}"
        )

    not_self = ~torch.eye(batch_size, device=z.device, dtype=torch.bool)
    logits = (z @ z.transpose(0, 1)) / float(tau)
    logits = logits.masked_fill(~not_self, torch.finfo(logits.dtype).min)

    dist_y = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))
    dist_c = torch.abs(cls7.unsqueeze(0) - cls7.unsqueeze(1))
    target_score = -dist_y / float(sigma_y) - float(alpha_c) * dist_c
    target_score = target_score.masked_fill(~not_self, torch.finfo(target_score.dtype).min)

    log_p = F.log_softmax(logits, dim=1)
    log_q = F.log_softmax(target_score, dim=1)
    q = torch.exp(log_q)
    return (q * (log_q - log_p)).sum(dim=1).mean()


def _oacr_enabled(args):
    return (
        bool(getattr(args, "enable_oacr", False))
        and float(getattr(args, "lambda_oacr", 0.0)) > 0.0
    )


def _compute_oacr_loss(model_out, raw_valence, cls7_target, args, projection_head):
    y_reg = model_out["y_reg"]
    if projection_head is None or not _oacr_enabled(args):
        return y_reg.new_zeros(())
    extras = model_out.get("extras", {}) or {}
    feat_task2 = extras.get("feat_task2")
    if feat_task2 is None:
        raise RuntimeError("OACR is enabled but model did not return extras['feat_task2']")
    return compute_ordinal_distribution_contrastive_loss(
        projection_head,
        feat_task2,
        raw_valence,
        cls7_target,
        tau=args.oacr_tau,
        sigma_y=args.oacr_sigma_y,
        alpha_c=args.oacr_alpha_c,
    )


TEMPORAL_KERNEL_CHOICES = (
    "legacy_exp",
    "legacy_sign_safe",
    "uniform",
    "hard_window",
    "linear",
    "gaussian",
    "exp",
    "semantic_exp",
    "label_exp",
    "sign_safe_exp",
)


def compute_temporal_positive_weights(
    embeddings,
    temporal_group_id,
    temporal_pos,
    decay_tau: float = 1.0,
    positive_radius: float = 1.0,
    weak_positive_radius: float = 4.0,
    min_positive_weight: float = 0.2,
    temporal_kernel: str = "legacy_exp",
    raw_valence=None,
    label_gate_beta: float = 1.0,
    zero_bridge_weight: float = 0.25,
):
    if decay_tau <= 0:
        raise ValueError("decay_tau must be positive")
    if positive_radius < 0:
        raise ValueError("positive_radius must be non-negative")
    if weak_positive_radius < positive_radius:
        raise ValueError("weak_positive_radius must be >= positive_radius")
    if min_positive_weight < 0:
        raise ValueError("min_positive_weight must be non-negative")
    if not (0.0 <= float(zero_bridge_weight) <= 1.0):
        raise ValueError("zero_bridge_weight must be in [0, 1]")
    if temporal_kernel not in TEMPORAL_KERNEL_CHOICES:
        raise ValueError(f"Unsupported temporal_kernel: {temporal_kernel}")
    if embeddings is None:
        raise ValueError("temporal embeddings are required for temporal contrastive loss")

    features = F.normalize(embeddings, dim=-1)
    batch_size = int(features.shape[0])
    empty_stats = {
        "valid_anchor_count": 0,
        "skipped_anchor_count": batch_size,
        "positive_pair_count": 0,
        "mean_positive_weight": 0.0,
        "positive_mass": 0.0,
        "legacy_valid_anchor_count": 0,
        "legacy_positive_pair_count": 0,
        "legacy_positive_mass": 0.0,
        "temporal_kernel": temporal_kernel,
    }
    if batch_size <= 1:
        return features.new_zeros((batch_size, batch_size)), empty_stats

    group_id = temporal_group_id.to(device=features.device).reshape(-1)
    pos = temporal_pos.to(device=features.device, dtype=features.dtype).reshape(-1)
    if group_id.shape[0] != batch_size or pos.shape[0] != batch_size:
        raise ValueError(
            "temporal metadata must have one value per embedding: "
            f"embeddings={batch_size} group_id={group_id.shape[0]} pos={pos.shape[0]}"
        )

    same_group = group_id.unsqueeze(0).eq(group_id.unsqueeze(1))
    not_self = ~torch.eye(batch_size, device=features.device, dtype=torch.bool)
    distance = torch.abs(pos.unsqueeze(0) - pos.unsqueeze(1))
    valid_window = same_group & not_self & (distance <= float(weak_positive_radius))
    strong_positive = distance <= float(positive_radius)
    weak_positive = (distance > float(positive_radius)) & (distance <= float(weak_positive_radius))
    exp_weights = torch.exp(-distance / float(decay_tau))
    label_gate = None
    if temporal_kernel in {"label_exp", "sign_safe_exp", "legacy_sign_safe"}:
        if raw_valence is None:
            raise ValueError(f"raw_valence is required for temporal_kernel={temporal_kernel}")
        if label_gate_beta <= 0:
            raise ValueError("label_gate_beta must be positive")
        y = raw_valence.to(device=features.device, dtype=features.dtype).reshape(-1)
        if y.shape[0] != batch_size:
            raise ValueError(f"raw_valence must have one value per embedding: embeddings={batch_size} raw_valence={y.shape[0]}")
        with torch.no_grad():
            if temporal_kernel == "label_exp":
                label_gate = torch.exp(-torch.abs(y.unsqueeze(0) - y.unsqueeze(1)) / float(label_gate_beta))
            elif temporal_kernel == "sign_safe_exp":
                sign = torch.sign(y)
                same_sign = sign.unsqueeze(0).eq(sign.unsqueeze(1))
                near_zero = (y.abs().unsqueeze(0) <= float(label_gate_beta)) | (y.abs().unsqueeze(1) <= float(label_gate_beta))
                label_gate = (same_sign | near_zero).to(dtype=features.dtype, device=features.device)
            else:
                zero = y.abs() <= 1e-12
                left_zero = zero.unsqueeze(1)
                right_zero = zero.unsqueeze(0)
                both_zero = left_zero & right_zero
                zero_bridge = left_zero ^ right_zero
                sign = torch.sign(y)
                same_nonzero_sign = sign.unsqueeze(1).eq(sign.unsqueeze(0)) & ~left_zero & ~right_zero
                label_gate = torch.zeros_like(distance)
                label_gate = torch.where(
                    same_nonzero_sign | both_zero,
                    torch.ones_like(label_gate),
                    label_gate,
                )
                label_gate = torch.where(
                    zero_bridge,
                    torch.full_like(label_gate, float(zero_bridge_weight)),
                    label_gate,
                )

    legacy_weights = torch.zeros_like(exp_weights)
    strong_mask = same_group & not_self & strong_positive
    weak_mask = same_group & not_self & weak_positive
    legacy_weights = torch.where(strong_mask, torch.clamp(exp_weights, min=float(min_positive_weight)), legacy_weights)
    legacy_weights = torch.where(weak_mask, exp_weights, legacy_weights)

    if temporal_kernel == "legacy_exp":
        positive_weights = legacy_weights
    elif temporal_kernel == "legacy_sign_safe":
        positive_weights = legacy_weights * label_gate
    elif temporal_kernel == "uniform":
        positive_weights = torch.where(valid_window, torch.ones_like(exp_weights), torch.zeros_like(exp_weights))
    elif temporal_kernel == "hard_window":
        hard_mask = same_group & not_self & strong_positive
        positive_weights = torch.where(hard_mask, torch.ones_like(exp_weights), torch.zeros_like(exp_weights))
    elif temporal_kernel == "linear":
        denom = max(float(weak_positive_radius), 1e-12)
        linear_weights = torch.clamp(1.0 - distance / denom, min=0.0)
        positive_weights = torch.where(valid_window, linear_weights, torch.zeros_like(linear_weights))
    elif temporal_kernel == "gaussian":
        gaussian_weights = torch.exp(-0.5 * (distance / float(decay_tau)).pow(2))
        positive_weights = torch.where(valid_window, gaussian_weights, torch.zeros_like(gaussian_weights))
    elif temporal_kernel == "exp":
        positive_weights = torch.where(valid_window, exp_weights, torch.zeros_like(exp_weights))
    elif temporal_kernel == "semantic_exp":
        with torch.no_grad():
            detached = F.normalize(embeddings.detach(), dim=-1)
            semantic_gate = torch.relu(detached @ detached.transpose(0, 1)).to(dtype=features.dtype, device=features.device)
        positive_weights = torch.where(valid_window, exp_weights * semantic_gate, torch.zeros_like(exp_weights))
    elif temporal_kernel in {"label_exp", "sign_safe_exp"}:
        positive_weights = torch.where(valid_window, exp_weights * label_gate, torch.zeros_like(exp_weights))
    else:
        raise AssertionError(f"Unhandled temporal_kernel: {temporal_kernel}")

    positive_weights = positive_weights.detach()
    positive_mass = positive_weights.sum(dim=1)
    valid_anchor = positive_mass > 0
    valid_count = int(valid_anchor.detach().sum().item())
    positive_pair_count = int((positive_weights > 0).detach().sum().item())
    legacy_mass = legacy_weights.detach().sum(dim=1)
    legacy_valid_count = int((legacy_mass > 0).detach().sum().item())
    legacy_pair_count = int((legacy_weights > 0).detach().sum().item())
    coverage_stats = {
        "positive_mass": float(positive_mass.detach().sum().item()),
        "legacy_valid_anchor_count": legacy_valid_count,
        "legacy_positive_pair_count": legacy_pair_count,
        "legacy_positive_mass": float(legacy_mass.detach().sum().item()),
    }
    if valid_count == 0:
        empty_stats.update(coverage_stats)
        return positive_weights, empty_stats

    selected_weights = positive_weights[positive_weights > 0]
    stats = {
        "valid_anchor_count": valid_count,
        "skipped_anchor_count": batch_size - valid_count,
        "positive_pair_count": positive_pair_count,
        "mean_positive_weight": float(selected_weights.detach().mean().item()) if selected_weights.numel() > 0 else 0.0,
        "temporal_kernel": temporal_kernel,
    }
    stats.update(coverage_stats)
    return positive_weights, stats


def compute_soft_temporal_contrastive_loss(
    embeddings,
    temporal_group_id,
    temporal_pos,
    temperature: float = 0.07,
    decay_tau: float = 1.0,
    positive_radius: float = 1.0,
    weak_positive_radius: float = 4.0,
    min_positive_weight: float = 0.2,
    temporal_kernel: str = "legacy_exp",
    raw_valence=None,
    label_gate_beta: float = 1.0,
    zero_bridge_weight: float = 0.25,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings is None:
        raise ValueError("temporal embeddings are required for temporal contrastive loss")

    features = F.normalize(embeddings, dim=-1)
    batch_size = int(features.shape[0])
    zero = features.new_zeros(())
    positive_weights, stats = compute_temporal_positive_weights(
        embeddings,
        temporal_group_id,
        temporal_pos,
        decay_tau=decay_tau,
        positive_radius=positive_radius,
        weak_positive_radius=weak_positive_radius,
        min_positive_weight=min_positive_weight,
        temporal_kernel=temporal_kernel,
        raw_valence=raw_valence,
        label_gate_beta=label_gate_beta,
        zero_bridge_weight=zero_bridge_weight,
    )
    positive_mass = positive_weights.sum(dim=1)
    valid_anchor = positive_mass > 0
    valid_count = int(valid_anchor.detach().sum().item())
    if batch_size <= 1 or valid_count == 0:
        return zero, stats

    logits = (features @ features.transpose(0, 1)) / float(temperature)
    not_self = ~torch.eye(batch_size, device=features.device, dtype=torch.bool)
    logits = logits.masked_fill(~not_self, torch.finfo(logits.dtype).min)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    targets = positive_weights / positive_mass.clamp_min(1e-12).unsqueeze(1)
    per_anchor_loss = -(targets[valid_anchor] * log_prob[valid_anchor]).sum(dim=1)
    if temporal_kernel == "legacy_sign_safe":
        # Target normalization would otherwise cancel a sub-unit zero bridge
        # whenever it is an anchor's only surviving positive. Preserve its
        # intended attenuation through safe/legacy positive-mass reliability.
        legacy_weights, _ = compute_temporal_positive_weights(
            embeddings,
            temporal_group_id,
            temporal_pos,
            decay_tau=decay_tau,
            positive_radius=positive_radius,
            weak_positive_radius=weak_positive_radius,
            min_positive_weight=min_positive_weight,
            temporal_kernel="legacy_exp",
        )
        legacy_mass = legacy_weights.sum(dim=1).clamp_min(1e-12)
        reliability = (positive_mass / legacy_mass).clamp(min=0.0, max=1.0)
        valid_reliability = reliability[valid_anchor]
        loss = (per_anchor_loss * valid_reliability).sum() / valid_reliability.sum().clamp_min(1e-12)
        stats["mean_anchor_reliability"] = float(valid_reliability.detach().mean().item())
        stats["effective_anchor_mass"] = float(valid_reliability.detach().sum().item())
    else:
        loss = per_anchor_loss.mean()
    return loss, stats


def _temporal_contrast_enabled(args):
    return (
        getattr(args, "output_head_mode", "legacy") != "legacy"
        and (
            bool(getattr(args, "enable_temporal_contrast_experts", False))
            or bool(getattr(args, "enable_temporal_contrast_loss", False))
        )
        and float(getattr(args, "temporal_contrast_weight", 0.0)) > 0.0
    )


def _compute_temporal_contrast_loss(model_out, batch, args):
    y_reg = model_out["y_reg"]
    if not _temporal_contrast_enabled(args):
        return y_reg.new_zeros(()), {
            "valid_anchor_count": 0,
            "skipped_anchor_count": 0,
            "positive_pair_count": 0,
            "mean_positive_weight": 0.0,
        }
    extras = model_out.get("extras", {}) or {}
    temporal_embedding = extras.get("temporal_embedding")
    if temporal_embedding is None:
        raise RuntimeError("Temporal contrast is enabled but model did not return extras['temporal_embedding']")
    if "temporal_group_id" not in batch or "temporal_pos" not in batch:
        return y_reg.new_zeros(()), {
            "valid_anchor_count": 0,
            "skipped_anchor_count": int(y_reg.shape[0]),
            "positive_pair_count": 0,
            "mean_positive_weight": 0.0,
        }
    return compute_soft_temporal_contrastive_loss(
        temporal_embedding,
        batch["temporal_group_id"],
        batch["temporal_pos"],
        temperature=args.temporal_contrast_temperature,
        decay_tau=args.temporal_decay_tau,
        positive_radius=args.temporal_positive_radius,
        weak_positive_radius=args.temporal_weak_positive_radius,
        min_positive_weight=args.temporal_min_positive_weight,
        temporal_kernel=getattr(args, "temporal_kernel", "legacy_exp"),
        raw_valence=batch.get("raw_valence", None),
        label_gate_beta=getattr(args, "temporal_label_gate_beta", 1.0),
        zero_bridge_weight=getattr(args, "temporal_zero_bridge_weight", 0.25),
    )


def _maybe_clamp_eval(y, clamp_eval: bool):
    return y.clamp(-3.0, 3.0) if clamp_eval else y


def _tensor_stats(values):
    if values is None or values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    v = values.detach().float().reshape(-1)
    return {
        "mean": float(v.mean().item()),
        "std": float(v.std(unbiased=False).item()),
        "min": float(v.min().item()),
        "max": float(v.max().item()),
    }


def _cls7_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
    return float(entropy.detach().mean().item())


def _signed_prediction_tensors(outputs, args):
    y_reg = outputs["y_reg"]
    cls7_logits = outputs["cls7_logits"]
    sign_logits = outputs.get("sign_logits", None)
    centers = get_cls7_centers(device=cls7_logits.device, dtype=cls7_logits.dtype)
    y_reg_eval = _maybe_clamp_eval(y_reg, args.clamp_regression_eval)
    y_cls_expected = cls7_expected_value(cls7_logits, centers=centers)
    y_final = compute_final_prediction(
        y_reg_eval,
        cls7_logits,
        eta=args.final_pred_eta,
        centers=centers,
        sign_logits=sign_logits,
        sign_beta=args.final_pred_sign_beta,
        neutral_positive_gate_threshold=getattr(
            args, "neutral_positive_gate_threshold", None
        ),
    )
    return y_reg_eval, y_cls_expected, y_final

class EarlyStopping:
    def __init__(self, mode="max", patience=3, min_delta=0.0):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.bad_epochs = 0

    def step(self, value):
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs > self.patience

def accuracy_from_lists(labels, preds):
    if len(labels) == 0:
        return 0.0
    correct = 0
    for y, p in zip(labels, preds):
        if int(y) == int(p):
            correct += 1
    return correct / len(labels)

def metrics_from_lists(labels, preds, num_classes):
    if len(labels) == 0:
        return 0.0, 0.0, [0.0 for _ in range(num_classes)], [0.0 for _ in range(num_classes)]

    conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y, p in zip(labels, preds):
        yi = int(y)
        pi = int(p)
        if 0 <= yi < num_classes and 0 <= pi < num_classes:
            conf[yi][pi] += 1

    per_class_precision = []
    per_class_recall = []
    f1s = []
    correct = 0
    total = 0

    for i in range(num_classes):
        tp = conf[i][i]
        fn = sum(conf[i][j] for j in range(num_classes)) - tp
        fp = sum(conf[j][i] for j in range(num_classes)) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_precision.append(prec)
        per_class_recall.append(rec)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

        correct += tp
        total += tp + fn

    acc = correct / total if total > 0 else 0.0
    macro_f1 = sum(f1s) / num_classes if num_classes > 0 else 0.0
    return acc, macro_f1, per_class_precision, per_class_recall

def _rankdata(values):
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0 for _ in range(n)]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks

def _pearsonr(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = 0.0
    denom_x = 0.0
    denom_y = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        dy = y - mean_y
        num += dx * dy
        denom_x += dx * dx
        denom_y += dy * dy
    denom = math.sqrt(denom_x) * math.sqrt(denom_y)
    if denom == 0.0:
        return 0.0
    return num / denom

def regression_metrics(preds, targets):
    if len(preds) == 0:
        return 0.0, 0.0, 0.0, 0.0
    diffs = [p - t for p, t in zip(preds, targets)]
    mse = sum(d * d for d in diffs) / len(diffs)
    rmse = math.sqrt(mse)
    mae = sum(abs(d) for d in diffs) / len(diffs)
    pearson = _pearsonr(preds, targets)
    ranks_p = _rankdata(preds)
    ranks_t = _rankdata(targets)
    spearman = _pearsonr(ranks_p, ranks_t)
    return rmse, mae, pearson, spearman


def score_metrics_from_lists(scores, raw_valences):
    rmse, mae, pearson, spearman = regression_metrics(scores, raw_valences)
    pred_7 = [_valence_to_7class(s) for s in scores]
    gt_7 = [_valence_to_7class(v) for v in raw_valences]
    acc7, macro_f1_7, _, _ = metrics_from_lists(gt_7, pred_7, 7)
    pred_2 = [_valence_to_binary(s) for s in scores]
    gt_2 = [_valence_to_binary(v) for v in raw_valences]
    acc2 = accuracy_from_lists(gt_2, pred_2)
    return {
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman,
        "acc2": acc2,
        "acc7": acc7,
        "macro_f1_7": macro_f1_7,
    }


def summarize_signed_metrics(reg_scores, cls_expected_scores, final_scores, raw_valences, cls7_labels, cls7_preds):
    cls7_acc, cls7_macro_f1, cls7_precision, cls7_recall = metrics_from_lists(cls7_labels, cls7_preds, 7)
    return {
        "reg": score_metrics_from_lists(reg_scores, raw_valences),
        "cls_expected": score_metrics_from_lists(cls_expected_scores, raw_valences),
        "final": score_metrics_from_lists(final_scores, raw_valences),
        "cls7_acc": cls7_acc,
        "cls7_macro_f1": cls7_macro_f1,
        "cls7_precision": cls7_precision,
        "cls7_recall": cls7_recall,
    }

class SlimTensorBoardWriter:
    """Keep TensorBoard focused on LR, losses, head accuracies, and core test metrics."""

    _DIRECT_SCALAR_TAGS = {
        "train/loss": ("loss/train_step_total",),
        "train/loss_intensity": ("loss/train_step_intensity",),
        "train/loss_polarity": ("loss/train_step_polarity",),
        "train/reg_loss": ("loss/train_step_reg",),
        "train/cls7_loss": ("loss/train_step_cls7",),
        "train/sign_loss": ("loss/train_step_sign",),
        "train/zero_sign_margin_loss": ("loss/train_step_zero_sign_margin",),
        "train/weighted_zero_sign_margin_loss": (
            "loss/train_step_weighted_zero_sign_margin",
        ),
        "train/reg_cls_mag_consistency_loss": ("loss/train_step_reg_cls_mag_consistency",),
        "train/weighted_reg_cls_mag_consistency_loss": (
            "loss/train_step_weighted_reg_cls_mag_consistency",
        ),
        "train/reg_cls_mag_consistency_selected_count": (
            "diagnostics/train_step_reg_cls_mag_consistency_selected_count",
        ),
        "train/reg_cls_mag_consistency_coverage": (
            "diagnostics/train_step_reg_cls_mag_consistency_coverage",
        ),
        "train/reg_cls_mag_consistency_mean_abs_gap": (
            "diagnostics/train_step_reg_cls_mag_consistency_mean_abs_gap",
        ),
        "train/oacr_loss": ("loss/train_step_oacr",),
        "train/weighted_oacr_loss": ("loss/train_step_weighted_oacr",),
        "train/temporal_contrast_loss": ("loss/train_step_temporal_contrast",),
        "train/weighted_temporal_contrast_loss": ("loss/train_step_weighted_temporal_contrast",),
        "train/tc_loss": ("loss/train_step_tc",),
        "train/lr_head": ("lr/head",),
        "train/lr_router": ("lr/router",),
        "train/lr_bert_last_layers": ("lr/bert_last_layers",),
        "train/lr_backbone": ("lr/backbone",),
        "epoch/train_loss": ("loss/train_total",),
        "epoch/val_loss": ("loss/val_total",),
        "epoch/test_loss": ("loss/test_total",),
        "epoch/train_acc": ("accuracy/train_cls7_head",),
        "epoch/val_acc": ("accuracy/val_cls7_head",),
        "epoch/test_acc": ("accuracy/test_cls7_head",),
        "epoch/test_mae": ("metrics/test_mae",),
        "epoch/test_final_acc7": ("accuracy/test_final_acc7", "metrics/test_acc7"),
        "epoch/test_final_acc2": ("accuracy/test_final_acc2", "metrics/test_acc2"),
        "Train_Intensity_Loss": ("loss/train_intensity",),
        "Train_Polarity_Loss": ("loss/train_polarity",),
        "Train_Reg_Loss": ("loss/train_reg",),
        "Train_Cls7_Loss": ("loss/train_cls7",),
        "Train_Sign_Loss": ("loss/train_sign",),
        "Train_Zero_Sign_Margin_Loss": ("loss/train_zero_sign_margin",),
        "Train_Weighted_Zero_Sign_Margin_Loss": (
            "loss/train_weighted_zero_sign_margin",
        ),
        "Train_Reg_Cls_Mag_Consistency_Loss": ("loss/train_reg_cls_mag_consistency",),
        "Train_Weighted_Reg_Cls_Mag_Consistency_Loss": (
            "loss/train_weighted_reg_cls_mag_consistency",
        ),
        "epoch/train_reg_cls_mag_consistency_selected_count": (
            "diagnostics/train_reg_cls_mag_consistency_selected_count",
        ),
        "epoch/train_reg_cls_mag_consistency_coverage": (
            "diagnostics/train_reg_cls_mag_consistency_coverage",
        ),
        "epoch/train_reg_cls_mag_consistency_mean_abs_gap": (
            "diagnostics/train_reg_cls_mag_consistency_mean_abs_gap",
        ),
        "Train_OACR_Loss": ("loss/train_oacr",),
        "Train_Weighted_OACR_Loss": ("loss/train_weighted_oacr",),
        "Train_Temporal_Contrast_Loss": ("loss/train_temporal_contrast",),
        "Train_Weighted_Temporal_Contrast_Loss": ("loss/train_weighted_temporal_contrast",),
        "Train_TC_Loss": ("loss/train_tc",),
        "Val_TC_Loss": ("loss/val_tc",),
        "Test_TC_Loss": ("loss/test_tc",),
        "Val_Intensity_Loss": ("loss/val_intensity",),
        "Val_Polarity_Loss": ("loss/val_polarity",),
        "Test_Intensity_Loss": ("loss/test_intensity",),
        "Test_Polarity_Loss": ("loss/test_polarity",),
    }
    _HEAD_ACCURACY_RE = re.compile(r"^epoch/(train|val|test)_(reg|cls_expected|final)_(acc2|acc7)$")
    _EPOCH_SPLIT_RE = re.compile(r"^epoch/(train|val|test)_(.+)$")

    def __init__(self, writer):
        self._writer = writer

    def _map_scalar_tags(self, tag):
        mapped = self._DIRECT_SCALAR_TAGS.get(tag)
        if mapped is not None:
            return mapped
        match = self._HEAD_ACCURACY_RE.match(tag)
        if match:
            split, head, metric = match.groups()
            return (f"accuracy/{split}_{head}_{metric}",)
        match = self._EPOCH_SPLIT_RE.match(tag)
        if not match:
            return ()
        split, rest = match.groups()
        if rest.startswith("loss_"):
            return (f"loss/{split}_{rest[5:]}",)
        if rest.endswith("_loss"):
            return (f"loss/{split}_{rest[:-5]}",)
        if rest.startswith("reg_cls_mag_consistency_"):
            return (f"diagnostics/{split}_{rest}",)
        return ()

    def add_scalar(self, tag, scalar_value, *args, **kwargs):
        for mapped_tag in self._map_scalar_tags(tag):
            self._writer.add_scalar(mapped_tag, scalar_value, *args, **kwargs)

    def add_text(self, tag, text_string, *args, **kwargs):
        if str(tag).startswith("config/"):
            return self._writer.add_text(tag, text_string, *args, **kwargs)
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def flush(self):
        return self._writer.flush()

    def close(self):
        return self._writer.close()

    def __getattr__(self, name):
        return getattr(self._writer, name)

def _make_writer(args):
    if args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("Warning: TensorBoard is not available. Install tensorboard to enable logging.")
        return None
    base = args.log_dir if args.log_dir else os.path.join(args.save_dir, "runs")
    run_name = args.run_name if args.run_name else time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(base, run_name)
    os.makedirs(log_dir, exist_ok=True)
    return SlimTensorBoardWriter(SummaryWriter(log_dir=log_dir))


def _normalize_run_name(run_name: str) -> str:
    if not run_name:
        return run_name
    tokens = [token for token in run_name.split("_") if token]
    cleaned = []
    for token in tokens:
        lowered = token.lower()
        if lowered == "vit":
            continue
        if re.fullmatch(r"a(?:lpha)?[0-9]+(?:\.[0-9]+)?", lowered):
            continue
        cleaned.append(token)
    normalized = "_".join(cleaned)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or run_name

def _cleanup_checkpoint_artifacts(save_dir):
    for name in [
        BEST_ACC7_CHECKPOINT_NAME,
        os.path.splitext(BEST_ACC7_CHECKPOINT_NAME)[0] + ".json",
        BEST_MAE_CHECKPOINT_NAME,
        os.path.splitext(BEST_MAE_CHECKPOINT_NAME)[0] + ".json",
        "best_model.pth",
        "last_model.pth",
        CHECKPOINT_SELECTION_SUMMARY,
    ]:
        path = os.path.join(save_dir, name)
        if os.path.isfile(path):
            os.remove(path)

def _record_checkpoint_snapshot(epoch, val_metrics, test_metrics, path):
    record = {
        "epoch": int(epoch),
        "val_acc7": float(val_metrics["acc7"]),
        "val_mae": float(val_metrics["mae"]),
        "val_acc2": float(val_metrics["acc2"]),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "path": path,
    }
    if test_metrics is not None:
        record.update(
            {
                "test_acc7": float(test_metrics["acc7"]),
                "test_mae": float(test_metrics["mae"]),
                "test_acc2": float(test_metrics["acc2"]),
            }
        )
    return record

def _json_safe(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return tensor.item()
        return tensor.tolist()
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def _write_checkpoint_hparams(path, args, checkpoint_epoch=None):
    if args is None:
        return
    payload = _json_safe(dict(vars(args)))
    payload["hparam_source"] = "train_args"
    payload["use_text_cache"] = _resolve_use_text_cache(args)
    payload["tcif_transition_gate_lr"] = float(
        getattr(args, "tcif_transition_gate_lr", None) or args.lr
    )
    payload["checkpoint_selection_split"] = getattr(args, "checkpoint_selection_split", "test")
    payload["zero_sign_margin_weight"] = getattr(args, "zero_sign_margin_weight", 0.0)
    payload["zero_sign_margin"] = getattr(args, "zero_sign_margin", 0.02)
    payload["zero_sign_margin_temperature"] = getattr(
        args, "zero_sign_margin_temperature", 0.02
    )
    payload["reg_cls_mag_consistency_weight"] = getattr(
        args, "reg_cls_mag_consistency_weight", 0.0
    )
    payload["reg_cls_mag_consistency_boundary_margin"] = getattr(
        args, "reg_cls_mag_consistency_boundary_margin", 0.1
    )
    payload["reg_cls_mag_consistency_smooth_l1_beta"] = getattr(
        args, "reg_cls_mag_consistency_smooth_l1_beta", 0.1
    )
    payload["reg_cls_mag_consistency_mode"] = getattr(
        args,
        "reg_cls_mag_consistency_mode",
        REG_CLS_MAG_CONSISTENCY_MODE,
    )
    if checkpoint_epoch is not None:
        payload["checkpoint_epoch"] = int(checkpoint_epoch)
    json_path = os.path.splitext(path)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _is_better_acc7(candidate, current_best, selection_split="test"):
    if current_best is None:
        return True
    prefix = str(selection_split)
    cand_key = (-float(candidate[f"{prefix}_acc7"]), float(candidate[f"{prefix}_mae"]), int(candidate["epoch"]))
    best_key = (-float(current_best[f"{prefix}_acc7"]), float(current_best[f"{prefix}_mae"]), int(current_best["epoch"]))
    return cand_key < best_key

def _is_better_mae(candidate, current_best, selection_split="test"):
    if current_best is None:
        return True
    prefix = str(selection_split)
    cand_key = (float(candidate[f"{prefix}_mae"]), -float(candidate[f"{prefix}_acc7"]), int(candidate["epoch"]))
    best_key = (float(current_best[f"{prefix}_mae"]), -float(current_best[f"{prefix}_acc7"]), int(current_best["epoch"]))
    return cand_key < best_key

def _update_best_checkpoints(model, save_dir, epoch, val_metrics, test_metrics, best_records, args=None):
    promoted = []
    selection_split = str(getattr(args, "checkpoint_selection_split", "test"))
    acc7_key = f"best_{selection_split}_acc7"
    mae_key = f"best_{selection_split}_mae"

    acc7_path = os.path.join(save_dir, BEST_ACC7_CHECKPOINT_NAME)
    acc7_candidate = _record_checkpoint_snapshot(epoch, val_metrics, test_metrics, acc7_path)
    if _is_better_acc7(acc7_candidate, best_records.get(acc7_key), selection_split):
        torch.save(model.state_dict(), acc7_path)
        _write_checkpoint_hparams(acc7_path, args, checkpoint_epoch=epoch)
        best_records[acc7_key] = acc7_candidate
        promoted.append(("acc7", acc7_candidate))

    if str(getattr(args, "checkpoint_metrics", "dual")) == "dual":
        mae_path = os.path.join(save_dir, BEST_MAE_CHECKPOINT_NAME)
        mae_candidate = _record_checkpoint_snapshot(epoch, val_metrics, test_metrics, mae_path)
        if _is_better_mae(mae_candidate, best_records.get(mae_key), selection_split):
            torch.save(model.state_dict(), mae_path)
            _write_checkpoint_hparams(mae_path, args, checkpoint_epoch=epoch)
            best_records[mae_key] = mae_candidate
            promoted.append(("mae", mae_candidate))

    return promoted

def _write_checkpoint_selection_summary(save_dir, best_records, args=None):
    selection_split = str(getattr(args, "checkpoint_selection_split", "test")) if args is not None else "test"
    checkpoint_metrics = str(getattr(args, "checkpoint_metrics", "dual")) if args is not None else "dual"
    acc7_key = f"best_{selection_split}_acc7"
    mae_key = f"best_{selection_split}_mae"
    summary = {
        "policy": {
            "checkpoint_policy": f"{selection_split}_best_{checkpoint_metrics}",
            "selection_split": selection_split,
            "report_split": "test",
        },
        acc7_key: best_records.get(acc7_key),
        mae_key: best_records.get(mae_key),
    }
    if args is not None:
        summary["head_config"] = {
            "output_head_mode": getattr(args, "output_head_mode", "legacy"),
            "reg_loss_type": getattr(args, "reg_loss_type", "smooth_l1"),
            "cls7_loss_type": getattr(args, "cls7_loss_type", "soft_ce"),
            "cls7_loss_weight": getattr(args, "cls7_loss_weight", 0.5),
            "cls7_class_weight_mode": getattr(args, "cls7_class_weight_mode", "none"),
            "cls7_class_weight_max": getattr(args, "cls7_class_weight_max", 0.0),
            "cls7_soft_tau": getattr(args, "cls7_soft_tau", 0.5),
            "cls7_head_type": getattr(args, "cls7_head_type", "flat"),
            "cumulative_p7_mix": getattr(args, "cumulative_p7_mix", 0.3),
            "cumulative_loss_weight": getattr(args, "cumulative_loss_weight", 0.0),
            "hier_sign_loss_weight": getattr(args, "hier_sign_loss_weight", 0.0),
            "hier_mag_loss_weight": getattr(args, "hier_mag_loss_weight", 0.0),
            "sign_struct_warmup_ratio": getattr(args, "sign_struct_warmup_ratio", 0.0),
            "sign_marginal_loss_type": getattr(args, "sign_marginal_loss_type", "none"),
            "sign_marginal_loss_weight": getattr(args, "sign_marginal_loss_weight", 0.0),
            "sign_marginal_focal_gamma": getattr(args, "sign_marginal_focal_gamma", 1.5),
            "signed_neutral_band_weight": getattr(args, "signed_neutral_band_weight", 0.0),
            "signed_neutral_band_margin": getattr(args, "signed_neutral_band_margin", 1.0 / 6.0),
            "signed_neutral_band_upper": getattr(args, "signed_neutral_band_upper", 0.45),
            "signed_neutral_band_temperature": getattr(args, "signed_neutral_band_temperature", 0.05),
            "signed_neutral_band_eta": getattr(args, "signed_neutral_band_eta", 0.4),
            "zero_sign_margin_weight": getattr(args, "zero_sign_margin_weight", 0.0),
            "zero_sign_margin": getattr(args, "zero_sign_margin", 0.02),
            "zero_sign_margin_temperature": getattr(args, "zero_sign_margin_temperature", 0.02),
            "reg_cls_mag_consistency_weight": getattr(args, "reg_cls_mag_consistency_weight", 0.0),
            "reg_cls_mag_consistency_boundary_margin": getattr(
                args, "reg_cls_mag_consistency_boundary_margin", 0.1
            ),
            "reg_cls_mag_consistency_smooth_l1_beta": getattr(
                args, "reg_cls_mag_consistency_smooth_l1_beta", 0.1
            ),
            "reg_cls_mag_consistency_mode": getattr(
                args, "reg_cls_mag_consistency_mode", REG_CLS_MAG_CONSISTENCY_MODE
            ),
            "final_pred_eta": getattr(args, "final_pred_eta", 0.4),
            "enable_sign_head": getattr(args, "enable_sign_head", False),
            "final_pred_sign_beta": getattr(args, "final_pred_sign_beta", 0.0),
            "neutral_positive_gate_threshold": getattr(
                args, "neutral_positive_gate_threshold", None
            ),
            "sign_aux_weight": getattr(args, "sign_aux_weight", 0.0),
            "sign_aux_loss_type": getattr(args, "sign_aux_loss_type", "bce"),
            "sign_aux_focal_gamma": getattr(args, "sign_aux_focal_gamma", 2.0),
            "sign_aux_low_abs_weight_threshold": getattr(args, "sign_aux_low_abs_weight_threshold", 0.0),
            "sign_aux_low_abs_weight": getattr(args, "sign_aux_low_abs_weight", 1.0),
            "sign_aux_zero_weight": getattr(args, "sign_aux_zero_weight", 1.0),
            "sign_aux_nonzero_weight": getattr(args, "sign_aux_nonzero_weight", 1.0),
            "sign_aux_positive_weight": getattr(args, "sign_aux_positive_weight", 1.0),
            "sign_aux_negative_weight": getattr(args, "sign_aux_negative_weight", 1.0),
            "score_sign_aux_weight": getattr(args, "score_sign_aux_weight", 0.0),
            "score_sign_aux_temperature": getattr(args, "score_sign_aux_temperature", 1.0),
            "score_sign_aux_loss_type": getattr(args, "score_sign_aux_loss_type", "bce"),
            "score_sign_aux_focal_gamma": getattr(args, "score_sign_aux_focal_gamma", 2.0),
            "score_sign_aux_low_abs_weight_threshold": getattr(args, "score_sign_aux_low_abs_weight_threshold", 0.0),
            "score_sign_aux_low_abs_weight": getattr(args, "score_sign_aux_low_abs_weight", 1.0),
            "score_sign_aux_zero_weight": getattr(args, "score_sign_aux_zero_weight", 1.0),
            "score_sign_aux_nonzero_weight": getattr(args, "score_sign_aux_nonzero_weight", 1.0),
            "score_sign_aux_positive_weight": getattr(args, "score_sign_aux_positive_weight", 1.0),
            "score_sign_aux_negative_weight": getattr(args, "score_sign_aux_negative_weight", 1.0),
            "clamp_regression_eval": getattr(args, "clamp_regression_eval", True),
            "enable_oacr": getattr(args, "enable_oacr", False),
            "lambda_oacr": getattr(args, "lambda_oacr", 0.0),
            "oacr_tau": getattr(args, "oacr_tau", 0.15),
            "oacr_sigma_y": getattr(args, "oacr_sigma_y", 0.7),
            "oacr_alpha_c": getattr(args, "oacr_alpha_c", 0.2),
            "contrast_proj_dim": getattr(args, "contrast_proj_dim", 128),
            "enable_temporal_contrast_experts": getattr(args, "enable_temporal_contrast_experts", False),
            "enable_temporal_contrast_loss": getattr(args, "enable_temporal_contrast_loss", False),
            "num_temporal_contrast_experts": getattr(args, "num_temporal_contrast_experts", 0),
            "temporal_embedding_dim": getattr(args, "temporal_embedding_dim", 128),
            "temporal_detach_task2": getattr(args, "temporal_detach_task2", False),
            "temporal_contrast_weight": getattr(args, "temporal_contrast_weight", 0.0),
            "temporal_contrast_temperature": getattr(args, "temporal_contrast_temperature", 0.07),
            "temporal_decay_tau": getattr(args, "temporal_decay_tau", 1.0),
            "temporal_positive_radius": getattr(args, "temporal_positive_radius", 1.0),
            "temporal_weak_positive_radius": getattr(args, "temporal_weak_positive_radius", 4.0),
            "temporal_min_positive_weight": getattr(args, "temporal_min_positive_weight", 0.2),
            "temporal_kernel": getattr(args, "temporal_kernel", "legacy_exp"),
            "temporal_label_gate_beta": getattr(args, "temporal_label_gate_beta", 1.0),
            "temporal_zero_bridge_weight": getattr(args, "temporal_zero_bridge_weight", 0.25),
            "temporal_batch_mode": getattr(args, "temporal_batch_mode", "shuffle"),
            "temporal_batch_window": getattr(args, "temporal_batch_window", 4.0),
            "target_sampler": getattr(args, "target_sampler", "none"),
            "vit_context_lambda_mode": getattr(args, "vit_context_lambda_mode", "static"),
            "vit_context_lambda_init": getattr(args, "vit_context_lambda_init", ""),
            "vit_context_attention_dim": getattr(args, "vit_context_attention_dim", 64),
            "acc7_mapping": "round_half_away_from_zero_clamp_-3_3_plus_3",
            "acc7_source": "legacy_final_score" if getattr(args, "output_head_mode", "legacy") == "legacy" else "signed_reg_cls7_final",
            "use_text_cache": _resolve_use_text_cache(args),
            "checkpoint_selection_split": getattr(args, "checkpoint_selection_split", "test"),
        }
    with open(os.path.join(save_dir, CHECKPOINT_SELECTION_SUMMARY), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

def _now_iso():
    return datetime.now().replace(microsecond=0).isoformat()

def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _infer_results_root_from_save_dir(save_dir: str):
    save_path = Path(save_dir).resolve()
    if save_path.parent.name == "checkpoints":
        return save_path.parent.parent
    return save_path.parent

def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model

def _embedding_cache_kwargs(batch, device, use_text_cache=True):
    kwargs = {}
    for batch_key, model_key in (
        ("cached_vision", "cached_vision"),
        ("cached_text", "cached_text"),
        ("cached_audio", "cached_audio"),
        ("cached_text_mask", "cached_text_mask"),
        ("cached_audio_mask", "cached_audio_mask"),
    ):
        if not use_text_cache and batch_key in {"cached_text", "cached_text_mask"}:
            continue
        value = batch.get(batch_key)
        if value is not None:
            kwargs[model_key] = value.to(device, non_blocking=True)
    return kwargs


def _tcif_context_kwargs(batch, device, use_text_cache=True):
    kwargs = {}
    for batch_key, model_key in (
        ("tcif_context_image", "tcif_context_images"),
        ("tcif_context_input_ids", "tcif_context_input_ids"),
        ("tcif_context_attention_mask", "tcif_context_attention_mask"),
        ("tcif_context_audio_values", "tcif_context_audio_values"),
        (
            "tcif_context_audio_attention_mask",
            "tcif_context_audio_attention_mask",
        ),
        ("tcif_context_cached_vision", "tcif_context_cached_vision"),
        ("tcif_context_cached_text", "tcif_context_cached_text"),
        ("tcif_context_cached_audio", "tcif_context_cached_audio"),
        ("tcif_context_cached_text_mask", "tcif_context_cached_text_mask"),
        ("tcif_context_cached_audio_mask", "tcif_context_cached_audio_mask"),
        ("tcif_context_valid_mask", "tcif_context_valid_mask"),
        ("tcif_context_relative_pos", "tcif_context_relative_pos"),
    ):
        if not use_text_cache and batch_key in {
            "tcif_context_cached_text",
            "tcif_context_cached_text_mask",
        }:
            continue
        value = batch.get(batch_key)
        if value is not None:
            kwargs[model_key] = value.to(device, non_blocking=True)
    return kwargs


def _resolve_use_text_cache(args):
    """Resolve the cache policy and reject stale text features for unfrozen text layers."""
    unfreeze_last_n = int(getattr(args, "unfreeze_bert_last_n_layers", 0))
    explicit = getattr(args, "use_text_cache", None)
    if explicit is None:
        return unfreeze_last_n <= 0
    if unfreeze_last_n > 0 and bool(explicit):
        raise ValueError("--use_text_cache cannot be enabled when BERT layers are unfrozen")
    return bool(explicit)

def _get_bert_encoder_layers(model):
    base_model = _unwrap_model(model)
    bert = getattr(base_model, "bert", None)
    if bert is None:
        return []
    candidate_paths = (
        ("encoder", "layer"),
        ("transformer", "layer"),
    )
    for parent_attr, layer_attr in candidate_paths:
        parent = getattr(bert, parent_attr, None)
        layers = getattr(parent, layer_attr, None) if parent is not None else None
        if layers is not None and len(layers) > 0:
            return list(layers)
    return []

def _get_bert_last_layer_module(model):
    layers = _get_bert_encoder_layers(model)
    if layers:
        return layers[-1]
    return None

def _get_bert_last_layer_modules(model, layer_count):
    layers = _get_bert_encoder_layers(model)
    if not layers:
        return []
    layer_count = max(0, min(int(layer_count), len(layers)))
    if layer_count <= 0:
        return []
    return layers[-layer_count:]

def _set_bert_last_layers_requires_grad(model, layer_count, requires_grad):
    layers = _get_bert_last_layer_modules(model, layer_count)
    if not layers:
        return 0
    param_count = 0
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = requires_grad
            param_count += p.numel()
    return param_count

def _set_bert_last_layer_requires_grad(model, requires_grad):
    return _set_bert_last_layers_requires_grad(model, 1, requires_grad)

def _set_bert_last_layers_train_mode(model, layer_count, training):
    layers = _get_bert_last_layer_modules(model, layer_count)
    for layer in layers:
        layer.train(training)
    return bool(layers)

def _set_bert_last_layer_train_mode(model, training):
    return _set_bert_last_layers_train_mode(model, 1, training)

def _get_vit_context_lambda_state(model):
    base_model = _unwrap_model(model)
    raw_lambdas = getattr(base_model, "vit_context_lambdas", None)
    if raw_lambdas is None:
        return None
    raw_logits = raw_lambdas.detach().cpu().float()
    base_lambda = torch.softmax(raw_logits, dim=0)
    mode = str(getattr(base_model, "vit_context_lambda_mode", "static"))
    state = {
        "mode": mode,
        "raw_logits": [float(x) for x in raw_logits.tolist()],
        "base_lambda": [float(x) for x in base_lambda.tolist()],
    }
    if mode in {"adaptive", "attention"}:
        get_stats = getattr(base_model, "get_vit_context_lambda_stats", None)
        stats = get_stats(reset=False) if get_stats is not None else None
        if stats is not None:
            state["effective_lambda_count"] = int(stats["count"])
            state["effective_lambda_mean"] = stats["mean"]
            state["effective_lambda_std"] = stats["std"]
            state["effective_lambda_min"] = stats["min"]
            state["effective_lambda_max"] = stats["max"]
            for key in [
                "attention_center_prob_mean",
                "attention_center_prob_std",
                "attention_center_prob_min",
                "attention_center_prob_max",
                "attention_matrix_mean",
                "attention_center_entropy_mean",
            ]:
                if key in stats:
                    state[key] = stats[key]
        return state
    state["effective_lambda"] = state["base_lambda"]
    state["effective_lambda_mean"] = state["base_lambda"]
    state["effective_lambda_std"] = [0.0, 0.0, 0.0]
    state["effective_lambda_min"] = state["base_lambda"]
    state["effective_lambda_max"] = state["base_lambda"]
    return state

_MSOE_GROUP_ORDER = ["text_specific", "audio_specific", "vision_specific", "shared", "temporal_contrast"]

def _get_train_msoe_group_dispatch(model):
    base_model = _unwrap_model(model)
    get_summary = getattr(base_model, "get_expert_load_summary", None)
    if get_summary is None:
        return {}
    summary = get_summary(reset=True)
    if not summary:
        return {}
    return summary.get("msoe_group_dispatch") or ExpertLoadTracker.msoe_group_masses(summary, mass_key="dispatch_expert_mass")

def _format_msoe_group_dispatch_shares(group_dispatch):
    parts = []
    for modality_name in ["text", "audio", "vision"]:
        payload = group_dispatch.get(modality_name, {}) if isinstance(group_dispatch, dict) else {}
        groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
        if not groups:
            continue
        group_text = ",".join(
            f"{group_name}:{float((groups.get(group_name, {}) or {}).get('share', 0.0)):.4f}"
            for group_name in _MSOE_GROUP_ORDER
            if group_name in groups
        )
        parts.append(f"{modality_name}={group_text}")
    return " ".join(parts)

def _flatten_msoe_group_dispatch_shares(group_dispatch):
    flat = {}
    for modality_name in ["text", "audio", "vision"]:
        payload = group_dispatch.get(modality_name, {}) if isinstance(group_dispatch, dict) else {}
        groups = payload.get("groups", {}) if isinstance(payload, dict) else {}
        for group_name, stats in groups.items():
            flat[f"{modality_name}_{group_name}"] = float((stats or {}).get("share", 0.0))
    return flat

def _append_lambda_history(save_dir: str, epoch: int, lambda_state):
    if lambda_state is None:
        return
    history_path = _infer_results_root_from_save_dir(save_dir) / LAMBDA_HISTORY_NAME
    history_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "recorded_at": _now_iso(),
    }
    payload.update(lambda_state)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def _write_final_lambda_distribution(save_dir: str, epoch: int, lambda_state):
    if lambda_state is None:
        return
    out_path = _infer_results_root_from_save_dir(save_dir) / "final_lambda_distribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": int(epoch),
        "recorded_at": _now_iso(),
    }
    payload.update(lambda_state)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def _upsert_total_table_rows(args, checkpoint_summary):
    results_root = _infer_results_root_from_save_dir(args.save_dir)
    source_path = results_root / "unified_eval_source.json"
    source_payload = _load_json(source_path, {"generated_at": _now_iso(), "root_dir": str(results_root), "rows": []})
    rows = source_payload.get("rows", [])
    by_id = {str(row.get("row_id")): row for row in rows if isinstance(row, dict) and row.get("row_id")}

    selection_split = str(getattr(args, "checkpoint_selection_split", "test"))
    checkpoint_entries = [
        (f"best_{selection_split}_acc7", BEST_ACC7_CHECKPOINT_NAME, "best_acc7_model"),
        (f"best_{selection_split}_mae", BEST_MAE_CHECKPOINT_NAME, "best_mae_model"),
    ]
    common_hparams = {
        "dataset": args.dataset,
        "dataset_root": args.dataset_root,
        "embedding_cache_root": args.embedding_cache_root,
        "vision_backbone_type": args.vision_backbone_type,
        "vit_backbone_path": args.vit_backbone_path,
        "bert_backbone_path": args.bert_backbone_path,
        "tokenizer_path": args.tokenizer_path,
        "hubert_model_path": args.hubert_model_path,
        "vision_backbone_path": args.vision_backbone_path,
        "num_experts_msoe": args.num_experts_msoe,
        "num_experts_mtoe": args.num_experts_mtoe,
        "num_shared_experts": args.num_shared_experts,
        "num_text_specific_experts": args.num_text_specific_experts,
        "num_audio_specific_experts": args.num_audio_specific_experts,
        "num_vision_specific_experts": args.num_vision_specific_experts,
        "enable_text_shared_experts": args.enable_text_shared_experts,
        "router_lr": args.router_lr,
        "tcif_transition_gate_lr": args.tcif_transition_gate_lr,
        "router_temperature": args.router_temperature,
        "alpha": args.alpha,
        "output_head_mode": args.output_head_mode,
        "intensity_loss_type": args.intensity_loss_type,
        "lambda_polarity": args.lambda_polarity,
        "reg_loss_type": args.reg_loss_type,
        "cls7_loss_type": args.cls7_loss_type,
        "cls7_loss_weight": args.cls7_loss_weight,
        "cls7_class_weight_mode": args.cls7_class_weight_mode,
        "cls7_class_weight_max": args.cls7_class_weight_max,
        "cls7_soft_tau": args.cls7_soft_tau,
        "cls7_head_type": getattr(args, "cls7_head_type", "flat"),
        "cumulative_p7_mix": getattr(args, "cumulative_p7_mix", 0.3),
        "cumulative_loss_weight": getattr(args, "cumulative_loss_weight", 0.0),
        "hier_sign_loss_weight": getattr(args, "hier_sign_loss_weight", 0.0),
        "hier_mag_loss_weight": getattr(args, "hier_mag_loss_weight", 0.0),
        "sign_struct_warmup_ratio": getattr(args, "sign_struct_warmup_ratio", 0.0),
        "sign_marginal_loss_type": getattr(args, "sign_marginal_loss_type", "none"),
        "sign_marginal_loss_weight": getattr(args, "sign_marginal_loss_weight", 0.0),
        "sign_marginal_focal_gamma": getattr(args, "sign_marginal_focal_gamma", 1.5),
        "signed_neutral_band_weight": getattr(args, "signed_neutral_band_weight", 0.0),
        "signed_neutral_band_margin": getattr(args, "signed_neutral_band_margin", 1.0 / 6.0),
        "signed_neutral_band_upper": getattr(args, "signed_neutral_band_upper", 0.45),
        "signed_neutral_band_temperature": getattr(args, "signed_neutral_band_temperature", 0.05),
        "signed_neutral_band_eta": getattr(args, "signed_neutral_band_eta", 0.4),
        "zero_sign_margin_weight": getattr(args, "zero_sign_margin_weight", 0.0),
        "zero_sign_margin": getattr(args, "zero_sign_margin", 0.02),
        "zero_sign_margin_temperature": getattr(args, "zero_sign_margin_temperature", 0.02),
        "reg_cls_mag_consistency_weight": getattr(args, "reg_cls_mag_consistency_weight", 0.0),
        "reg_cls_mag_consistency_boundary_margin": getattr(
            args, "reg_cls_mag_consistency_boundary_margin", 0.1
        ),
        "reg_cls_mag_consistency_smooth_l1_beta": getattr(
            args, "reg_cls_mag_consistency_smooth_l1_beta", 0.1
        ),
        "reg_cls_mag_consistency_mode": getattr(
            args, "reg_cls_mag_consistency_mode", REG_CLS_MAG_CONSISTENCY_MODE
        ),
        "final_pred_eta": args.final_pred_eta,
        "enable_sign_head": args.enable_sign_head,
        "final_pred_sign_beta": args.final_pred_sign_beta,
        "neutral_positive_gate_threshold": getattr(
            args, "neutral_positive_gate_threshold", None
        ),
        "sign_aux_weight": args.sign_aux_weight,
        "sign_aux_loss_type": args.sign_aux_loss_type,
        "sign_aux_focal_gamma": args.sign_aux_focal_gamma,
        "sign_aux_low_abs_weight_threshold": args.sign_aux_low_abs_weight_threshold,
        "sign_aux_low_abs_weight": args.sign_aux_low_abs_weight,
        "sign_aux_zero_weight": args.sign_aux_zero_weight,
        "sign_aux_nonzero_weight": args.sign_aux_nonzero_weight,
        "sign_aux_positive_weight": getattr(args, "sign_aux_positive_weight", 1.0),
        "sign_aux_negative_weight": getattr(args, "sign_aux_negative_weight", 1.0),
        "score_sign_aux_weight": getattr(args, "score_sign_aux_weight", 0.0),
        "score_sign_aux_temperature": getattr(args, "score_sign_aux_temperature", 1.0),
        "score_sign_aux_loss_type": getattr(args, "score_sign_aux_loss_type", "bce"),
        "score_sign_aux_focal_gamma": getattr(args, "score_sign_aux_focal_gamma", 2.0),
        "score_sign_aux_low_abs_weight_threshold": getattr(args, "score_sign_aux_low_abs_weight_threshold", 0.0),
        "score_sign_aux_low_abs_weight": getattr(args, "score_sign_aux_low_abs_weight", 1.0),
        "score_sign_aux_zero_weight": getattr(args, "score_sign_aux_zero_weight", 1.0),
        "score_sign_aux_nonzero_weight": getattr(args, "score_sign_aux_nonzero_weight", 1.0),
        "score_sign_aux_positive_weight": getattr(args, "score_sign_aux_positive_weight", 1.0),
        "score_sign_aux_negative_weight": getattr(args, "score_sign_aux_negative_weight", 1.0),
        "enable_oacr": getattr(args, "enable_oacr", False),
        "lambda_oacr": getattr(args, "lambda_oacr", 0.0),
        "oacr_tau": getattr(args, "oacr_tau", 0.15),
        "oacr_sigma_y": getattr(args, "oacr_sigma_y", 0.7),
        "oacr_alpha_c": getattr(args, "oacr_alpha_c", 0.2),
        "contrast_proj_dim": getattr(args, "contrast_proj_dim", 128),
        "enable_temporal_contrast_experts": getattr(args, "enable_temporal_contrast_experts", False),
        "enable_temporal_contrast_loss": getattr(args, "enable_temporal_contrast_loss", False),
        "num_temporal_contrast_experts": getattr(args, "num_temporal_contrast_experts", 0),
        "temporal_embedding_dim": getattr(args, "temporal_embedding_dim", 128),
        "temporal_detach_task2": getattr(args, "temporal_detach_task2", False),
        "temporal_contrast_weight": getattr(args, "temporal_contrast_weight", 0.0),
        "temporal_contrast_temperature": getattr(args, "temporal_contrast_temperature", 0.07),
        "temporal_decay_tau": getattr(args, "temporal_decay_tau", 1.0),
        "temporal_positive_radius": getattr(args, "temporal_positive_radius", 1.0),
        "temporal_weak_positive_radius": getattr(args, "temporal_weak_positive_radius", 4.0),
        "temporal_min_positive_weight": getattr(args, "temporal_min_positive_weight", 0.2),
        "temporal_kernel": getattr(args, "temporal_kernel", "legacy_exp"),
        "temporal_label_gate_beta": getattr(args, "temporal_label_gate_beta", 1.0),
        "temporal_zero_bridge_weight": getattr(args, "temporal_zero_bridge_weight", 0.25),
        "temporal_batch_mode": getattr(args, "temporal_batch_mode", "shuffle"),
        "temporal_batch_window": getattr(args, "temporal_batch_window", 4.0),
        "target_sampler": getattr(args, "target_sampler", "none"),
        "clamp_regression_eval": args.clamp_regression_eval,
        "acc7_mapping": "round_half_away_from_zero_clamp_-3_3_plus_3",
        "acc7_source": "legacy_final_score" if args.output_head_mode == "legacy" else "signed_reg_cls7_final",
        "batch_size": args.batch_size,
        "num_frames": args.num_frames,
        "dropout": args.dropout,
        "attention_dropout": args.attention_dropout,
        "epochs": args.epochs,
        "lr": args.lr,
        "scheduler": args.scheduler,
        "plateau_patience": args.plateau_patience,
        "plateau_factor": args.plateau_factor,
        "min_lr": args.min_lr,
        "lr_min_factor": args.lr_min_factor,
        "lr_warmup_epochs": args.lr_warmup_epochs,
        "lr_warmup_start_factor": args.lr_warmup_start_factor,
        "backbone_lr_ratio": args.backbone_lr_ratio,
        "unfreeze_bert_last_layer": args.unfreeze_bert_last_layer,
        "unfreeze_bert_last_n_layers": args.unfreeze_bert_last_n_layers,
        "use_text_cache": _resolve_use_text_cache(args),
        "checkpoint_selection_split": getattr(args, "checkpoint_selection_split", "test"),
        "bert_last_layer_lr_ratio": args.bert_last_layer_lr_ratio,
        "vit_context_ratio": args.vit_context_ratio,
        "vit_context_lambda_mode": args.vit_context_lambda_mode,
        "vit_context_lambda_init": args.vit_context_lambda_init,
        "vit_context_attention_dim": args.vit_context_attention_dim,
        "seed": args.seed,
        "freeze_backbone_epochs": args.freeze_backbone_epochs,
        "frame_policy": args.frame_policy,
        "local_files_only": args.local_files_only,
        "checkpoint_policy": f"{selection_split}_best_{getattr(args, 'checkpoint_metrics', 'dual')}",
    }

    exp_dir = Path(args.save_dir).resolve().parent
    for summary_key, model_name, model_stem in checkpoint_entries:
        record = checkpoint_summary.get(summary_key)
        if not record:
            continue
        test_mae = record.get("test_mae")
        test_acc7 = record.get("test_acc7")
        test_acc2 = record.get("test_acc2")
        row_id = f"{args.run_name}/{model_stem}" if args.run_name else f"{exp_dir.name}/{model_stem}"
        row = {
            "row_id": row_id,
            "results_root": str(results_root),
            "output_dir": str(exp_dir),
            "model_path": str(Path(record["path"]).resolve()),
            "relative_model_path": f"{exp_dir.name}/checkpoints/{model_name}",
            "model_name": model_name,
            "model_stem": model_stem,
            "status": "ok",
            "stage": "test_eval",
            "run_name": args.run_name or exp_dir.name,
            "combo_id": args.run_name or exp_dir.name,
            "hparams": common_hparams,
            "val_mae": float(record["val_mae"]),
            "test_mae": float(test_mae) if test_mae is not None else None,
            "val_acc7": float(record["val_acc7"]),
            "test_acc7": float(test_acc7) if test_acc7 is not None else None,
            "val_acc2": float(record["val_acc2"]),
            "test_acc2": float(test_acc2) if test_acc2 is not None else None,
            "selected_epoch": int(record["epoch"]),
            "checkpoint_policy": f"{selection_split}_best_{getattr(args, 'checkpoint_metrics', 'dual')}",
            "updated_at": _now_iso(),
        }
        by_id[row_id] = row

    source_payload["generated_at"] = _now_iso()
    source_payload["root_dir"] = str(results_root)
    source_payload["rows"] = sorted(
        by_id.values(),
        key=lambda row: (str(row.get("results_root", "")), str(row.get("relative_model_path", "")), str(row.get("row_id", ""))),
    )
    _write_json(source_path, source_payload)

    export_script = results_root / "build_unified_eval_table.py"
    dashboard_script = results_root / "unified_eval_dashboard.py"
    table_json = results_root / "unified_eval_table.json"
    dashboard_html = results_root / "unified_eval_dashboard.html"
    if export_script.exists():
        subprocess.run(
            [sys.executable, str(export_script), "--mode", "export", "--root-dir", str(results_root)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    if dashboard_script.exists() and table_json.exists():
        subprocess.run(
            [sys.executable, str(dashboard_script), "--input-json", str(table_json), "--output-html", str(dashboard_html)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

def _set_requires_grad(module, requires_grad):
    for p in module.parameters():
        p.requires_grad = requires_grad

def _build_lr_scheduler(optimizer, args):
    if args.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    if args.scheduler == "cosine":
        total_epochs = max(1, int(args.epochs))
        warmup_epochs = max(0, min(int(args.lr_warmup_epochs), total_epochs))
        warmup_start = float(args.lr_warmup_start_factor)
        if not 0.0 <= warmup_start <= 1.0:
            raise ValueError("--lr_warmup_start_factor must be in [0, 1]")
        min_lr = max(0.0, float(args.min_lr))
        lr_min_factor = float(getattr(args, "lr_min_factor", -1.0))
        if lr_min_factor >= 0.0 and not 0.0 <= lr_min_factor <= 1.0:
            raise ValueError("--lr_min_factor must be in [0, 1], or negative to disable")
        base_lrs = [float(group["lr"]) for group in optimizer.param_groups]

        def make_lambda(base_lr):
            if base_lr <= 0.0:
                min_factor = 0.0
            elif lr_min_factor >= 0.0:
                min_factor = lr_min_factor
            else:
                min_factor = min(1.0, min_lr / base_lr)

            def lr_lambda(epoch_index):
                epoch_index = max(0, int(epoch_index))
                if warmup_epochs > 0 and epoch_index < warmup_epochs:
                    progress = float(epoch_index + 1) / float(warmup_epochs)
                    return warmup_start + (1.0 - warmup_start) * progress
                if warmup_epochs > 0:
                    decay_index = max(0, epoch_index - warmup_epochs + 1)
                    decay_span = max(1, total_epochs - warmup_epochs)
                else:
                    decay_index = epoch_index
                    decay_span = max(1, total_epochs - 1)
                progress = min(1.0, float(decay_index) / float(decay_span))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_factor + (1.0 - min_factor) * cosine

            return lr_lambda

        return optim.lr_scheduler.LambdaLR(optimizer, [make_lambda(lr) for lr in base_lrs])
    return None

class _GatingStatsCollector:
    def __init__(self):
        self.entropies = []
        self.variances = []

    def clear(self):
        self.entropies = []
        self.variances = []

    def add(self, entropy, variance):
        self.entropies.append(entropy)
        self.variances.append(variance)

    def mean_entropy(self):
        if not self.entropies:
            return None
        return float(sum(self.entropies) / len(self.entropies))

    def mean_variance(self):
        if not self.variances:
            return None
        return float(sum(self.variances) / len(self.variances))

def _compute_gating_stats(module, x):
    if getattr(module, "normalize", False):
        x = torch.nn.functional.normalize(x, dim=2)
        phi = module.scale * torch.nn.functional.normalize(module.phi, dim=0)
    else:
        phi = module.phi
    logits = torch.einsum("bmd,dnp->bmnp", x, phi)
    c = softmax(logits, dim=(2, 3))
    p_expert = c.sum(dim=3)
    entropy = -(p_expert * torch.log(p_expert + 1e-8)).sum(dim=2).mean()
    load = p_expert.mean(dim=1)
    variance = load.var(dim=1, unbiased=False).mean()
    return entropy.item(), variance.item()

def _register_gating_hooks(model, collector):
    handles = []

    def hook(module, inputs, outputs):
        try:
            x = inputs[0]
        except Exception:
            return
        with torch.no_grad():
            entropy, variance = _compute_gating_stats(module, x)
        collector.add(entropy, variance)

    for m in model.modules():
        if isinstance(m, (SoftMoELayerWrapper, SoftMoELayerWrapperMET)):
            handles.append(m.register_forward_hook(hook))
    return handles

def _compute_balanced_weights_from_counts(counts):
    total = sum(counts)
    n = len(counts)
    if total <= 0 or n <= 0:
        return [1.0 for _ in range(n)]
    weights = []
    for c in counts:
        if c <= 0:
            weights.append(0.0)
        else:
            weights.append(total / (n * c))
    nonzero = [w for w in weights if w > 0]
    if nonzero:
        m = sum(nonzero) / len(nonzero)
        if m > 0:
            weights = [w / m for w in weights]
    return weights

def _compute_polarity_class_weights(train_dataset, num_classes):
    counts = [0 for _ in range(num_classes)]
    if isinstance(train_dataset, (CMUMOSEIProcessDataset, CMUMOSIProcessDataset)):
        for sample_id in train_dataset.ids:
            val = float(train_dataset.labels[sample_id]["val"])
            y = _valence_to_polarity(val)
            if 0 <= y < num_classes:
                counts[y] += 1
    elif isinstance(train_dataset, MultimodalEmotionDataset):
        for row in train_dataset.data:
            raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
            if raw_valence is None or raw_valence == "":
                continue
            y = _valence_to_polarity(float(raw_valence))
            if 0 <= y < num_classes:
                counts[y] += 1
    return _compute_balanced_weights_from_counts(counts)

def _compute_cls7_class_weights(train_dataset, mode: str, max_weight: float = 0.0):
    mode = str(mode or "none")
    if mode == "none":
        return None, [0 for _ in range(7)]
    values = []
    if isinstance(train_dataset, (CMUMOSEIProcessDataset, CMUMOSIProcessDataset)):
        for sample_id in train_dataset.ids:
            values.append(float(train_dataset.labels[sample_id]["val"]))
    elif isinstance(train_dataset, MultimodalEmotionDataset):
        for row in train_dataset.data:
            raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
            if raw_valence is not None and raw_valence != "":
                values.append(float(raw_valence))
    else:
        try:
            for i in range(len(train_dataset)):
                sample = train_dataset[i]
                raw_valence = sample.get("raw_valence", sample.get("valence", sample.get("val", None)))
                if raw_valence is not None:
                    values.append(float(raw_valence))
        except Exception:
            values = []
    counts = [0 for _ in range(7)]
    if values:
        labels = continuous_to_cls7_hard(torch.tensor(values, dtype=torch.float32))
        for label in labels.tolist():
            if 0 <= int(label) < 7:
                counts[int(label)] += 1
    weights = _compute_balanced_weights_from_counts(counts)
    if mode == "sqrt_balanced":
        weights = [math.sqrt(max(float(weight), 0.0)) for weight in weights]
    elif mode != "balanced":
        raise ValueError(f"Unsupported cls7_class_weight_mode: {mode}")
    max_weight = float(max_weight)
    if max_weight > 0:
        weights = [min(float(weight), max_weight) if weight > 0 else 0.0 for weight in weights]
    nonzero = [weight for weight in weights if weight > 0]
    if nonzero:
        mean_weight = sum(nonzero) / len(nonzero)
        if mean_weight > 0:
            weights = [float(weight) / mean_weight if weight > 0 else 0.0 for weight in weights]
    return torch.tensor(weights, dtype=torch.float32), counts

def _collect_train_polarity_labels(train_dataset):
    labels = []
    if isinstance(train_dataset, (CMUMOSEIProcessDataset, CMUMOSIProcessDataset)):
        for sample_id in train_dataset.ids:
            val = float(train_dataset.labels[sample_id]["val"])
            labels.append(_valence_to_polarity(val))
    elif isinstance(train_dataset, MultimodalEmotionDataset):
        for row in train_dataset.data:
            raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
            if raw_valence is None or raw_valence == "":
                labels.append(-1)
                continue
            labels.append(_valence_to_polarity(float(raw_valence)))
    else:
        try:
            for i in range(len(train_dataset)):
                sample = train_dataset[i]
                y = int(sample.get("polarity", -1))
                labels.append(y)
        except Exception:
            pass
    if not labels:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(labels, dtype=torch.long)


def _collect_raw_valence_summary(dataset):
    values = []
    if isinstance(dataset, (CMUMOSEIProcessDataset, CMUMOSIProcessDataset)):
        for sample_id in dataset.ids:
            values.append(float(dataset.labels[sample_id]["val"]))
    elif isinstance(dataset, MultimodalEmotionDataset):
        for row in dataset.data:
            raw_valence = row.get("raw_valence", row.get("valence", row.get("val", None)))
            if raw_valence is not None and raw_valence != "":
                values.append(float(raw_valence))
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean_abs": sum(abs(v) for v in values) / len(values),
        "num_samples": len(values),
    }

def _is_tcif_transition_gate_parameter(name):
    return name.startswith(
        (
            "tcif_regression.transition_gate.",
            "tcif_ordinal.transition_gate.",
        )
    )


def _build_optimizer_param_groups(model, contrast_head, args):
    """Partition parameters exactly once and isolate TCIF transition gates."""
    bert_last_layer_params = []
    bert_last_layer_param_ids = set()
    if args.unfreeze_bert_last_n_layers > 0:
        bert_last_layers = _get_bert_last_layer_modules(
            model, args.unfreeze_bert_last_n_layers
        )
        if not bert_last_layers:
            raise ValueError(
                "--unfreeze_bert_last_n_layers was set, but the text backbone "
                "exposes no supported encoder layers"
            )
        if len(bert_last_layers) != args.unfreeze_bert_last_n_layers:
            raise ValueError(
                f"Requested {args.unfreeze_bert_last_n_layers} BERT layers, "
                f"but only found {len(bert_last_layers)}"
            )
        bert_last_layer_params = [
            p for layer in bert_last_layers for p in layer.parameters()
        ]
        bert_last_layer_param_ids = {id(p) for p in bert_last_layer_params}

    buckets = {
        "head": [],
        "tcif_transition_gate": [],
        "router": [],
        "bert_last_layers": bert_last_layer_params,
        "backbone": [],
    }
    parameter_names = {name: p for name, p in model.named_parameters()}
    for name, parameter in parameter_names.items():
        if id(parameter) in bert_last_layer_param_ids:
            continue
        if _is_tcif_transition_gate_parameter(name):
            buckets["tcif_transition_gate"].append(parameter)
        elif (
            ".phi" in name
            or ".scale" in name
            or name.startswith("fusion_task2_text_router.")
        ):
            buckets["router"].append(parameter)
        elif name.startswith(
            ("vit.", "bert.", "hubert.", "resnet.")
        ):
            buckets["backbone"].append(parameter)
        else:
            buckets["head"].append(parameter)
    if contrast_head is not None:
        buckets["head"].extend(list(contrast_head.parameters()))

    learning_rates = {
        "head": args.lr,
        "tcif_transition_gate": args.tcif_transition_gate_lr,
        "router": args.router_lr,
        "bert_last_layers": args.lr * args.bert_last_layer_lr_ratio,
        "backbone": args.lr * args.backbone_lr_ratio,
    }
    param_groups = [
        {"name": name, "params": params, "lr": learning_rates[name]}
        for name, params in buckets.items()
        if params
    ]

    expected_ids = {id(parameter) for parameter in model.parameters()}
    if contrast_head is not None:
        expected_ids.update(id(parameter) for parameter in contrast_head.parameters())
    grouped_ids = [id(parameter) for group in param_groups for parameter in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise RuntimeError("optimizer parameter groups overlap")
    if set(grouped_ids) != expected_ids:
        raise RuntimeError("optimizer parameter groups do not cover the parameter set")
    return param_groups


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(args, "run_name", ""):
        normalized_run_name = _normalize_run_name(args.run_name)
        if normalized_run_name != args.run_name:
            print(f"Normalized run_name: {args.run_name} -> {normalized_run_name}")
        args.run_name = normalized_run_name
    print(f"Using device: {device}")
    _set_seed(args.seed, deterministic=args.deterministic)
    print(f"Using seed: {args.seed} deterministic={args.deterministic}")
    _cleanup_checkpoint_artifacts(args.save_dir)

    writer = _make_writer(args)

    transform_steps = []
    if args.vision_backbone_type == "vit":
        transform_steps.append(transforms.Resize((224, 224)))
    transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform = transforms.Compose(transform_steps)

    default_tokenizer_path = "/path/to/models/AI-ModelScope_roberta-base"
    args.bert_model_path = args.bert_backbone_path
    args.vit_model_path = args.vit_backbone_path
    if args.tokenizer_path == default_tokenizer_path and args.bert_model_path != default_tokenizer_path:
        args.tokenizer_path = args.bert_model_path

    if args.dataset == "cmumosei":
        train_dataset = CMUMOSEIProcessDataset(
            root=args.dataset_root,
            split="train",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
            embedding_cache_root=args.embedding_cache_root,
            temporal_context_radius=(args.tcif_context_radius if args.enable_tcif else 0),
            temporal_context_mode=args.tcif_context_mode,
            temporal_context_seed=args.seed,
        )
        val_dataset = CMUMOSEIProcessDataset(
            root=args.dataset_root,
            split="val",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
            embedding_cache_root=args.embedding_cache_root,
            temporal_context_radius=(args.tcif_context_radius if args.enable_tcif else 0),
            temporal_context_mode=args.tcif_context_mode,
            temporal_context_seed=args.seed,
        )
        test_dataset = CMUMOSEIProcessDataset(
            root=args.dataset_root,
            split="test",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
            embedding_cache_root=args.embedding_cache_root,
            temporal_context_radius=(args.tcif_context_radius if args.enable_tcif else 0),
            temporal_context_mode=args.tcif_context_mode,
            temporal_context_seed=args.seed,
        )
    elif args.dataset == "cmumosi":
        train_dataset = CMUMOSIProcessDataset(
            root=args.dataset_root,
            split="train",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
        )
        val_dataset = CMUMOSIProcessDataset(
            root=args.dataset_root,
            split="val",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
        )
        test_dataset = CMUMOSIProcessDataset(
            root=args.dataset_root,
            split="test",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            frame_policy=args.frame_policy,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
            vit_context_ratio=args.vit_context_ratio,
        )
    else:
        train_dataset = MultimodalEmotionDataset(
            csv_file=args.train_csv,
            split="train",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
        )
        val_dataset = MultimodalEmotionDataset(
            csv_file=args.val_csv,
            split="val",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
        )
        if not args.test_csv:
            raise RuntimeError("CSV dataset mode now requires --test_csv so checkpoint selection can run on the test split each epoch.")
        test_dataset = MultimodalEmotionDataset(
            csv_file=args.test_csv,
            split="test",
            transform=transform,
            tokenizer_path=args.tokenizer_path,
            local_files_only=args.local_files_only,
            max_length=args.max_length,
            audio_sample_rate=args.audio_sample_rate,
            audio_max_seconds=args.audio_max_seconds,
            audio_frame_ms=args.audio_frame_ms,
            audio_hop_ms=args.audio_hop_ms,
            audio_denoise=not args.disable_audio_denoise,
            num_frames=args.num_frames,
        )

    print(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}, Test set: {len(test_dataset)}")
    if args.enable_tcif:
        for split_name, split_dataset in (
            ("train", train_dataset),
            ("val", val_dataset),
            ("test", test_dataset),
        ):
            context_index = getattr(split_dataset, "temporal_context_index", {})
            valid_slots = sum(
                int(row["valid"])
                for rows in context_index.values()
                for row in rows
            )
            total_slots = sum(len(rows) for rows in context_index.values())
            samples_with_context = sum(
                any(bool(row["valid"]) for row in rows)
                for rows in context_index.values()
            )
            print(
                f"TCIF {split_name}: mode={args.tcif_context_mode} "
                f"radius={args.tcif_context_radius} "
                f"valid_slots={valid_slots}/{total_slots} "
                f"samples_with_context={samples_with_context}/{len(split_dataset)}"
            )
    raw_summary = _collect_raw_valence_summary(train_dataset)
    if raw_summary is not None:
        print(
            "Train raw_valence summary: "
            f"min={raw_summary['min']:.4f} max={raw_summary['max']:.4f} "
            f"mean_abs={raw_summary['mean_abs']:.4f} n={raw_summary['num_samples']}"
        )
        if -1.05 <= raw_summary["min"] and raw_summary["max"] <= 1.05:
            print("Warning: raw_valence appears normalized to about [-1, 1]; signed_reg_cls7 assumes MOSEI/MOSI-style [-3, 3] labels.")
    if len(train_dataset) == 0:
        raise RuntimeError(f"Empty train dataset. Check --train_csv: {args.train_csv}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"Empty val dataset. Check --val_csv: {args.val_csv}")
    if len(test_dataset) == 0:
        raise RuntimeError("Empty test dataset. Check dataset root or --test_csv.")

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(args.seed)
    val_loader_generator = torch.Generator()
    val_loader_generator.manual_seed(args.seed + 1)
    train_loader = _make_train_loader(train_dataset, args, train_loader_generator)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker,
        generator=val_loader_generator,
    )
    test_loader_generator = torch.Generator()
    test_loader_generator.manual_seed(args.seed + 2)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_seed_worker,
        generator=test_loader_generator,
    )

    num_polarity_classes = 2

    model = EmotionM4OE(
        num_classes=num_polarity_classes,
        num_aux_classes=num_polarity_classes,
        embed_dim=768,
        depth_msoe=2,
        depth_mtoe=1,
        num_experts_msoe=args.num_experts_msoe,
        num_experts_mtoe=args.num_experts_mtoe,
        num_shared_experts=args.num_shared_experts,
        num_text_specific_experts=args.num_text_specific_experts,
        num_audio_specific_experts=args.num_audio_specific_experts,
        num_vision_specific_experts=args.num_vision_specific_experts,
        num_temporal_contrast_experts=args.num_temporal_contrast_experts,
        enable_text_shared_experts=args.enable_text_shared_experts,
        enable_temporal_contrast_experts=args.enable_temporal_contrast_experts,
        enable_temporal_contrast_loss=args.enable_temporal_contrast_loss,
        temporal_embedding_dim=args.temporal_embedding_dim,
        temporal_detach_task2=args.temporal_detach_task2,
        vision_backbone_type=args.vision_backbone_type,
        vision_backbone_path=args.vision_backbone_path,
        vit_model_path=args.vit_model_path,
        bert_model_path=args.bert_model_path,
        hubert_model_path=args.hubert_model_path,
        local_files_only=args.local_files_only,
        dropout=args.dropout,
        attention_dropout=args.attention_dropout,
        router_temperature=args.router_temperature,
        vit_context_ratio=args.vit_context_ratio,
        vit_context_lambda_mode=args.vit_context_lambda_mode,
        vit_context_lambda_init=args.vit_context_lambda_init,
        vit_context_attention_dim=args.vit_context_attention_dim,
        output_head_mode=args.output_head_mode,
        final_pred_eta=args.final_pred_eta,
        enable_sign_head=args.enable_sign_head,
        final_pred_sign_beta=args.final_pred_sign_beta,
        neutral_positive_gate_threshold=args.neutral_positive_gate_threshold,
        cls7_head_type=args.cls7_head_type,
        cumulative_p7_mix=args.cumulative_p7_mix,
        enable_tcif=args.enable_tcif,
        tcif_ablation=getattr(args, "tcif_ablation", "full"),
        tcif_latent_dim=args.tcif_latent_dim,
        tcif_output_mode=args.tcif_output_mode,
        tcif_context_temperature=args.tcif_context_temperature,
        tcif_enable_transition_gate=args.tcif_enable_transition_gate,
        tcif_transition_gate_hidden_dim=args.tcif_transition_gate_hidden_dim,
        tcif_transition_gate_init_bias=args.tcif_transition_gate_init_bias,
    ).to(device)
    contrast_head = None
    if _oacr_enabled(args):
        if args.output_head_mode != "signed_reg_cls7":
            raise ValueError("OACR requires --output_head_mode signed_reg_cls7")
        contrast_head = ContrastiveProjectionHead(model.head_input_dim, args.contrast_proj_dim).to(device)

    if args.enable_tcif:
        # named_parameters deduplicates the shared filter's tensors.
        args.tcif_parameter_counts = {
            "total": sum(p.numel() for n, p in model.named_parameters() if n.startswith("tcif_")),
            "trainable": sum(p.numel() for n, p in model.named_parameters() if n.startswith("tcif_") and p.requires_grad),
        }
        print(f"TCIF ablation={getattr(args, 'tcif_ablation', 'full')} parameters={args.tcif_parameter_counts}")
    gating_collector = _GatingStatsCollector()
    gating_handles = _register_gating_hooks(model, gating_collector)

    param_groups = _build_optimizer_param_groups(model, contrast_head, args)
    print(
        "Optimizer groups: "
        + ", ".join(
            f"{group['name']}[params={sum(p.numel() for p in group['params'])},lr={group['lr']:.8g}]"
            for group in param_groups
        )
    )
    optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = _build_lr_scheduler(optimizer, args)

    train_labels_t = _collect_train_polarity_labels(train_dataset)
    polarity_weights = compute_inverse_freq_weights(train_labels_t, num_classes=num_polarity_classes)
    polarity_weights_t = polarity_weights.to(device)
    cls7_class_weights, cls7_class_counts = _compute_cls7_class_weights(
        train_dataset,
        args.cls7_class_weight_mode,
        args.cls7_class_weight_max,
    )
    cls7_class_weights_t = cls7_class_weights.to(device) if cls7_class_weights is not None else None
    args._cls7_class_weights_t = cls7_class_weights_t
    if cls7_class_weights_t is not None:
        print(
            "Cls7 class weights "
            f"mode={args.cls7_class_weight_mode} max={args.cls7_class_weight_max} "
            f"counts={cls7_class_counts} weights={[round(float(x), 4) for x in cls7_class_weights.tolist()]}"
        )

    if args.loss_type == "weighted_ce":
        criterion_polarity = nn.CrossEntropyLoss(weight=polarity_weights_t, ignore_index=-1)
    else:
        criterion_polarity = FocalLoss(alpha=polarity_weights_t, gamma=args.focal_gamma, ignore_index=-1)

    selection_split = str(getattr(args, "checkpoint_selection_split", "test"))
    best_checkpoint_records = {
        f"best_{selection_split}_acc7": None,
        f"best_{selection_split}_mae": None,
    }
    global_step = 0
    if args.early_stop_metric in ["val_loss", "val_mae"]:
        early_stopper = EarlyStopping(mode="min", patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)
    else:
        early_stopper = EarlyStopping(mode="max", patience=args.early_stop_patience, min_delta=args.early_stop_min_delta)

    if writer is not None:
        writer.add_text("config/dataset", args.dataset, 0)
        writer.add_text("config/save_dir", args.save_dir, 0)
        writer.add_text("config/embedding_cache_root", args.embedding_cache_root or "", 0)
        writer.add_text("config/vit_model_path", args.vit_model_path, 0)
        writer.add_text("config/vision_backbone_type", args.vision_backbone_type, 0)
        writer.add_text("config/vision_backbone_path", args.vision_backbone_path, 0)
        writer.add_text("config/bert_model_path", args.bert_model_path, 0)
        writer.add_text("config/hubert_model_path", args.hubert_model_path, 0)
        writer.add_text("config/tokenizer_path", args.tokenizer_path, 0)
        writer.add_text("config/loss_type", args.loss_type, 0)
        writer.add_text("config/output_head_mode", args.output_head_mode, 0)
        writer.add_text("config/intensity_loss_type", args.intensity_loss_type, 0)
        writer.add_text("config/lambda_polarity", str(args.lambda_polarity), 0)
        writer.add_text("config/reg_loss_type", args.reg_loss_type, 0)
        writer.add_text("config/cls7_loss_type", args.cls7_loss_type, 0)
        writer.add_text("config/cls7_loss_weight", str(args.cls7_loss_weight), 0)
        writer.add_text("config/cls7_class_weight_mode", str(args.cls7_class_weight_mode), 0)
        writer.add_text("config/cls7_class_weight_max", str(args.cls7_class_weight_max), 0)
        writer.add_text("config/cls7_low_abs_sample_weight_threshold", str(args.cls7_low_abs_sample_weight_threshold), 0)
        writer.add_text("config/cls7_low_abs_sample_weight", str(args.cls7_low_abs_sample_weight), 0)
        writer.add_text("config/cls7_soft_tau", str(args.cls7_soft_tau), 0)
        writer.add_text("config/final_pred_eta", str(args.final_pred_eta), 0)
        writer.add_text("config/enable_sign_head", str(args.enable_sign_head), 0)
        writer.add_text("config/final_pred_sign_beta", str(args.final_pred_sign_beta), 0)
        writer.add_text(
            "config/neutral_positive_gate_threshold",
            str(args.neutral_positive_gate_threshold),
            0,
        )
        writer.add_text("config/zero_sign_margin_weight", str(args.zero_sign_margin_weight), 0)
        writer.add_text("config/zero_sign_margin", str(args.zero_sign_margin), 0)
        writer.add_text(
            "config/zero_sign_margin_temperature",
            str(args.zero_sign_margin_temperature),
            0,
        )
        writer.add_text(
            "config/reg_cls_mag_consistency_weight",
            str(args.reg_cls_mag_consistency_weight),
            0,
        )
        writer.add_text(
            "config/reg_cls_mag_consistency_boundary_margin",
            str(args.reg_cls_mag_consistency_boundary_margin),
            0,
        )
        writer.add_text(
            "config/reg_cls_mag_consistency_smooth_l1_beta",
            str(args.reg_cls_mag_consistency_smooth_l1_beta),
            0,
        )
        writer.add_text(
            "config/reg_cls_mag_consistency_mode",
            str(args.reg_cls_mag_consistency_mode),
            0,
        )
        writer.add_text("config/sign_aux_weight", str(args.sign_aux_weight), 0)
        writer.add_text("config/sign_aux_loss_type", str(args.sign_aux_loss_type), 0)
        writer.add_text("config/sign_aux_focal_gamma", str(args.sign_aux_focal_gamma), 0)
        writer.add_text("config/sign_aux_low_abs_weight_threshold", str(args.sign_aux_low_abs_weight_threshold), 0)
        writer.add_text("config/sign_aux_low_abs_weight", str(args.sign_aux_low_abs_weight), 0)
        writer.add_text("config/sign_aux_zero_weight", str(args.sign_aux_zero_weight), 0)
        writer.add_text("config/sign_aux_nonzero_weight", str(args.sign_aux_nonzero_weight), 0)
        writer.add_text("config/sign_aux_positive_weight", str(args.sign_aux_positive_weight), 0)
        writer.add_text("config/sign_aux_negative_weight", str(args.sign_aux_negative_weight), 0)
        writer.add_text("config/enable_oacr", str(args.enable_oacr), 0)
        writer.add_text("config/lambda_oacr", str(args.lambda_oacr), 0)
        writer.add_text("config/oacr_tau", str(args.oacr_tau), 0)
        writer.add_text("config/oacr_sigma_y", str(args.oacr_sigma_y), 0)
        writer.add_text("config/oacr_alpha_c", str(args.oacr_alpha_c), 0)
        writer.add_text("config/contrast_proj_dim", str(args.contrast_proj_dim), 0)
        writer.add_text("config/clamp_regression_eval", str(args.clamp_regression_eval), 0)
        writer.add_text("config/focal_gamma", str(args.focal_gamma), 0)
        writer.add_text("config/dropout", str(args.dropout), 0)
        writer.add_text("config/attention_dropout", str(args.attention_dropout), 0)
        writer.add_text("config/lr", str(args.lr), 0)
        writer.add_text("config/scheduler", str(args.scheduler), 0)
        writer.add_text("config/plateau_patience", str(args.plateau_patience), 0)
        writer.add_text("config/plateau_factor", str(args.plateau_factor), 0)
        writer.add_text("config/min_lr", str(args.min_lr), 0)
        writer.add_text("config/lr_min_factor", str(args.lr_min_factor), 0)
        writer.add_text("config/lr_warmup_epochs", str(args.lr_warmup_epochs), 0)
        writer.add_text("config/lr_warmup_start_factor", str(args.lr_warmup_start_factor), 0)
        writer.add_text("config/router_temperature", str(args.router_temperature), 0)
        writer.add_text("config/router_lr", str(args.router_lr), 0)
        writer.add_text(
            "config/tcif_transition_gate_lr",
            str(args.tcif_transition_gate_lr),
            0,
        )
        writer.add_text("config/backbone_lr_ratio", str(args.backbone_lr_ratio), 0)
        writer.add_text("config/unfreeze_bert_last_layer", str(args.unfreeze_bert_last_layer), 0)
        writer.add_text("config/unfreeze_bert_last_n_layers", str(args.unfreeze_bert_last_n_layers), 0)
        writer.add_text("config/bert_last_layer_lr_ratio", str(args.bert_last_layer_lr_ratio), 0)
        writer.add_text("config/freeze_backbone_epochs", str(args.freeze_backbone_epochs), 0)
        writer.add_text("config/seed", str(args.seed), 0)
        writer.add_text("config/deterministic", str(args.deterministic), 0)
        writer.add_text("config/grad_accum_steps", str(args.grad_accum_steps), 0)
        writer.add_text("config/num_experts_mtoe", str(args.num_experts_mtoe), 0)
        writer.add_text("config/num_experts_msoe", str(args.num_experts_msoe), 0)
        writer.add_text("config/num_shared_experts", str(args.num_shared_experts), 0)
        writer.add_text("config/num_text_specific_experts", str(args.num_text_specific_experts), 0)
        writer.add_text("config/num_audio_specific_experts", str(args.num_audio_specific_experts), 0)
        writer.add_text("config/num_vision_specific_experts", str(args.num_vision_specific_experts), 0)
        writer.add_text("config/num_temporal_contrast_experts", str(args.num_temporal_contrast_experts), 0)
        writer.add_text("config/enable_text_shared_experts", str(args.enable_text_shared_experts), 0)
        writer.add_text("config/enable_temporal_contrast_experts", str(args.enable_temporal_contrast_experts), 0)
        writer.add_text("config/enable_temporal_contrast_loss", str(args.enable_temporal_contrast_loss), 0)
        writer.add_text("config/temporal_detach_task2", str(args.temporal_detach_task2), 0)
        writer.add_text("config/enable_tcif", str(args.enable_tcif), 0)
        writer.add_text("config/tcif_context_radius", str(args.tcif_context_radius), 0)
        writer.add_text("config/tcif_context_mode", str(args.tcif_context_mode), 0)
        writer.add_text("config/tcif_latent_dim", str(args.tcif_latent_dim), 0)
        writer.add_text("config/tcif_output_mode", str(args.tcif_output_mode), 0)
        writer.add_text("config/tcif_context_temperature", str(args.tcif_context_temperature), 0)
        writer.add_text("config/tcif_context_aux_weight", str(args.tcif_context_aux_weight), 0)
        writer.add_text("config/tcif_enable_transition_gate", str(args.tcif_enable_transition_gate), 0)
        writer.add_text("config/tcif_transition_gate_loss_weight", str(args.tcif_transition_gate_loss_weight), 0)
        writer.add_text("config/tcif_transition_gate_tau", str(args.tcif_transition_gate_tau), 0)
        writer.add_text("config/temporal_contrast_weight", str(args.temporal_contrast_weight), 0)
        writer.add_text("config/temporal_contrast_temperature", str(args.temporal_contrast_temperature), 0)
        writer.add_text("config/temporal_decay_tau", str(args.temporal_decay_tau), 0)
        writer.add_text("config/vit_context_ratio", str(args.vit_context_ratio), 0)
        writer.add_text("config/vit_context_lambda_mode", str(args.vit_context_lambda_mode), 0)
        writer.add_text("config/vit_context_lambda_init", str(args.vit_context_lambda_init), 0)
        writer.add_text("config/vit_context_attention_dim", str(args.vit_context_attention_dim), 0)
        writer.add_histogram("config/polarity_weights", polarity_weights_t.detach().float().cpu(), 0)

    for epoch in range(args.epochs):
        train_group_dispatch = {}
        freeze_backbone = epoch < args.freeze_backbone_epochs
        base_model = _unwrap_model(model)
        if hasattr(base_model, "reset_expert_load_stats"):
            base_model.reset_expert_load_stats()
        if hasattr(base_model, "reset_vit_context_lambda_stats"):
            base_model.reset_vit_context_lambda_stats()
        if getattr(model, "vit", None) is not None:
            _set_requires_grad(model.vit, not freeze_backbone)
        if getattr(model, "resnet", None) is not None:
            _set_requires_grad(model.resnet, False)
        _set_requires_grad(model.bert, not freeze_backbone)
        if args.unfreeze_bert_last_n_layers > 0:
            _set_bert_last_layers_requires_grad(model, args.unfreeze_bert_last_n_layers, True)
        _set_requires_grad(model.hubert, not freeze_backbone)
        model.train()
        if contrast_head is not None:
            contrast_head.train()
        if freeze_backbone:
            if getattr(model, "vit", None) is not None:
                model.vit.eval()
            if getattr(model, "resnet", None) is not None:
                model.resnet.eval()
            if getattr(model, "bert", None) is not None:
                model.bert.eval()
                if args.unfreeze_bert_last_n_layers > 0:
                    _set_bert_last_layers_train_mode(model, args.unfreeze_bert_last_n_layers, True)
            if getattr(model, "hubert", None) is not None:
                model.hubert.eval()
        elif getattr(model, "resnet", None) is not None:
            model.resnet.eval()
        total_loss = 0
        total_loss_intensity = 0
        total_loss_polarity = 0
        total_tc = 0
        train_batches = 0
        gating_entropy_sum = 0.0
        gating_var_sum = 0.0
        gating_steps = 0
        grad_norm_sum = 0.0
        grad_norm_steps = 0
        train_polarity_preds = []
        train_polarity_labels = []
        train_intensity_preds = []
        train_intensity_targets = []
        train_final_scores = []
        train_raw_valences = []
        train_reg_scores = []
        train_cls_expected_scores = []
        train_cls7_preds = []
        train_cls7_labels = []
        total_loss_reg = 0.0
        total_loss_cls7 = 0.0
        total_loss_sign = 0.0
        total_loss_score_sign = 0.0
        total_loss_hier_sign = 0.0
        total_loss_hier_mag = 0.0
        total_loss_sign_marginal = 0.0
        total_loss_signed_neutral_band = 0.0
        total_loss_zero_sign_margin = 0.0
        total_loss_reg_cls_mag_consistency = 0.0
        total_reg_cls_mag_consistency_selected = 0
        total_reg_cls_mag_consistency_positive_selected = 0
        total_reg_cls_mag_consistency_negative_selected = 0
        total_reg_cls_mag_consistency_balanced_batches = 0
        total_reg_cls_mag_consistency_fallback_batches = 0
        total_reg_cls_mag_consistency_candidates = 0
        total_reg_cls_mag_consistency_abs_gap = 0.0
        total_loss_cumulative = 0.0
        total_loss_oacr = 0.0
        total_loss_temporal_contrast = 0.0
        total_loss_tcif_context = 0.0
        total_loss_tcif_context_reg = 0.0
        total_loss_tcif_context_cls7 = 0.0
        total_tcif_context_valid = 0
        total_loss_tcif_transition_gate = 0.0
        total_tcif_transition_gate_valid = 0
        total_tcif_transition_gate_conflicts = 0
        total_tcif_transition_gate_target = 0.0
        total_tcif_regression_gate = 0.0
        total_tcif_ordinal_gate = 0.0
        total_temporal_valid_anchors = 0
        total_temporal_skipped_anchors = 0
        total_temporal_positive_pairs = 0
        total_temporal_positive_mass = 0.0
        total_temporal_legacy_valid_anchors = 0
        total_temporal_legacy_positive_pairs = 0
        total_temporal_legacy_positive_mass = 0.0
        total_temporal_effective_anchor_mass = 0.0
        train_signed_diag_accum = {
            "y_reg": [],
            "y_cls_expected": [],
            "y_final": [],
            "cls7_entropy": [],
        }

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()
        for step_idx, batch in enumerate(loop):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intensity = batch['intensity'].to(device)
            polarity = batch['polarity'].to(device)
            raw_valence = batch['raw_valence'].to(device)
            audio_values = batch["audio_values"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)

            gating_collector.clear()
            model_out = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                **_embedding_cache_kwargs(batch, device, use_text_cache=args.use_text_cache),
                **_tcif_context_kwargs(batch, device, use_text_cache=args.use_text_cache),
            )
            if args.output_head_mode == "legacy":
                out_intensity, out_polarity, tc_loss = model_out
                loss_intensity = compute_intensity_loss(out_intensity, intensity, args.intensity_loss_type)
                loss_polarity = compute_emotion_loss(out_polarity, polarity, criterion_polarity, {"loss_type": args.loss_type})
                loss_reg = out_intensity.new_tensor(0.0)
                loss_cls7 = out_intensity.new_tensor(0.0)
                loss_sign = out_intensity.new_tensor(0.0)
                loss_score_sign = out_intensity.new_tensor(0.0)
                loss_hier_sign = out_intensity.new_tensor(0.0)
                loss_hier_mag = out_intensity.new_tensor(0.0)
                loss_sign_marginal = out_intensity.new_tensor(0.0)
                loss_signed_neutral_band = out_intensity.new_tensor(0.0)
                loss_zero_sign_margin = out_intensity.new_tensor(0.0)
                loss_reg_cls_mag_consistency = out_intensity.new_tensor(0.0)
                reg_cls_mag_consistency_stats = _empty_reg_cls_mag_consistency_stats(raw_valence.numel())
                loss_cumulative = out_intensity.new_tensor(0.0)
                loss_oacr = out_intensity.new_tensor(0.0)
                loss_temporal_contrast = out_intensity.new_tensor(0.0)
                loss_tcif_context = out_intensity.new_tensor(0.0)
                loss_tcif_context_reg = out_intensity.new_tensor(0.0)
                loss_tcif_context_cls7 = out_intensity.new_tensor(0.0)
                tcif_context_valid_count = 0
                loss_tcif_transition_gate = out_intensity.new_tensor(0.0)
                tcif_transition_gate_stats = {
                    "valid_count": 0,
                    "sign_conflict_count": 0,
                    "target_mean": 0.0,
                    "regression_gate_mean": 0.0,
                    "ordinal_gate_mean": 0.0,
                }
                temporal_contrast_stats = {
                    "valid_anchor_count": 0,
                    "skipped_anchor_count": 0,
                    "positive_pair_count": 0,
                    "mean_positive_weight": 0.0,
                }
            else:
                out_intensity = None
                out_polarity = None
                y_reg = model_out["y_reg"]
                cls7_logits = model_out["cls7_logits"]
                sign_logits = model_out.get("sign_logits", None)
                tc_loss = model_out["tc_loss"]
                loss_reg = compute_regression_loss(y_reg, raw_valence, args.reg_loss_type)
                loss_cls7, cls7_target = _compute_cls7_loss(
                    cls7_logits,
                    raw_valence,
                    args.cls7_loss_type,
                    args.cls7_soft_tau,
                    args.cls7_low_abs_sample_weight_threshold,
                    args.cls7_low_abs_sample_weight,
                    cls7_class_weights_t,
                )
                sign_loss_logits = sign_logits if sign_logits is not None else y_reg
                loss_sign = _compute_sign_aux_loss(sign_loss_logits, raw_valence, args) if args.sign_aux_weight > 0 else y_reg.new_tensor(0.0)
                loss_hier_sign, loss_hier_mag = _compute_hier_sign_mag_losses(model_out, raw_valence)
                loss_sign_marginal = (
                    _compute_sign_marginal_loss(cls7_logits, raw_valence, args)
                    if args.sign_marginal_loss_weight > 0
                    else y_reg.new_tensor(0.0)
                )
                loss_signed_neutral_band = (
                    _compute_signed_neutral_band_loss(model_out, raw_valence, args)
                    if args.signed_neutral_band_weight > 0
                    else y_reg.new_tensor(0.0)
                )
                loss_zero_sign_margin = (
                    _compute_zero_sign_margin_loss(model_out, raw_valence, args)
                    if args.zero_sign_margin_weight > 0
                    else y_reg.new_tensor(0.0)
                )
                if args.reg_cls_mag_consistency_weight > 0:
                    loss_reg_cls_mag_consistency, reg_cls_mag_consistency_stats = (
                        _compute_reg_cls_mag_consistency_loss(model_out, args)
                    )
                else:
                    loss_reg_cls_mag_consistency = y_reg.new_tensor(0.0)
                    reg_cls_mag_consistency_stats = _empty_reg_cls_mag_consistency_stats(y_reg.numel())
                loss_cumulative = (
                    _compute_cumulative_loss(model_out, raw_valence)
                    if args.cumulative_loss_weight > 0
                    else y_reg.new_tensor(0.0)
                )
                score_sign_scores = model_out.get("extras", {}).get("y_final", y_reg)
                loss_score_sign = (
                    _compute_score_sign_aux_loss(score_sign_scores, raw_valence, args)
                    if args.score_sign_aux_weight > 0
                    else y_reg.new_tensor(0.0)
                )
                loss_oacr = _compute_oacr_loss(model_out, raw_valence, cls7_target, args, contrast_head)
                loss_temporal_contrast, temporal_contrast_stats = _compute_temporal_contrast_loss(model_out, batch, args)
                if args.tcif_context_aux_weight > 0:
                    (
                        loss_tcif_context,
                        loss_tcif_context_reg,
                        loss_tcif_context_cls7,
                        tcif_context_valid_count,
                    ) = _compute_tcif_context_aux_loss(model_out, raw_valence, args)
                else:
                    loss_tcif_context = y_reg.new_tensor(0.0)
                    loss_tcif_context_reg = y_reg.new_tensor(0.0)
                    loss_tcif_context_cls7 = y_reg.new_tensor(0.0)
                    tcif_context_valid_count = 0
                if args.tcif_transition_gate_loss_weight > 0:
                    (
                        loss_tcif_transition_gate,
                        tcif_transition_gate_stats,
                    ) = _compute_tcif_transition_gate_loss(
                        model_out,
                        batch,
                        raw_valence,
                        args,
                    )
                else:
                    loss_tcif_transition_gate = y_reg.new_tensor(0.0)
                    tcif_transition_gate_stats = {
                        "valid_count": 0,
                        "sign_conflict_count": 0,
                        "target_mean": 0.0,
                        "regression_gate_mean": 0.0,
                        "ordinal_gate_mean": 0.0,
                    }
                loss_intensity = loss_reg
                loss_polarity = y_reg.new_tensor(0.0)

            denom_epochs = max(1, int(args.epochs) - 1)
            cosine_alpha = 0.5 * float(args.alpha) * (1.0 + math.cos(math.pi * float(epoch) / float(denom_epochs)))
            if args.alpha_warmup_epochs > 0:
                warm = min(1.0, (epoch + 1) / args.alpha_warmup_epochs)
            else:
                warm = 1.0
            eff_alpha = cosine_alpha * warm
            if args.output_head_mode == "legacy":
                loss_raw = loss_intensity + args.lambda_polarity * loss_polarity + eff_alpha * tc_loss
            else:
                sign_struct_scale = _sign_struct_scale(args, epoch)
                args._sign_struct_scale = sign_struct_scale
                loss_raw = (
                    loss_reg
                    + args.cls7_loss_weight * loss_cls7
                    + args.sign_aux_weight * loss_sign
                    + sign_struct_scale * args.hier_sign_loss_weight * loss_hier_sign
                    + sign_struct_scale * args.hier_mag_loss_weight * loss_hier_mag
                    + sign_struct_scale * args.sign_marginal_loss_weight * loss_sign_marginal
                    + sign_struct_scale * args.signed_neutral_band_weight * loss_signed_neutral_band
                    + sign_struct_scale * args.zero_sign_margin_weight * loss_zero_sign_margin
                    + sign_struct_scale * args.reg_cls_mag_consistency_weight * loss_reg_cls_mag_consistency
                    + sign_struct_scale * args.cumulative_loss_weight * loss_cumulative
                    + args.score_sign_aux_weight * loss_score_sign
                    + args.lambda_oacr * loss_oacr
                    + args.temporal_contrast_weight * loss_temporal_contrast
                    + args.tcif_context_aux_weight * loss_tcif_context
                    + args.tcif_transition_gate_loss_weight
                    * loss_tcif_transition_gate
                    + eff_alpha * tc_loss
                )
            if not torch.isfinite(loss_raw):
                if (step_idx + 1) % max(1, args.log_every) == 0:
                    loop.set_postfix(loss=float("nan"), tc_loss=tc_loss.item())
                continue
            loss = loss_raw / max(1, args.grad_accum_steps)

            loss.backward()
            do_step = ((step_idx + 1) % max(1, args.grad_accum_steps) == 0) or (step_idx + 1 == len(train_loader))
            if do_step:
                if args.clip_grad_norm > 0:
                    grad_norm = clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        max_norm=args.clip_grad_norm,
                    )
                    grad_norm_sum += float(grad_norm)
                    grad_norm_steps += 1
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss_raw.item()
            total_loss_intensity += loss_intensity.item()
            total_loss_polarity += loss_polarity.item()
            total_loss_reg += loss_reg.item()
            total_loss_cls7 += loss_cls7.item()
            total_loss_sign += loss_sign.item()
            total_loss_score_sign += loss_score_sign.item()
            total_loss_hier_sign += loss_hier_sign.item()
            total_loss_hier_mag += loss_hier_mag.item()
            total_loss_sign_marginal += loss_sign_marginal.item()
            total_loss_signed_neutral_band += loss_signed_neutral_band.item()
            total_loss_zero_sign_margin += loss_zero_sign_margin.item()
            total_loss_reg_cls_mag_consistency += loss_reg_cls_mag_consistency.item()
            total_reg_cls_mag_consistency_selected += int(
                reg_cls_mag_consistency_stats.get("selected_count", 0)
            )
            total_reg_cls_mag_consistency_positive_selected += int(
                reg_cls_mag_consistency_stats.get("positive_selected_count", 0)
            )
            total_reg_cls_mag_consistency_negative_selected += int(
                reg_cls_mag_consistency_stats.get("negative_selected_count", 0)
            )
            total_reg_cls_mag_consistency_balanced_batches += int(
                reg_cls_mag_consistency_stats.get(
                    "sign_balanced_reduction_applied", 0
                )
            )
            total_reg_cls_mag_consistency_fallback_batches += int(
                reg_cls_mag_consistency_stats.get("single_branch_fallback", 0)
            )
            total_reg_cls_mag_consistency_candidates += int(
                reg_cls_mag_consistency_stats.get("candidate_count", 0)
            )
            total_reg_cls_mag_consistency_abs_gap += float(
                reg_cls_mag_consistency_stats.get("abs_gap_sum", 0.0)
            )
            total_loss_cumulative += loss_cumulative.item()
            total_loss_oacr += loss_oacr.item()
            total_loss_temporal_contrast += loss_temporal_contrast.item()
            total_loss_tcif_context += loss_tcif_context.item()
            total_loss_tcif_context_reg += loss_tcif_context_reg.item()
            total_loss_tcif_context_cls7 += loss_tcif_context_cls7.item()
            total_tcif_context_valid += int(tcif_context_valid_count)
            total_loss_tcif_transition_gate += loss_tcif_transition_gate.item()
            gate_valid = int(tcif_transition_gate_stats.get("valid_count", 0))
            total_tcif_transition_gate_valid += gate_valid
            total_tcif_transition_gate_conflicts += int(
                tcif_transition_gate_stats.get("sign_conflict_count", 0)
            )
            total_tcif_transition_gate_target += gate_valid * float(
                tcif_transition_gate_stats.get("target_mean", 0.0)
            )
            total_tcif_regression_gate += gate_valid * float(
                tcif_transition_gate_stats.get("regression_gate_mean", 0.0)
            )
            total_tcif_ordinal_gate += gate_valid * float(
                tcif_transition_gate_stats.get("ordinal_gate_mean", 0.0)
            )
            total_temporal_valid_anchors += int(temporal_contrast_stats.get("valid_anchor_count", 0))
            total_temporal_skipped_anchors += int(temporal_contrast_stats.get("skipped_anchor_count", 0))
            total_temporal_positive_pairs += int(temporal_contrast_stats.get("positive_pair_count", 0))
            total_temporal_positive_mass += float(temporal_contrast_stats.get("positive_mass", 0.0))
            total_temporal_legacy_valid_anchors += int(temporal_contrast_stats.get("legacy_valid_anchor_count", 0))
            total_temporal_legacy_positive_pairs += int(temporal_contrast_stats.get("legacy_positive_pair_count", 0))
            total_temporal_legacy_positive_mass += float(temporal_contrast_stats.get("legacy_positive_mass", 0.0))
            total_temporal_effective_anchor_mass += float(temporal_contrast_stats.get("effective_anchor_mass", 0.0))
            total_tc += tc_loss.item()
            train_batches += 1

            if args.output_head_mode == "legacy":
                polarity_preds = torch.argmax(out_polarity, dim=1)
                train_polarity_preds.extend(polarity_preds.cpu().numpy())
                train_polarity_labels.extend(polarity.cpu().numpy())
                train_intensity_preds.extend(out_intensity.detach().cpu().numpy().tolist())
                train_intensity_targets.extend(intensity.detach().cpu().numpy().tolist())
                signs = torch.where(polarity_preds == 0, -torch.ones_like(out_intensity), torch.ones_like(out_intensity))
                final_scores = out_intensity.detach() * signs
                train_final_scores.extend(final_scores.cpu().numpy().tolist())
                train_raw_valences.extend(raw_valence.detach().cpu().numpy().tolist())
            else:
                y_reg_eval, y_cls_expected, y_final = _signed_prediction_tensors(model_out, args)
                cls7_preds = torch.argmax(cls7_logits, dim=1)
                train_reg_scores.extend(y_reg_eval.detach().cpu().numpy().tolist())
                train_cls_expected_scores.extend(y_cls_expected.detach().cpu().numpy().tolist())
                train_final_scores.extend(y_final.detach().cpu().numpy().tolist())
                train_raw_valences.extend(raw_valence.detach().cpu().numpy().tolist())
                train_cls7_preds.extend(cls7_preds.detach().cpu().numpy().tolist())
                train_cls7_labels.extend(cls7_target.detach().cpu().numpy().tolist())
                train_signed_diag_accum["y_reg"].append(_tensor_stats(y_reg_eval))
                train_signed_diag_accum["y_cls_expected"].append(_tensor_stats(y_cls_expected))
                train_signed_diag_accum["y_final"].append(_tensor_stats(y_final))
                train_signed_diag_accum["cls7_entropy"].append(_cls7_entropy(cls7_logits))

            step_entropy = gating_collector.mean_entropy()
            step_variance = gating_collector.mean_variance()
            if step_entropy is not None and step_variance is not None:
                gating_entropy_sum += step_entropy
                gating_var_sum += step_variance
                gating_steps += 1

            if (step_idx + 1) % max(1, args.log_every) == 0:
                loop.set_postfix(loss=loss_raw.item(), tc_loss=tc_loss.item())

            if writer is not None and (global_step % args.log_every == 0):
                lr_by_group = {group.get("name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
                writer.add_scalar("train/loss", loss_raw.item(), global_step)
                writer.add_scalar("train/loss_intensity", loss_intensity.item(), global_step)
                writer.add_scalar("train/loss_polarity", loss_polarity.item(), global_step)
                writer.add_scalar("train/reg_loss", loss_reg.item(), global_step)
                writer.add_scalar("train/cls7_loss", loss_cls7.item(), global_step)
                writer.add_scalar("train/sign_loss", loss_sign.item(), global_step)
                writer.add_scalar(
                    "train/zero_sign_margin_loss",
                    loss_zero_sign_margin.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/weighted_zero_sign_margin_loss",
                    _sign_struct_scale(args)
                    * args.zero_sign_margin_weight
                    * loss_zero_sign_margin.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_loss",
                    loss_reg_cls_mag_consistency.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/weighted_reg_cls_mag_consistency_loss",
                    _sign_struct_scale(args)
                    * args.reg_cls_mag_consistency_weight
                    * loss_reg_cls_mag_consistency.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_selected_count",
                    reg_cls_mag_consistency_stats.get("selected_count", 0),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_positive_selected_count",
                    reg_cls_mag_consistency_stats.get("positive_selected_count", 0),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_negative_selected_count",
                    reg_cls_mag_consistency_stats.get("negative_selected_count", 0),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_coverage",
                    reg_cls_mag_consistency_stats.get("coverage", 0.0),
                    global_step,
                )
                writer.add_scalar(
                    "train/reg_cls_mag_consistency_mean_abs_gap",
                    reg_cls_mag_consistency_stats.get("mean_abs_gap", 0.0),
                    global_step,
                )
                writer.add_scalar("train/score_sign_loss", loss_score_sign.item(), global_step)
                writer.add_scalar("train/weighted_score_sign_loss", args.score_sign_aux_weight * loss_score_sign.item(), global_step)
                writer.add_scalar("train/oacr_loss", loss_oacr.item(), global_step)
                writer.add_scalar("train/weighted_oacr_loss", args.lambda_oacr * loss_oacr.item(), global_step)
                writer.add_scalar("train/temporal_contrast_loss", loss_temporal_contrast.item(), global_step)
                writer.add_scalar("train/weighted_temporal_contrast_loss", args.temporal_contrast_weight * loss_temporal_contrast.item(), global_step)
                writer.add_scalar("train/tcif_context_loss", loss_tcif_context.item(), global_step)
                writer.add_scalar("train/weighted_tcif_context_loss", args.tcif_context_aux_weight * loss_tcif_context.item(), global_step)
                writer.add_scalar("train/tcif_context_valid", tcif_context_valid_count, global_step)
                writer.add_scalar(
                    "train/tcif_transition_gate_loss",
                    loss_tcif_transition_gate.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/weighted_tcif_transition_gate_loss",
                    args.tcif_transition_gate_loss_weight
                    * loss_tcif_transition_gate.item(),
                    global_step,
                )
                writer.add_scalar(
                    "train/tcif_regression_continuation_gate",
                    tcif_transition_gate_stats.get("regression_gate_mean", 0.0),
                    global_step,
                )
                writer.add_scalar(
                    "train/tcif_ordinal_continuation_gate",
                    tcif_transition_gate_stats.get("ordinal_gate_mean", 0.0),
                    global_step,
                )
                writer.add_scalar(
                    "train/tcif_transition_gate_target",
                    tcif_transition_gate_stats.get("target_mean", 0.0),
                    global_step,
                )
                writer.add_scalar("train/temporal_valid_anchors", temporal_contrast_stats.get("valid_anchor_count", 0), global_step)
                writer.add_scalar("train/temporal_skipped_anchors", temporal_contrast_stats.get("skipped_anchor_count", 0), global_step)
                writer.add_scalar("train/temporal_positive_pairs", temporal_contrast_stats.get("positive_pair_count", 0), global_step)
                writer.add_scalar("train/tc_loss", tc_loss.item(), global_step)
                writer.add_scalar("train/alpha", eff_alpha, global_step)
                writer.add_scalar("train/lr_head", lr_by_group.get("head", 0.0), global_step)
                writer.add_scalar("train/lr_router", lr_by_group.get("router", 0.0), global_step)
                writer.add_scalar("train/lr_bert_last_layer", lr_by_group.get("bert_last_layers", 0.0), global_step)
                writer.add_scalar("train/lr_bert_last_layers", lr_by_group.get("bert_last_layers", 0.0), global_step)
                writer.add_scalar("train/lr_backbone", lr_by_group.get("backbone", 0.0), global_step)
                writer.add_scalar("Learning_Rate", lr_by_group.get("head", 0.0), global_step)
                if step_entropy is not None:
                    writer.add_scalar("train/Gating_Entropy", step_entropy, global_step)
                if step_variance is not None:
                    writer.add_scalar("train/Expert_Load_Variance", step_variance, global_step)
                if grad_norm_steps > 0:
                    writer.add_scalar("Train_Grad_Norm", grad_norm_sum / grad_norm_steps, global_step)
                writer.add_scalar("train/freeze_backbone", 1.0 if freeze_backbone else 0.0, global_step)
                if args.output_head_mode == "signed_reg_cls7":
                    writer.add_scalar("train/cls7_entropy", _cls7_entropy(cls7_logits), global_step)
                    for name, stats in [
                        ("y_reg", _tensor_stats(y_reg_eval)),
                        ("y_cls_expected", _tensor_stats(y_cls_expected)),
                        ("y_final", _tensor_stats(y_final)),
                    ]:
                        for key, value in stats.items():
                            writer.add_scalar(f"train/{name}_{key}", value, global_step)
            global_step += 1
        avg_loss = total_loss / max(1, train_batches)
        if args.output_head_mode == "legacy":
            train_acc, train_macro_f1, train_prec, train_rec = metrics_from_lists(
                train_polarity_labels,
                train_polarity_preds,
                num_polarity_classes,
            )
            train_rmse, train_mae, train_pearson, train_spearman = regression_metrics(
                train_final_scores,
                train_raw_valences,
            )
            train_final_pred_7 = [_valence_to_7class(s) for s in train_final_scores]
            train_final_gt_7 = [_valence_to_7class(v) for v in train_raw_valences]
            train_final_acc7, train_final_macro_f1, _, _ = metrics_from_lists(train_final_gt_7, train_final_pred_7, 7)
            train_final_pred_2 = [_valence_to_binary(s) for s in train_final_scores]
            train_final_gt_2 = [_valence_to_binary(v) for v in train_raw_valences]
            train_final_acc2 = accuracy_from_lists(train_final_gt_2, train_final_pred_2)
            print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} Polarity Acc: {train_acc:.4f} MacroF1: {train_macro_f1:.4f}")
            print(f"Epoch {epoch+1} FinalScore Acc2: {train_final_acc2:.4f} Acc7: {train_final_acc7:.4f} MacroF1_7: {train_final_macro_f1:.4f}")
            print(
                f"Epoch {epoch+1} Parts: "
                f"intensity={total_loss_intensity/max(1, train_batches):.4f} "
                f"polarity={total_loss_polarity/max(1, train_batches):.4f} "
                f"tc={total_tc/max(1, train_batches):.4f} "
                f"alpha={eff_alpha:.4f}"
            )
        else:
            train_signed_metrics = summarize_signed_metrics(
                train_reg_scores,
                train_cls_expected_scores,
                train_final_scores,
                train_raw_valences,
                train_cls7_labels,
                train_cls7_preds,
            )
            train_acc = train_signed_metrics["cls7_acc"]
            train_macro_f1 = train_signed_metrics["cls7_macro_f1"]
            train_prec = train_signed_metrics["cls7_precision"]
            train_rec = train_signed_metrics["cls7_recall"]
            train_rmse = train_signed_metrics["final"]["rmse"]
            train_mae = train_signed_metrics["final"]["mae"]
            train_pearson = train_signed_metrics["final"]["pearson"]
            train_spearman = train_signed_metrics["final"]["spearman"]
            train_final_acc2 = train_signed_metrics["final"]["acc2"]
            train_final_acc7 = train_signed_metrics["final"]["acc7"]
            train_final_macro_f1 = train_signed_metrics["final"]["macro_f1_7"]
            print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f} Cls7 Acc: {train_acc:.4f} MacroF1: {train_macro_f1:.4f}")
            print(
                f"Epoch {epoch+1} Train SignedMetrics "
                f"reg_mae={train_signed_metrics['reg']['mae']:.4f} reg_acc7={train_signed_metrics['reg']['acc7']:.4f} "
                f"cls_exp_mae={train_signed_metrics['cls_expected']['mae']:.4f} cls_exp_acc7={train_signed_metrics['cls_expected']['acc7']:.4f} "
                f"final_mae={train_mae:.4f} final_acc7={train_final_acc7:.4f} final_acc2={train_final_acc2:.4f}"
            )
            print(
                f"Epoch {epoch+1} Parts: "
                f"reg={total_loss_reg/max(1, train_batches):.4f} "
                f"cls7={total_loss_cls7/max(1, train_batches):.4f} "
                f"sign={total_loss_sign/max(1, train_batches):.4f} "
                f"score_sign={total_loss_score_sign/max(1, train_batches):.4f} "
                f"weighted_score_sign={args.score_sign_aux_weight * total_loss_score_sign/max(1, train_batches):.4f} "
                f"signed_neutral_band={total_loss_signed_neutral_band/max(1, train_batches):.4f} "
                f"weighted_signed_neutral_band={_sign_struct_scale(args) * args.signed_neutral_band_weight * total_loss_signed_neutral_band/max(1, train_batches):.4f} "
                f"zero_sign_margin={total_loss_zero_sign_margin/max(1, train_batches):.4f} "
                f"weighted_zero_sign_margin={_sign_struct_scale(args) * args.zero_sign_margin_weight * total_loss_zero_sign_margin/max(1, train_batches):.4f} "
                f"zero_sign_margin_effective_weight={_sign_struct_scale(args) * args.zero_sign_margin_weight:.4f} "
                f"reg_cls_mag_consistency={total_loss_reg_cls_mag_consistency/max(1, train_batches):.4f} "
                f"weighted_reg_cls_mag_consistency={_sign_struct_scale(args) * args.reg_cls_mag_consistency_weight * total_loss_reg_cls_mag_consistency/max(1, train_batches):.4f} "
                f"reg_cls_mag_effective_weight={_sign_struct_scale(args) * args.reg_cls_mag_consistency_weight:.4f} "
                f"reg_cls_mag_selected={total_reg_cls_mag_consistency_selected} "
                f"reg_cls_mag_positive_selected={total_reg_cls_mag_consistency_positive_selected} "
                f"reg_cls_mag_negative_selected={total_reg_cls_mag_consistency_negative_selected} "
                f"reg_cls_mag_balanced_batches={total_reg_cls_mag_consistency_balanced_batches} "
                f"reg_cls_mag_fallback_batches={total_reg_cls_mag_consistency_fallback_batches} "
                f"reg_cls_mag_coverage={total_reg_cls_mag_consistency_selected/max(1, total_reg_cls_mag_consistency_candidates):.6f} "
                f"reg_cls_mag_gap={total_reg_cls_mag_consistency_abs_gap/max(1, total_reg_cls_mag_consistency_selected):.6f} "
                f"oacr={total_loss_oacr/max(1, train_batches):.4f} "
                f"weighted_oacr={args.lambda_oacr * total_loss_oacr/max(1, train_batches):.4f} "
                f"temporal={total_loss_temporal_contrast/max(1, train_batches):.4f} "
                f"tcif_context={total_loss_tcif_context/max(1, train_batches):.4f} "
                f"weighted_tcif_context={args.tcif_context_aux_weight * total_loss_tcif_context/max(1, train_batches):.4f} "
                f"tcif_context_valid={total_tcif_context_valid} "
                f"tcif_gate={total_loss_tcif_transition_gate/max(1, train_batches):.4f} "
                f"weighted_tcif_gate={args.tcif_transition_gate_loss_weight * total_loss_tcif_transition_gate/max(1, train_batches):.4f} "
                f"tcif_gate_mean_reg={total_tcif_regression_gate/max(1, total_tcif_transition_gate_valid):.4f} "
                f"tcif_gate_mean_cls={total_tcif_ordinal_gate/max(1, total_tcif_transition_gate_valid):.4f} "
                f"tcif_gate_target={total_tcif_transition_gate_target/max(1, total_tcif_transition_gate_valid):.4f} "
                f"tcif_gate_conflicts={total_tcif_transition_gate_conflicts} "
                f"tc={total_tc/max(1, train_batches):.4f} "
                f"alpha={eff_alpha:.4f}"
            )
            if _temporal_contrast_enabled(args):
                anchor_retention = total_temporal_valid_anchors / max(1, total_temporal_legacy_valid_anchors)
                pair_retention = total_temporal_positive_pairs / max(1, total_temporal_legacy_positive_pairs)
                mass_retention = total_temporal_positive_mass / max(1e-12, total_temporal_legacy_positive_mass)
                print(
                    f"Epoch {epoch+1} TemporalContrast "
                    f"valid_anchors={total_temporal_valid_anchors} "
                    f"legacy_valid_anchors={total_temporal_legacy_valid_anchors} "
                    f"anchor_retention={anchor_retention:.6f} "
                    f"skipped_anchors={total_temporal_skipped_anchors} "
                    f"positive_pairs={total_temporal_positive_pairs} "
                    f"legacy_positive_pairs={total_temporal_legacy_positive_pairs} "
                    f"pair_retention={pair_retention:.6f} "
                    f"mass_retention={mass_retention:.6f}"
                )
                coverage_record = {
                    "epoch": epoch + 1,
                    "temporal_kernel": getattr(args, "temporal_kernel", "legacy_exp"),
                    "zero_bridge_weight": getattr(args, "temporal_zero_bridge_weight", 0.25),
                    "valid_anchor_count": total_temporal_valid_anchors,
                    "legacy_valid_anchor_count": total_temporal_legacy_valid_anchors,
                    "anchor_retention": anchor_retention,
                    "positive_pair_count": total_temporal_positive_pairs,
                    "legacy_positive_pair_count": total_temporal_legacy_positive_pairs,
                    "directed_pair_retention": pair_retention,
                    "positive_mass": total_temporal_positive_mass,
                    "legacy_positive_mass": total_temporal_legacy_positive_mass,
                    "positive_mass_retention": mass_retention,
                    "effective_anchor_mass": total_temporal_effective_anchor_mass,
                }
                coverage_path = _infer_results_root_from_save_dir(args.save_dir) / "temporal_coverage_history.jsonl"
                coverage_path.parent.mkdir(parents=True, exist_ok=True)
                with coverage_path.open("a", encoding="utf-8") as coverage_file:
                    coverage_file.write(json.dumps(coverage_record, ensure_ascii=False) + "\n")
        train_group_dispatch = _get_train_msoe_group_dispatch(model)
        if train_group_dispatch:
            print(
                "Epoch "
                f"{epoch+1} MSoEGroupDispatchShare "
                f"{_format_msoe_group_dispatch_shares(train_group_dispatch)}"
            )

        val_acc, val_loss, val_macro_f1, val_prec, val_rec, val_rmse, val_mae, val_pearson, val_spearman, val_final_acc2, val_final_acc7, val_final_macro_f1, val_loss_parts = validate(
            model,
            val_loader,
            device,
            criterion_polarity,
            num_polarity_classes,
            args.lambda_polarity,
            eff_alpha,
            args.intensity_loss_type,
            args=args,
            contrast_head=contrast_head,
        )
        if args.output_head_mode == "legacy":
            print(
                f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Polarity Acc: {val_acc:.4f} "
                f"MacroF1: {val_macro_f1:.4f} RMSE: {val_rmse:.4f} MAE: {val_mae:.4f}"
            )
            print(
                f"Epoch {epoch+1} Val Parts: "
                f"intensity={val_loss_parts['intensity']:.4f} "
                f"polarity={val_loss_parts['polarity']:.4f} "
                f"weighted_polarity={val_loss_parts['weighted_polarity']:.4f} "
                f"tc={val_loss_parts['tc']:.4f} "
                f"weighted_tc={val_loss_parts['weighted_tc']:.4f}"
            )
        else:
            val_signed = val_loss_parts["metrics"]
            print(
                f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Cls7 Acc: {val_acc:.4f} "
                f"MacroF1: {val_macro_f1:.4f} Final RMSE: {val_rmse:.4f} Final MAE: {val_mae:.4f}"
            )
            print(
                f"Epoch {epoch+1} Val Parts: "
                f"reg={val_loss_parts['reg']:.4f} "
                f"cls7={val_loss_parts['cls7']:.4f} "
                f"weighted_cls7={val_loss_parts['weighted_cls7']:.4f} "
                f"sign={val_loss_parts['sign']:.4f} "
                f"score_sign={val_loss_parts['score_sign']:.4f} "
                f"weighted_score_sign={val_loss_parts['weighted_score_sign']:.4f} "
                f"signed_neutral_band={val_loss_parts['signed_neutral_band']:.4f} "
                f"weighted_signed_neutral_band={val_loss_parts['weighted_signed_neutral_band']:.4f} "
                f"zero_sign_margin={val_loss_parts['zero_sign_margin']:.4f} "
                f"weighted_zero_sign_margin={val_loss_parts['weighted_zero_sign_margin']:.4f} "
                f"zero_sign_margin_effective_weight={val_loss_parts['zero_sign_margin_effective_weight']:.4f} "
                f"reg_cls_mag_consistency={val_loss_parts['reg_cls_mag_consistency']:.4f} "
                f"weighted_reg_cls_mag_consistency={val_loss_parts['weighted_reg_cls_mag_consistency']:.4f} "
                f"reg_cls_mag_effective_weight={val_loss_parts['reg_cls_mag_consistency_effective_weight']:.4f} "
                f"reg_cls_mag_selected={val_loss_parts['reg_cls_mag_consistency_selected_count']} "
                f"reg_cls_mag_positive_selected={val_loss_parts['reg_cls_mag_consistency_positive_selected_count']} "
                f"reg_cls_mag_negative_selected={val_loss_parts['reg_cls_mag_consistency_negative_selected_count']} "
                f"reg_cls_mag_balanced_batches={val_loss_parts['reg_cls_mag_consistency_balanced_batch_count']} "
                f"reg_cls_mag_fallback_batches={val_loss_parts['reg_cls_mag_consistency_fallback_batch_count']} "
                f"reg_cls_mag_coverage={val_loss_parts['reg_cls_mag_consistency_coverage']:.6f} "
                f"reg_cls_mag_gap={val_loss_parts['reg_cls_mag_consistency_mean_abs_gap']:.6f} "
                f"oacr={val_loss_parts['oacr']:.4f} "
                f"weighted_oacr={val_loss_parts['weighted_oacr']:.4f} "
                f"temporal={val_loss_parts['temporal_contrast']:.4f} "
                f"weighted_temporal={val_loss_parts['weighted_temporal_contrast']:.4f} "
                f"tcif_context={val_loss_parts['tcif_context']:.4f} "
                f"weighted_tcif_context={val_loss_parts['weighted_tcif_context']:.4f} "
                f"tcif_context_valid={val_loss_parts['tcif_context_valid']} "
                f"tc={val_loss_parts['tc']:.4f} "
                f"weighted_tc={val_loss_parts['weighted_tc']:.4f}"
            )
            print(
                f"Epoch {epoch+1} Val SignedMetrics "
                f"reg_mae={val_signed['reg']['mae']:.4f} reg_acc7={val_signed['reg']['acc7']:.4f} "
                f"cls_exp_mae={val_signed['cls_expected']['mae']:.4f} cls_exp_acc7={val_signed['cls_expected']['acc7']:.4f} "
                f"final_mae={val_signed['final']['mae']:.4f} final_acc7={val_signed['final']['acc7']:.4f}"
            )
        print(f"Epoch {epoch+1} FinalScore Acc2: {val_final_acc2:.4f} Acc7: {val_final_acc7:.4f} MacroF1_7: {val_final_macro_f1:.4f}")

        test_metrics = None
        test_loss_parts = None
        if args.evaluate_test_each_epoch:
            test_acc, test_loss, test_macro_f1, _test_prec, _test_rec, test_rmse, test_mae, test_pearson, test_spearman, test_final_acc2, test_final_acc7, test_final_macro_f1, test_loss_parts = validate(
                model,
                test_loader,
                device,
                criterion_polarity,
                num_polarity_classes,
                args.lambda_polarity,
                eff_alpha,
                args.intensity_loss_type,
                args=args,
                contrast_head=contrast_head,
            )
            if args.output_head_mode == "legacy":
                print(
                    f"Epoch {epoch+1} Test Loss: {test_loss:.4f} Polarity Acc: {test_acc:.4f} "
                    f"MacroF1: {test_macro_f1:.4f} RMSE: {test_rmse:.4f} MAE: {test_mae:.4f}"
                )
                print(
                    f"Epoch {epoch+1} Test Parts: "
                    f"intensity={test_loss_parts['intensity']:.4f} "
                    f"polarity={test_loss_parts['polarity']:.4f} "
                    f"weighted_polarity={test_loss_parts['weighted_polarity']:.4f} "
                    f"tc={test_loss_parts['tc']:.4f} "
                    f"weighted_tc={test_loss_parts['weighted_tc']:.4f}"
                )
            else:
                test_signed = test_loss_parts["metrics"]
                print(
                    f"Epoch {epoch+1} Test Loss: {test_loss:.4f} Cls7 Acc: {test_acc:.4f} "
                    f"MacroF1: {test_macro_f1:.4f} Final RMSE: {test_rmse:.4f} Final MAE: {test_mae:.4f}"
                )
                print(
                    f"Epoch {epoch+1} Test Parts: "
                    f"reg={test_loss_parts['reg']:.4f} "
                    f"cls7={test_loss_parts['cls7']:.4f} "
                    f"weighted_cls7={test_loss_parts['weighted_cls7']:.4f} "
                    f"sign={test_loss_parts['sign']:.4f} "
                    f"score_sign={test_loss_parts['score_sign']:.4f} "
                    f"weighted_score_sign={test_loss_parts['weighted_score_sign']:.4f} "
                    f"signed_neutral_band={test_loss_parts['signed_neutral_band']:.4f} "
                    f"weighted_signed_neutral_band={test_loss_parts['weighted_signed_neutral_band']:.4f} "
                    f"zero_sign_margin={test_loss_parts['zero_sign_margin']:.4f} "
                    f"weighted_zero_sign_margin={test_loss_parts['weighted_zero_sign_margin']:.4f} "
                    f"zero_sign_margin_effective_weight={test_loss_parts['zero_sign_margin_effective_weight']:.4f} "
                    f"reg_cls_mag_consistency={test_loss_parts['reg_cls_mag_consistency']:.4f} "
                    f"weighted_reg_cls_mag_consistency={test_loss_parts['weighted_reg_cls_mag_consistency']:.4f} "
                    f"reg_cls_mag_effective_weight={test_loss_parts['reg_cls_mag_consistency_effective_weight']:.4f} "
                    f"reg_cls_mag_selected={test_loss_parts['reg_cls_mag_consistency_selected_count']} "
                    f"reg_cls_mag_positive_selected={test_loss_parts['reg_cls_mag_consistency_positive_selected_count']} "
                    f"reg_cls_mag_negative_selected={test_loss_parts['reg_cls_mag_consistency_negative_selected_count']} "
                    f"reg_cls_mag_balanced_batches={test_loss_parts['reg_cls_mag_consistency_balanced_batch_count']} "
                    f"reg_cls_mag_fallback_batches={test_loss_parts['reg_cls_mag_consistency_fallback_batch_count']} "
                    f"reg_cls_mag_coverage={test_loss_parts['reg_cls_mag_consistency_coverage']:.6f} "
                    f"reg_cls_mag_gap={test_loss_parts['reg_cls_mag_consistency_mean_abs_gap']:.6f} "
                    f"oacr={test_loss_parts['oacr']:.4f} "
                    f"weighted_oacr={test_loss_parts['weighted_oacr']:.4f} "
                    f"temporal={test_loss_parts['temporal_contrast']:.4f} "
                    f"weighted_temporal={test_loss_parts['weighted_temporal_contrast']:.4f} "
                    f"tcif_context={test_loss_parts['tcif_context']:.4f} "
                    f"weighted_tcif_context={test_loss_parts['weighted_tcif_context']:.4f} "
                    f"tcif_context_valid={test_loss_parts['tcif_context_valid']} "
                    f"tc={test_loss_parts['tc']:.4f} "
                    f"weighted_tc={test_loss_parts['weighted_tc']:.4f}"
                )
                print(
                    f"Epoch {epoch+1} Test SignedMetrics "
                    f"reg_mae={test_signed['reg']['mae']:.4f} reg_acc7={test_signed['reg']['acc7']:.4f} "
                    f"cls_exp_mae={test_signed['cls_expected']['mae']:.4f} cls_exp_acc7={test_signed['cls_expected']['acc7']:.4f} "
                    f"final_mae={test_signed['final']['mae']:.4f} final_acc7={test_signed['final']['acc7']:.4f}"
                )
            print(f"Epoch {epoch+1} Test FinalScore Acc2: {test_final_acc2:.4f} Acc7: {test_final_acc7:.4f} MacroF1_7: {test_final_macro_f1:.4f}")
        else:
            print(f"Epoch {epoch+1} Test evaluation skipped; checkpoint selection is validation-only")
        print(f"Epoch {epoch+1} Freeze backbone: {freeze_backbone}")
        lambda_state = _get_vit_context_lambda_state(model)
        if lambda_state is not None:
            if lambda_state.get("mode") in {"adaptive", "attention"}:
                mode_name = lambda_state.get("mode")
                base_values = lambda_state["base_lambda"]
                mean_values = lambda_state.get("effective_lambda_mean")
                std_values = lambda_state.get("effective_lambda_std")
                if mean_values is not None and std_values is not None:
                    entropy_text = ""
                    if "attention_center_entropy_mean" in lambda_state:
                        entropy_text = f" attn_entropy={lambda_state['attention_center_entropy_mean']:.6f}"
                    print(
                        f"Epoch {epoch+1} VitContextLambda {mode_name} "
                        f"base=({base_values[0]:.8f},{base_values[1]:.8f},{base_values[2]:.8f}) "
                        f"mean=({mean_values[0]:.8f},{mean_values[1]:.8f},{mean_values[2]:.8f}) "
                        f"std=({std_values[0]:.8f},{std_values[1]:.8f},{std_values[2]:.8f})"
                        f"{entropy_text}"
                    )
                else:
                    print(
                        f"Epoch {epoch+1} VitContextLambda {mode_name} "
                        f"base=({base_values[0]:.8f},{base_values[1]:.8f},{base_values[2]:.8f}) "
                        "mean/std=unavailable"
                    )
            else:
                lambda_values = lambda_state["effective_lambda"]
                print(
                    f"Epoch {epoch+1} VitContextLambda "
                    f"main={lambda_values[0]:.8f} prev={lambda_values[1]:.8f} next={lambda_values[2]:.8f}"
                )
            _append_lambda_history(args.save_dir, epoch + 1, lambda_state)
            _write_final_lambda_distribution(args.save_dir, epoch + 1, lambda_state)

        if writer is not None:
            writer.add_scalar("epoch/train_loss", avg_loss, epoch)
            writer.add_scalar("epoch/train_loss_intensity", total_loss_intensity / len(train_loader), epoch)
            writer.add_scalar("epoch/train_loss_polarity", total_loss_polarity / len(train_loader), epoch)
            writer.add_scalar("epoch/train_tc_loss", total_tc / len(train_loader), epoch)
            writer.add_scalar("epoch/train_macro_f1", train_macro_f1, epoch)
            writer.add_scalar("epoch/train_acc", train_acc, epoch)
            writer.add_scalar("epoch/train_rmse", train_rmse, epoch)
            writer.add_scalar("epoch/train_mae", train_mae, epoch)
            writer.add_scalar("epoch/train_pearson", train_pearson, epoch)
            writer.add_scalar("epoch/train_spearman", train_spearman, epoch)
            writer.add_scalar("epoch/train_final_acc2", train_final_acc2, epoch)
            writer.add_scalar("epoch/train_final_acc7", train_final_acc7, epoch)
            writer.add_scalar("epoch/train_final_macro_f1_7", train_final_macro_f1, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_tc_loss", val_loss_parts["tc"], epoch)
            writer.add_scalar("epoch/val_loss_weighted_tc", val_loss_parts["weighted_tc"], epoch)
            writer.add_scalar("epoch/val_tcif_context_loss", val_loss_parts.get("tcif_context", 0.0), epoch)
            writer.add_scalar("epoch/val_weighted_tcif_context_loss", val_loss_parts.get("weighted_tcif_context", 0.0), epoch)
            writer.add_scalar("epoch/val_macro_f1", val_macro_f1, epoch)
            writer.add_scalar("epoch/val_acc", val_acc, epoch)
            writer.add_scalar("epoch/val_rmse", val_rmse, epoch)
            writer.add_scalar("epoch/val_mae", val_mae, epoch)
            writer.add_scalar("epoch/val_pearson", val_pearson, epoch)
            writer.add_scalar("epoch/val_spearman", val_spearman, epoch)
            writer.add_scalar("epoch/val_final_acc2", val_final_acc2, epoch)
            writer.add_scalar("epoch/val_final_acc7", val_final_acc7, epoch)
            writer.add_scalar("epoch/val_final_macro_f1_7", val_final_macro_f1, epoch)
            if test_loss_parts is not None:
                writer.add_scalar("epoch/test_loss", test_loss, epoch)
                writer.add_scalar("epoch/test_tc_loss", test_loss_parts["tc"], epoch)
                writer.add_scalar("epoch/test_loss_weighted_tc", test_loss_parts["weighted_tc"], epoch)
                writer.add_scalar("epoch/test_tcif_context_loss", test_loss_parts.get("tcif_context", 0.0), epoch)
                writer.add_scalar("epoch/test_weighted_tcif_context_loss", test_loss_parts.get("weighted_tcif_context", 0.0), epoch)
                writer.add_scalar("epoch/test_macro_f1", test_macro_f1, epoch)
                writer.add_scalar("epoch/test_acc", test_acc, epoch)
                writer.add_scalar("epoch/test_rmse", test_rmse, epoch)
                writer.add_scalar("epoch/test_mae", test_mae, epoch)
                writer.add_scalar("epoch/test_pearson", test_pearson, epoch)
                writer.add_scalar("epoch/test_spearman", test_spearman, epoch)
                writer.add_scalar("epoch/test_final_acc2", test_final_acc2, epoch)
                writer.add_scalar("epoch/test_final_acc7", test_final_acc7, epoch)
                writer.add_scalar("epoch/test_final_macro_f1_7", test_final_macro_f1, epoch)
            writer.add_scalar("Loss/Train_Total", avg_loss, epoch)
            writer.add_scalar("Loss/Val_Total", val_loss, epoch)
            if test_loss_parts is not None:
                writer.add_scalar("Loss/Test_Total", test_loss, epoch)
            writer.add_scalar("Train_Intensity_Loss", total_loss_intensity / max(1, train_batches), epoch)
            writer.add_scalar("Train_Polarity_Loss", total_loss_polarity / max(1, train_batches), epoch)
            writer.add_scalar("Train_Reg_Loss", total_loss_reg / max(1, train_batches), epoch)
            writer.add_scalar("Train_Cls7_Loss", total_loss_cls7 / max(1, train_batches), epoch)
            writer.add_scalar("Train_Sign_Loss", total_loss_sign / max(1, train_batches), epoch)
            writer.add_scalar(
                "Train_Zero_Sign_Margin_Loss",
                total_loss_zero_sign_margin / max(1, train_batches),
                epoch,
            )
            writer.add_scalar(
                "Train_Weighted_Zero_Sign_Margin_Loss",
                _sign_struct_scale(args)
                * args.zero_sign_margin_weight
                * total_loss_zero_sign_margin
                / max(1, train_batches),
                epoch,
            )
            writer.add_scalar(
                "Train_Reg_Cls_Mag_Consistency_Loss",
                total_loss_reg_cls_mag_consistency / max(1, train_batches),
                epoch,
            )
            writer.add_scalar(
                "Train_Weighted_Reg_Cls_Mag_Consistency_Loss",
                _sign_struct_scale(args)
                * args.reg_cls_mag_consistency_weight
                * total_loss_reg_cls_mag_consistency
                / max(1, train_batches),
                epoch,
            )
            writer.add_scalar(
                "epoch/train_reg_cls_mag_consistency_selected_count",
                total_reg_cls_mag_consistency_selected,
                epoch,
            )
            writer.add_scalar(
                "epoch/train_reg_cls_mag_consistency_coverage",
                total_reg_cls_mag_consistency_selected
                / max(1, total_reg_cls_mag_consistency_candidates),
                epoch,
            )
            writer.add_scalar(
                "epoch/train_reg_cls_mag_consistency_mean_abs_gap",
                total_reg_cls_mag_consistency_abs_gap
                / max(1, total_reg_cls_mag_consistency_selected),
                epoch,
            )
            writer.add_scalar("Train_Score_Sign_Loss", total_loss_score_sign / max(1, train_batches), epoch)
            writer.add_scalar("Train_Weighted_Score_Sign_Loss", args.score_sign_aux_weight * (total_loss_score_sign / max(1, train_batches)), epoch)
            writer.add_scalar("Train_OACR_Loss", total_loss_oacr / max(1, train_batches), epoch)
            writer.add_scalar("Train_Weighted_OACR_Loss", args.lambda_oacr * (total_loss_oacr / max(1, train_batches)), epoch)
            writer.add_scalar("Train_Temporal_Contrast_Loss", total_loss_temporal_contrast / max(1, train_batches), epoch)
            writer.add_scalar("Train_Weighted_Temporal_Contrast_Loss", args.temporal_contrast_weight * (total_loss_temporal_contrast / max(1, train_batches)), epoch)
            writer.add_scalar("Train_TCIF_Context_Loss", total_loss_tcif_context / max(1, train_batches), epoch)
            writer.add_scalar("Train_Weighted_TCIF_Context_Loss", args.tcif_context_aux_weight * (total_loss_tcif_context / max(1, train_batches)), epoch)
            writer.add_scalar("Train_TCIF_Context_Reg_Loss", total_loss_tcif_context_reg / max(1, train_batches), epoch)
            writer.add_scalar("Train_TCIF_Context_Cls7_Loss", total_loss_tcif_context_cls7 / max(1, train_batches), epoch)
            writer.add_scalar("Train_TCIF_Context_Valid", total_tcif_context_valid, epoch)
            writer.add_scalar("Train_TCIF_Transition_Gate_Loss", total_loss_tcif_transition_gate / max(1, train_batches), epoch)
            writer.add_scalar("Train_Weighted_TCIF_Transition_Gate_Loss", args.tcif_transition_gate_loss_weight * total_loss_tcif_transition_gate / max(1, train_batches), epoch)
            writer.add_scalar("Train_TCIF_Regression_Continuation_Gate", total_tcif_regression_gate / max(1, total_tcif_transition_gate_valid), epoch)
            writer.add_scalar("Train_TCIF_Ordinal_Continuation_Gate", total_tcif_ordinal_gate / max(1, total_tcif_transition_gate_valid), epoch)
            writer.add_scalar("Train_TCIF_Transition_Gate_Target", total_tcif_transition_gate_target / max(1, total_tcif_transition_gate_valid), epoch)
            writer.add_scalar("epoch/train_temporal_valid_anchors", total_temporal_valid_anchors, epoch)
            writer.add_scalar("epoch/train_temporal_skipped_anchors", total_temporal_skipped_anchors, epoch)
            writer.add_scalar("epoch/train_temporal_positive_pairs", total_temporal_positive_pairs, epoch)
            writer.add_scalar("Train_TC_Loss", total_tc / max(1, train_batches), epoch)
            writer.add_scalar("Val_TC_Loss", val_loss_parts["tc"], epoch)
            if test_loss_parts is not None:
                writer.add_scalar("Test_TC_Loss", test_loss_parts["tc"], epoch)
            if args.output_head_mode == "legacy":
                writer.add_scalar("epoch/val_loss_intensity", val_loss_parts["intensity"], epoch)
                writer.add_scalar("epoch/val_loss_polarity", val_loss_parts["polarity"], epoch)
                writer.add_scalar("epoch/val_loss_weighted_polarity", val_loss_parts["weighted_polarity"], epoch)
                writer.add_scalar("Val_Intensity_Loss", val_loss_parts["intensity"], epoch)
                writer.add_scalar("Val_Polarity_Loss", val_loss_parts["polarity"], epoch)
                if test_loss_parts is not None:
                    writer.add_scalar("epoch/test_loss_intensity", test_loss_parts["intensity"], epoch)
                    writer.add_scalar("epoch/test_loss_polarity", test_loss_parts["polarity"], epoch)
                    writer.add_scalar("epoch/test_loss_weighted_polarity", test_loss_parts["weighted_polarity"], epoch)
                    writer.add_scalar("Test_Intensity_Loss", test_loss_parts["intensity"], epoch)
                    writer.add_scalar("Test_Polarity_Loss", test_loss_parts["polarity"], epoch)
            else:
                split_loss_parts = [("val", val_loss_parts)]
                if test_loss_parts is not None:
                    split_loss_parts.append(("test", test_loss_parts))
                for split_name, parts in split_loss_parts:
                    writer.add_scalar(f"epoch/{split_name}_reg_loss", parts["reg"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_cls7_loss", parts["cls7"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_weighted_cls7_loss", parts["weighted_cls7"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_sign_loss", parts["sign"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_weighted_sign_loss", parts["weighted_sign"], epoch)
                    writer.add_scalar(
                        f"epoch/{split_name}_zero_sign_margin_loss",
                        parts["zero_sign_margin"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_weighted_zero_sign_margin_loss",
                        parts["weighted_zero_sign_margin"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_reg_cls_mag_consistency_loss",
                        parts["reg_cls_mag_consistency"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_weighted_reg_cls_mag_consistency_loss",
                        parts["weighted_reg_cls_mag_consistency"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_reg_cls_mag_consistency_selected_count",
                        parts["reg_cls_mag_consistency_selected_count"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_reg_cls_mag_consistency_coverage",
                        parts["reg_cls_mag_consistency_coverage"],
                        epoch,
                    )
                    writer.add_scalar(
                        f"epoch/{split_name}_reg_cls_mag_consistency_mean_abs_gap",
                        parts["reg_cls_mag_consistency_mean_abs_gap"],
                        epoch,
                    )
                    writer.add_scalar(f"epoch/{split_name}_score_sign_loss", parts["score_sign"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_weighted_score_sign_loss", parts["weighted_score_sign"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_oacr_loss", parts["oacr"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_weighted_oacr_loss", parts["weighted_oacr"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_temporal_contrast_loss", parts["temporal_contrast"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_weighted_temporal_contrast_loss", parts["weighted_temporal_contrast"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_temporal_valid_anchors", parts["temporal_valid_anchors"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_temporal_skipped_anchors", parts["temporal_skipped_anchors"], epoch)
                    writer.add_scalar(f"epoch/{split_name}_temporal_positive_pairs", parts["temporal_positive_pairs"], epoch)
                    for pred_name, metric_map in parts["metrics"].items():
                        if isinstance(metric_map, dict):
                            for metric_name, metric_value in metric_map.items():
                                if isinstance(metric_value, (int, float)):
                                    writer.add_scalar(f"epoch/{split_name}_{pred_name}_{metric_name}", metric_value, epoch)
                    for diag_name, diag_value in parts["diagnostics"].items():
                        if isinstance(diag_value, dict):
                            for stat_name, stat_value in diag_value.items():
                                writer.add_scalar(f"epoch/{split_name}_{diag_name}_{stat_name}", stat_value, epoch)
                        else:
                            writer.add_scalar(f"epoch/{split_name}_{diag_name}", diag_value, epoch)
            if gating_steps > 0:
                writer.add_scalar("epoch/Gating_Entropy", gating_entropy_sum / gating_steps, epoch)
                writer.add_scalar("epoch/Expert_Load_Variance", gating_var_sum / gating_steps, epoch)
            if grad_norm_steps > 0:
                writer.add_scalar("Train_Grad_Norm", grad_norm_sum / grad_norm_steps, epoch)
            for i, p in enumerate(val_prec):
                writer.add_scalar(f"epoch/val_precision_class_{i}", p, epoch)
            for i, r in enumerate(val_rec):
                writer.add_scalar(f"epoch/val_recall_class_{i}", r, epoch)
            if train_group_dispatch:
                for dispatch_name, dispatch_share in _flatten_msoe_group_dispatch_shares(train_group_dispatch).items():
                    writer.add_scalar(f"epoch/msoe_group_dispatch_share/{dispatch_name}", dispatch_share, epoch)
            if lambda_state is not None:
                if lambda_state.get("mode") in {"adaptive", "attention"}:
                    for idx, name in enumerate(["main", "prev", "next"]):
                        writer.add_scalar(f"epoch/vit_context_base_lambda_{name}", lambda_state["base_lambda"][idx], epoch)
                    for stat_name in ["mean", "std", "min", "max"]:
                        values = lambda_state.get(f"effective_lambda_{stat_name}")
                        if values is not None:
                            for idx, name in enumerate(["main", "prev", "next"]):
                                writer.add_scalar(f"epoch/vit_context_lambda_{stat_name}_{name}", values[idx], epoch)
                        attn_values = lambda_state.get(f"attention_center_prob_{stat_name}")
                        if attn_values is not None:
                            for idx, name in enumerate(["center", "prev", "next"]):
                                writer.add_scalar(f"epoch/vit_context_attention_center_prob_{stat_name}_{name}", attn_values[idx], epoch)
                    attention_matrix_mean = lambda_state.get("attention_matrix_mean")
                    if attention_matrix_mean is not None:
                        frame_names = ["center", "prev", "next"]
                        for row_idx, row_name in enumerate(frame_names):
                            for col_idx, col_name in enumerate(frame_names):
                                writer.add_scalar(
                                    f"epoch/vit_context_attention_matrix_mean_{row_name}_{col_name}",
                                    attention_matrix_mean[row_idx][col_idx],
                                    epoch,
                                )
                    if "attention_center_entropy_mean" in lambda_state:
                        writer.add_scalar(
                            "epoch/vit_context_attention_center_entropy_mean",
                            lambda_state["attention_center_entropy_mean"],
                            epoch,
                        )
                    if "effective_lambda_count" in lambda_state:
                        writer.add_scalar("epoch/vit_context_lambda_count", lambda_state["effective_lambda_count"], epoch)
                else:
                    writer.add_scalar("epoch/vit_context_lambda_main", lambda_state["effective_lambda"][0], epoch)
                    writer.add_scalar("epoch/vit_context_lambda_prev", lambda_state["effective_lambda"][1], epoch)
                    writer.add_scalar("epoch/vit_context_lambda_next", lambda_state["effective_lambda"][2], epoch)
                writer.add_scalar("epoch/vit_context_lambda_logit_main", lambda_state["raw_logits"][0], epoch)
                writer.add_scalar("epoch/vit_context_lambda_logit_prev", lambda_state["raw_logits"][1], epoch)
                writer.add_scalar("epoch/vit_context_lambda_logit_next", lambda_state["raw_logits"][2], epoch)

        val_metrics = {
            "acc7": val_final_acc7,
            "mae": val_mae,
            "acc2": val_final_acc2,
        }
        if test_loss_parts is not None:
            test_metrics = {
                "acc7": test_final_acc7,
                "mae": test_mae,
                "acc2": test_final_acc2,
            }
        if args.output_head_mode == "signed_reg_cls7":
            metric_targets = [(val_loss_parts["metrics"], val_metrics)]
            if test_loss_parts is not None:
                metric_targets.append((test_loss_parts["metrics"], test_metrics))
            for metrics_obj, target in metric_targets:
                target.update({
                    "cls7_acc": float(metrics_obj["cls7_acc"]),
                    "cls7_macro_f1": float(metrics_obj["cls7_macro_f1"]),
                    "reg_mae": float(metrics_obj["reg"]["mae"]),
                    "reg_acc7": float(metrics_obj["reg"]["acc7"]),
                    "reg_acc2": float(metrics_obj["reg"]["acc2"]),
                    "cls_expected_mae": float(metrics_obj["cls_expected"]["mae"]),
                    "cls_expected_acc7": float(metrics_obj["cls_expected"]["acc7"]),
                    "cls_expected_acc2": float(metrics_obj["cls_expected"]["acc2"]),
                    "final_mae": float(metrics_obj["final"]["mae"]),
                    "final_acc7": float(metrics_obj["final"]["acc7"]),
                    "final_acc2": float(metrics_obj["final"]["acc2"]),
                })
        promoted = _update_best_checkpoints(
            model=model,
            save_dir=args.save_dir,
            epoch=epoch + 1,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            best_records=best_checkpoint_records,
            args=args,
        )
        if args.save_latest_checkpoint:
            latest_ckpt_dir = os.path.join(args.save_dir, "recovery")
            os.makedirs(latest_ckpt_dir, exist_ok=True)
            latest_ckpt_path = os.path.join(latest_ckpt_dir, "latest_model.pth")
            latest_tmp_path = latest_ckpt_path + ".tmp"
            torch.save(model.state_dict(), latest_tmp_path)
            os.replace(latest_tmp_path, latest_ckpt_path)
            _write_checkpoint_hparams(
                latest_ckpt_path, args, checkpoint_epoch=epoch + 1
            )
        if args.save_epoch_checkpoints:
            epoch_ckpt_dir = os.path.join(args.save_dir, "epoch_checkpoints")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            epoch_ckpt_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoch + 1:03d}.pth")
            torch.save(model.state_dict(), epoch_ckpt_path)
            _write_checkpoint_hparams(epoch_ckpt_path, args, checkpoint_epoch=epoch + 1)
            _record_checkpoint_snapshot(epoch + 1, val_metrics, test_metrics, epoch_ckpt_path)
        for metric_name, candidate in promoted:
            if metric_name == "acc7":
                print(
                    f"Updated best {selection_split}-acc7 checkpoint "
                    f"epoch={candidate['epoch']} "
                    f"{selection_split}_acc7={candidate[f'{selection_split}_acc7']:.4f} "
                    f"{selection_split}_mae={candidate[f'{selection_split}_mae']:.4f}"
                )
            else:
                print(
                    f"Updated best {selection_split}-mae checkpoint "
                    f"epoch={candidate['epoch']} "
                    f"{selection_split}_mae={candidate[f'{selection_split}_mae']:.4f} "
                    f"{selection_split}_acc7={candidate[f'{selection_split}_acc7']:.4f}"
                )

        if scheduler is not None:
            if args.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if args.early_stop_metric == "val_loss":
            stop_value = val_loss
        elif args.early_stop_metric == "val_macro_f1":
            stop_value = val_macro_f1
        elif args.early_stop_metric == "val_mae":
            stop_value = val_mae
        else:
            stop_value = val_acc
        if early_stopper.step(stop_value):
            print(f"Early stopping at epoch {epoch+1}")
            break

    checkpoint_summary = _write_checkpoint_selection_summary(args.save_dir, best_checkpoint_records, args=args)
    _upsert_total_table_rows(args, checkpoint_summary)
    best_acc7 = checkpoint_summary.get(f"best_{selection_split}_acc7")
    best_mae = checkpoint_summary.get(f"best_{selection_split}_mae")
    if best_acc7 is not None:
        print(
            f"Saved final {selection_split}-selected acc7 checkpoint: epoch={best_acc7['epoch']} "
            f"{selection_split}_acc7={best_acc7[f'{selection_split}_acc7']:.4f} "
            f"{selection_split}_mae={best_acc7[f'{selection_split}_mae']:.4f}"
        )
    if best_mae is not None:
        print(
            f"Saved final {selection_split}-selected mae checkpoint: epoch={best_mae['epoch']} "
            f"{selection_split}_mae={best_mae[f'{selection_split}_mae']:.4f} "
            f"{selection_split}_acc7={best_mae[f'{selection_split}_acc7']:.4f}"
        )
    if writer is not None:
        writer.add_text(
            "checkpoint_selection/final",
            json.dumps(checkpoint_summary, ensure_ascii=False, indent=2),
            0,
        )

    if writer is not None:
        writer.flush()
        writer.close()

    for h in gating_handles:
        h.remove()


def validate_signed_reg_cls7(model, loader, device, eff_alpha, args, contrast_head=None):
    model.eval()
    if contrast_head is not None:
        contrast_head.eval()
    reg_scores = []
    cls_expected_scores = []
    final_scores = []
    raw_valences = []
    cls7_preds_all = []
    cls7_labels_all = []
    total_loss = 0.0
    total_loss_reg = 0.0
    total_loss_cls7 = 0.0
    total_loss_sign = 0.0
    total_loss_score_sign = 0.0
    total_loss_hier_sign = 0.0
    total_loss_hier_mag = 0.0
    total_loss_sign_marginal = 0.0
    total_loss_signed_neutral_band = 0.0
    total_loss_zero_sign_margin = 0.0
    total_loss_reg_cls_mag_consistency = 0.0
    total_reg_cls_mag_consistency_selected = 0
    total_reg_cls_mag_consistency_positive_selected = 0
    total_reg_cls_mag_consistency_negative_selected = 0
    total_reg_cls_mag_consistency_balanced_batches = 0
    total_reg_cls_mag_consistency_fallback_batches = 0
    total_reg_cls_mag_consistency_candidates = 0
    total_reg_cls_mag_consistency_abs_gap = 0.0
    total_loss_cumulative = 0.0
    total_loss_oacr = 0.0
    total_loss_temporal_contrast = 0.0
    total_loss_tcif_context = 0.0
    total_loss_tcif_context_reg = 0.0
    total_loss_tcif_context_cls7 = 0.0
    total_tcif_context_valid = 0
    total_loss_tcif_transition_gate = 0.0
    total_tcif_transition_gate_valid = 0
    total_tcif_transition_gate_conflicts = 0
    total_tcif_transition_gate_target = 0.0
    total_tcif_regression_gate = 0.0
    total_tcif_ordinal_gate = 0.0
    total_temporal_valid_anchors = 0
    total_temporal_skipped_anchors = 0
    total_temporal_positive_pairs = 0
    total_tc = 0.0
    entropy_sum = 0.0
    confidence_correct_sum = 0.0
    confidence_correct_n = 0
    confidence_wrong_sum = 0.0
    confidence_wrong_n = 0
    val_batches = 0
    diag_tensors = {"y_reg": [], "y_cls_expected": [], "y_final": []}

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            raw_valence = batch['raw_valence'].to(device)
            audio_values = batch["audio_values"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)

            outputs = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                **_embedding_cache_kwargs(batch, device, use_text_cache=args.use_text_cache),
                **_tcif_context_kwargs(batch, device, use_text_cache=args.use_text_cache),
            )
            y_reg = outputs["y_reg"]
            cls7_logits = outputs["cls7_logits"]
            sign_logits = outputs.get("sign_logits", None)
            tc_loss = outputs["tc_loss"]
            loss_reg = compute_regression_loss(y_reg, raw_valence, args.reg_loss_type)
            loss_cls7, cls7_target = _compute_cls7_loss(
                cls7_logits,
                raw_valence,
                args.cls7_loss_type,
                args.cls7_soft_tau,
                args.cls7_low_abs_sample_weight_threshold,
                args.cls7_low_abs_sample_weight,
                getattr(args, "_cls7_class_weights_t", None),
            )
            sign_loss_logits = sign_logits if sign_logits is not None else y_reg
            loss_sign = _compute_sign_aux_loss(sign_loss_logits, raw_valence, args) if args.sign_aux_weight > 0 else y_reg.new_tensor(0.0)
            loss_hier_sign, loss_hier_mag = _compute_hier_sign_mag_losses(outputs, raw_valence)
            loss_sign_marginal = (
                _compute_sign_marginal_loss(cls7_logits, raw_valence, args)
                if args.sign_marginal_loss_weight > 0
                else y_reg.new_tensor(0.0)
            )
            loss_signed_neutral_band = (
                _compute_signed_neutral_band_loss(outputs, raw_valence, args)
                if args.signed_neutral_band_weight > 0
                else y_reg.new_tensor(0.0)
            )
            loss_zero_sign_margin = (
                _compute_zero_sign_margin_loss(outputs, raw_valence, args)
                if args.zero_sign_margin_weight > 0
                else y_reg.new_tensor(0.0)
            )
            if args.reg_cls_mag_consistency_weight > 0:
                loss_reg_cls_mag_consistency, reg_cls_mag_consistency_stats = (
                    _compute_reg_cls_mag_consistency_loss(outputs, args)
                )
            else:
                loss_reg_cls_mag_consistency = y_reg.new_tensor(0.0)
                reg_cls_mag_consistency_stats = _empty_reg_cls_mag_consistency_stats(y_reg.numel())
            loss_cumulative = (
                _compute_cumulative_loss(outputs, raw_valence)
                if args.cumulative_loss_weight > 0
                else y_reg.new_tensor(0.0)
            )
            score_sign_scores = outputs.get("extras", {}).get("y_final", y_reg)
            loss_score_sign = (
                _compute_score_sign_aux_loss(score_sign_scores, raw_valence, args)
                if args.score_sign_aux_weight > 0
                else y_reg.new_tensor(0.0)
            )
            loss_oacr = _compute_oacr_loss(outputs, raw_valence, cls7_target, args, contrast_head)
            loss_temporal_contrast, temporal_contrast_stats = _compute_temporal_contrast_loss(outputs, batch, args)
            if args.tcif_context_aux_weight > 0:
                (
                    loss_tcif_context,
                    loss_tcif_context_reg,
                    loss_tcif_context_cls7,
                    tcif_context_valid_count,
                ) = _compute_tcif_context_aux_loss(outputs, raw_valence, args)
            else:
                loss_tcif_context = y_reg.new_tensor(0.0)
                loss_tcif_context_reg = y_reg.new_tensor(0.0)
                loss_tcif_context_cls7 = y_reg.new_tensor(0.0)
                tcif_context_valid_count = 0
            if args.tcif_transition_gate_loss_weight > 0:
                (
                    loss_tcif_transition_gate,
                    tcif_transition_gate_stats,
                ) = _compute_tcif_transition_gate_loss(
                    outputs,
                    batch,
                    raw_valence,
                    args,
                )
            else:
                loss_tcif_transition_gate = y_reg.new_tensor(0.0)
                tcif_transition_gate_stats = {
                    "valid_count": 0,
                    "sign_conflict_count": 0,
                    "target_mean": 0.0,
                    "regression_gate_mean": 0.0,
                    "ordinal_gate_mean": 0.0,
                }
            batch_total = (
                loss_reg
                + args.cls7_loss_weight * loss_cls7
                + args.sign_aux_weight * loss_sign
                + _sign_struct_scale(args) * args.hier_sign_loss_weight * loss_hier_sign
                + _sign_struct_scale(args) * args.hier_mag_loss_weight * loss_hier_mag
                + _sign_struct_scale(args) * args.sign_marginal_loss_weight * loss_sign_marginal
                + _sign_struct_scale(args) * args.signed_neutral_band_weight * loss_signed_neutral_band
                + _sign_struct_scale(args) * args.zero_sign_margin_weight * loss_zero_sign_margin
                + _sign_struct_scale(args)
                * args.reg_cls_mag_consistency_weight
                * loss_reg_cls_mag_consistency
                + _sign_struct_scale(args) * args.cumulative_loss_weight * loss_cumulative
                + args.score_sign_aux_weight * loss_score_sign
                + args.lambda_oacr * loss_oacr
                + args.temporal_contrast_weight * loss_temporal_contrast
                + args.tcif_context_aux_weight * loss_tcif_context
                + args.tcif_transition_gate_loss_weight
                * loss_tcif_transition_gate
                + eff_alpha * tc_loss
            )

            total_loss += batch_total.item()
            total_loss_reg += loss_reg.item()
            total_loss_cls7 += loss_cls7.item()
            total_loss_sign += loss_sign.item()
            total_loss_score_sign += loss_score_sign.item()
            total_loss_hier_sign += loss_hier_sign.item()
            total_loss_hier_mag += loss_hier_mag.item()
            total_loss_sign_marginal += loss_sign_marginal.item()
            total_loss_signed_neutral_band += loss_signed_neutral_band.item()
            total_loss_zero_sign_margin += loss_zero_sign_margin.item()
            total_loss_reg_cls_mag_consistency += loss_reg_cls_mag_consistency.item()
            total_reg_cls_mag_consistency_selected += int(
                reg_cls_mag_consistency_stats.get("selected_count", 0)
            )
            total_reg_cls_mag_consistency_positive_selected += int(
                reg_cls_mag_consistency_stats.get("positive_selected_count", 0)
            )
            total_reg_cls_mag_consistency_negative_selected += int(
                reg_cls_mag_consistency_stats.get("negative_selected_count", 0)
            )
            total_reg_cls_mag_consistency_balanced_batches += int(
                reg_cls_mag_consistency_stats.get(
                    "sign_balanced_reduction_applied", 0
                )
            )
            total_reg_cls_mag_consistency_fallback_batches += int(
                reg_cls_mag_consistency_stats.get("single_branch_fallback", 0)
            )
            total_reg_cls_mag_consistency_candidates += int(
                reg_cls_mag_consistency_stats.get("candidate_count", 0)
            )
            total_reg_cls_mag_consistency_abs_gap += float(
                reg_cls_mag_consistency_stats.get("abs_gap_sum", 0.0)
            )
            total_loss_cumulative += loss_cumulative.item()
            total_loss_oacr += loss_oacr.item()
            total_loss_temporal_contrast += loss_temporal_contrast.item()
            total_loss_tcif_context += loss_tcif_context.item()
            total_loss_tcif_context_reg += loss_tcif_context_reg.item()
            total_loss_tcif_context_cls7 += loss_tcif_context_cls7.item()
            total_tcif_context_valid += int(tcif_context_valid_count)
            total_loss_tcif_transition_gate += loss_tcif_transition_gate.item()
            gate_valid = int(tcif_transition_gate_stats.get("valid_count", 0))
            total_tcif_transition_gate_valid += gate_valid
            total_tcif_transition_gate_conflicts += int(
                tcif_transition_gate_stats.get("sign_conflict_count", 0)
            )
            total_tcif_transition_gate_target += gate_valid * float(
                tcif_transition_gate_stats.get("target_mean", 0.0)
            )
            total_tcif_regression_gate += gate_valid * float(
                tcif_transition_gate_stats.get("regression_gate_mean", 0.0)
            )
            total_tcif_ordinal_gate += gate_valid * float(
                tcif_transition_gate_stats.get("ordinal_gate_mean", 0.0)
            )
            total_temporal_valid_anchors += int(temporal_contrast_stats.get("valid_anchor_count", 0))
            total_temporal_skipped_anchors += int(temporal_contrast_stats.get("skipped_anchor_count", 0))
            total_temporal_positive_pairs += int(temporal_contrast_stats.get("positive_pair_count", 0))
            total_tc += tc_loss.item()
            val_batches += 1

            y_reg_eval, y_cls_expected, y_final = _signed_prediction_tensors(outputs, args)
            cls7_preds = torch.argmax(cls7_logits, dim=1)
            probs = torch.softmax(cls7_logits, dim=-1)
            conf = probs.max(dim=1).values
            correct = cls7_preds == cls7_target
            if bool(correct.any().item()):
                confidence_correct_sum += float(conf[correct].sum().item())
                confidence_correct_n += int(correct.sum().item())
            if bool((~correct).any().item()):
                confidence_wrong_sum += float(conf[~correct].sum().item())
                confidence_wrong_n += int((~correct).sum().item())
            entropy_sum += _cls7_entropy(cls7_logits)

            reg_scores.extend(y_reg_eval.detach().cpu().numpy().tolist())
            cls_expected_scores.extend(y_cls_expected.detach().cpu().numpy().tolist())
            final_scores.extend(y_final.detach().cpu().numpy().tolist())
            raw_valences.extend(raw_valence.detach().cpu().numpy().tolist())
            cls7_preds_all.extend(cls7_preds.detach().cpu().numpy().tolist())
            cls7_labels_all.extend(cls7_target.detach().cpu().numpy().tolist())
            diag_tensors["y_reg"].append(y_reg_eval.detach().cpu())
            diag_tensors["y_cls_expected"].append(y_cls_expected.detach().cpu())
            diag_tensors["y_final"].append(y_final.detach().cpu())

    metrics = summarize_signed_metrics(
        reg_scores,
        cls_expected_scores,
        final_scores,
        raw_valences,
        cls7_labels_all,
        cls7_preds_all,
    )
    denom = max(1, val_batches)
    diagnostics = {}
    for name, chunks in diag_tensors.items():
        diagnostics[name] = _tensor_stats(torch.cat(chunks, dim=0)) if chunks else _tensor_stats(torch.empty(0))
    diagnostics["cls7_entropy"] = entropy_sum / denom
    diagnostics["cls7_conf_correct"] = confidence_correct_sum / max(1, confidence_correct_n)
    diagnostics["cls7_conf_wrong"] = confidence_wrong_sum / max(1, confidence_wrong_n)
    loss_parts = {
        "total": total_loss / denom,
        "reg": total_loss_reg / denom,
        "cls7": total_loss_cls7 / denom,
        "weighted_cls7": args.cls7_loss_weight * (total_loss_cls7 / denom),
        "sign": total_loss_sign / denom,
        "weighted_sign": args.sign_aux_weight * (total_loss_sign / denom),
        "score_sign": total_loss_score_sign / denom,
        "weighted_score_sign": args.score_sign_aux_weight * (total_loss_score_sign / denom),
        "hier_sign": total_loss_hier_sign / denom,
        "weighted_hier_sign": _sign_struct_scale(args) * args.hier_sign_loss_weight * (total_loss_hier_sign / denom),
        "hier_mag": total_loss_hier_mag / denom,
        "weighted_hier_mag": _sign_struct_scale(args) * args.hier_mag_loss_weight * (total_loss_hier_mag / denom),
        "sign_marginal": total_loss_sign_marginal / denom,
        "weighted_sign_marginal": _sign_struct_scale(args) * args.sign_marginal_loss_weight * (total_loss_sign_marginal / denom),
        "signed_neutral_band": total_loss_signed_neutral_band / denom,
        "weighted_signed_neutral_band": _sign_struct_scale(args) * args.signed_neutral_band_weight * (total_loss_signed_neutral_band / denom),
        "zero_sign_margin": total_loss_zero_sign_margin / denom,
        "weighted_zero_sign_margin": _sign_struct_scale(args)
        * args.zero_sign_margin_weight
        * (total_loss_zero_sign_margin / denom),
        "zero_sign_margin_effective_weight": _sign_struct_scale(args)
        * args.zero_sign_margin_weight,
        "reg_cls_mag_consistency": total_loss_reg_cls_mag_consistency / denom,
        "weighted_reg_cls_mag_consistency": _sign_struct_scale(args)
        * args.reg_cls_mag_consistency_weight
        * (total_loss_reg_cls_mag_consistency / denom),
        "reg_cls_mag_consistency_effective_weight": _sign_struct_scale(args)
        * args.reg_cls_mag_consistency_weight,
        "reg_cls_mag_consistency_selected_count": total_reg_cls_mag_consistency_selected,
        "reg_cls_mag_consistency_positive_selected_count": (
            total_reg_cls_mag_consistency_positive_selected
        ),
        "reg_cls_mag_consistency_negative_selected_count": (
            total_reg_cls_mag_consistency_negative_selected
        ),
        "reg_cls_mag_consistency_balanced_batch_count": (
            total_reg_cls_mag_consistency_balanced_batches
        ),
        "reg_cls_mag_consistency_fallback_batch_count": (
            total_reg_cls_mag_consistency_fallback_batches
        ),
        "reg_cls_mag_consistency_candidate_count": total_reg_cls_mag_consistency_candidates,
        "reg_cls_mag_consistency_coverage": total_reg_cls_mag_consistency_selected
        / max(1, total_reg_cls_mag_consistency_candidates),
        "reg_cls_mag_consistency_mean_abs_gap": total_reg_cls_mag_consistency_abs_gap
        / max(1, total_reg_cls_mag_consistency_selected),
        "cumulative": total_loss_cumulative / denom,
        "weighted_cumulative": _sign_struct_scale(args) * args.cumulative_loss_weight * (total_loss_cumulative / denom),
        "sign_struct_scale": _sign_struct_scale(args),
        "oacr": total_loss_oacr / denom,
        "weighted_oacr": args.lambda_oacr * (total_loss_oacr / denom),
        "temporal_contrast": total_loss_temporal_contrast / denom,
        "weighted_temporal_contrast": args.temporal_contrast_weight * (total_loss_temporal_contrast / denom),
        "tcif_context": total_loss_tcif_context / denom,
        "weighted_tcif_context": args.tcif_context_aux_weight * (total_loss_tcif_context / denom),
        "tcif_context_reg": total_loss_tcif_context_reg / denom,
        "tcif_context_cls7": total_loss_tcif_context_cls7 / denom,
        "tcif_context_valid": total_tcif_context_valid,
        "tcif_transition_gate": total_loss_tcif_transition_gate / denom,
        "weighted_tcif_transition_gate": args.tcif_transition_gate_loss_weight
        * (total_loss_tcif_transition_gate / denom),
        "tcif_transition_gate_valid": total_tcif_transition_gate_valid,
        "tcif_transition_gate_conflicts": total_tcif_transition_gate_conflicts,
        "tcif_transition_gate_target": total_tcif_transition_gate_target
        / max(1, total_tcif_transition_gate_valid),
        "tcif_regression_gate": total_tcif_regression_gate
        / max(1, total_tcif_transition_gate_valid),
        "tcif_ordinal_gate": total_tcif_ordinal_gate
        / max(1, total_tcif_transition_gate_valid),
        "temporal_valid_anchors": total_temporal_valid_anchors,
        "temporal_skipped_anchors": total_temporal_skipped_anchors,
        "temporal_positive_pairs": total_temporal_positive_pairs,
        "tc": total_tc / denom,
        "weighted_tc": eff_alpha * (total_tc / denom),
        "metrics": metrics,
        "diagnostics": diagnostics,
    }
    final = metrics["final"]
    return (
        metrics["cls7_acc"],
        loss_parts["total"],
        metrics["cls7_macro_f1"],
        metrics["cls7_precision"],
        metrics["cls7_recall"],
        final["rmse"],
        final["mae"],
        final["pearson"],
        final["spearman"],
        final["acc2"],
        final["acc7"],
        final["macro_f1_7"],
        loss_parts,
    )


def validate(model, loader, device, criterion_polarity, num_polarity_classes, lambda_polarity, eff_alpha, intensity_loss_type, args=None, contrast_head=None):
    if getattr(model, "output_head_mode", "legacy") == "signed_reg_cls7":
        if args is None:
            raise ValueError("args is required for signed_reg_cls7 validation")
        return validate_signed_reg_cls7(model, loader, device, eff_alpha, args, contrast_head=contrast_head)

    model.eval()
    preds_all = []
    labels_all = []
    intensity_preds = []
    intensity_targets = []
    final_scores = []
    raw_valences = []
    total_loss = 0.0
    total_loss_intensity = 0.0
    total_loss_polarity = 0.0
    total_tc = 0.0
    val_batches = 0

    with torch.no_grad():
        for step_idx, batch in enumerate(loader):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intensity = batch['intensity'].to(device)
            polarity = batch['polarity'].to(device)
            raw_valence = batch['raw_valence'].to(device)
            audio_values = batch["audio_values"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)

            out_intensity, out_polarity, tc_loss = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                **_embedding_cache_kwargs(batch, device, use_text_cache=_resolve_use_text_cache(args)),
                **_tcif_context_kwargs(batch, device, use_text_cache=_resolve_use_text_cache(args)),
            )
            loss_intensity = compute_intensity_loss(out_intensity, intensity, intensity_loss_type)
            loss_polarity = compute_emotion_loss(out_polarity, polarity, criterion_polarity, {"loss_type": "polarity"})
            total_loss += (loss_intensity + lambda_polarity * loss_polarity + eff_alpha * tc_loss).item()
            total_loss_intensity += loss_intensity.item()
            total_loss_polarity += loss_polarity.item()
            total_tc += tc_loss.item()
            preds = torch.argmax(out_polarity, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(polarity.cpu().numpy())
            intensity_preds.extend(out_intensity.detach().cpu().numpy().tolist())
            intensity_targets.extend(intensity.detach().cpu().numpy().tolist())
            signs = torch.where(preds == 0, -torch.ones_like(out_intensity), torch.ones_like(out_intensity))
            final_scores.extend((out_intensity.detach() * signs).cpu().numpy().tolist())
            raw_valences.extend(raw_valence.detach().cpu().numpy().tolist())
            val_batches += 1

    acc, macro_f1, per_class_precision, per_class_recall = metrics_from_lists(labels_all, preds_all, num_polarity_classes)
    
    # FIX: Calculate regression metrics on actual valence (signed), not just intensity (absolute)
    # previously: rmse, mae, pearson, spearman = regression_metrics(intensity_preds, intensity_targets)
    rmse, mae, pearson, spearman = regression_metrics(final_scores, raw_valences)

    pred_7 = [_valence_to_7class(s) for s in final_scores]
    gt_7 = [_valence_to_7class(v) for v in raw_valences]
    acc7, macro_f1_7, _, _ = metrics_from_lists(gt_7, pred_7, 7)
    pred_2 = [_valence_to_binary(s) for s in final_scores]
    gt_2 = [_valence_to_binary(v) for v in raw_valences]
    acc2 = accuracy_from_lists(gt_2, pred_2)
    denom = max(1, val_batches)
    avg_intensity = total_loss_intensity / denom
    avg_polarity = total_loss_polarity / denom
    avg_tc = total_tc / denom
    loss_parts = {
        "total": total_loss / denom,
        "intensity": avg_intensity,
        "polarity": avg_polarity,
        "weighted_polarity": lambda_polarity * avg_polarity,
        "tc": avg_tc,
        "weighted_tc": eff_alpha * avg_tc,
    }
    return acc, loss_parts["total"], macro_f1, per_class_precision, per_class_recall, rmse, mae, pearson, spearman, acc2, acc7, macro_f1_7, loss_parts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", default="", help="JSON defaults; explicit CLI options take precedence")
    parser.add_argument("--dataset", type=str, default="csv", choices=["csv", "cmumosei", "cmumosi"])
    parser.add_argument("--dataset_root", type=str, default="/path/to/datasets/cmumosei-process-complete-20260408")
    parser.add_argument("--embedding_cache_root", type=str, default="", help="Optional root containing train/val/test embedding cache shards.")
    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="")
    parser.add_argument("--save_dir", type=str, default=os.path.join(STRUCTV7_REMOTE_ROOT, "train", "checkpoints"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=11)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--router_lr", type=float, default=2e-4)
    parser.add_argument(
        "--tcif_transition_gate_lr",
        type=float,
        default=None,
        help=(
            "Learning rate for TCIF regression/ordinal transition gates. "
            "Defaults to --lr for exact compatibility with historical grouping."
        ),
    )
    parser.add_argument("--router_temperature", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_experts_msoe", type=int, default=16)
    parser.add_argument("--num_experts_mtoe", type=int, default=32)
    parser.add_argument("--num_shared_experts", type=int, default=2)
    parser.add_argument("--num_text_specific_experts", type=int, default=4)
    parser.add_argument("--num_audio_specific_experts", type=int, default=4)
    parser.add_argument("--num_vision_specific_experts", type=int, default=4)
    parser.add_argument("--num_temporal_contrast_experts", type=int, default=0)
    parser.add_argument(
        "--enable_text_shared_experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否允许文本模态访问 shared experts",
    )
    parser.add_argument("--enable_temporal_contrast_experts", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--enable_temporal_contrast_loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable temporal contrastive loss without adding temporal contrast experts.",
    )
    parser.add_argument("--temporal_embedding_dim", type=int, default=128)
    parser.add_argument(
        "--temporal_detach_task2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Stop temporal-contrast gradients at task-2 features while preserving forward values.",
    )
    parser.add_argument("--temporal_contrast_weight", type=float, default=0.03)
    parser.add_argument("--temporal_contrast_temperature", type=float, default=0.07)
    parser.add_argument("--temporal_decay_tau", type=float, default=1.0)
    parser.add_argument("--temporal_positive_radius", type=float, default=1.0)
    parser.add_argument("--temporal_weak_positive_radius", type=float, default=4.0)
    parser.add_argument("--temporal_min_positive_weight", type=float, default=0.2)
    parser.add_argument("--temporal_kernel", type=str, default="legacy_exp", choices=TEMPORAL_KERNEL_CHOICES)
    parser.add_argument("--temporal_label_gate_beta", type=float, default=1.0)
    parser.add_argument("--temporal_zero_bridge_weight", type=float, default=0.25)
    parser.add_argument("--temporal_batch_mode", type=str, default="shuffle", choices=["shuffle", "group_window"])
    parser.add_argument("--temporal_batch_window", type=float, default=4.0)
    parser.add_argument(
        "--target_sampler",
        type=str,
        default="none",
        choices=["none", "binary_balance"],
        help="Optional training sampler. binary_balance samples negative and nonnegative raw valence with equal expected frequency.",
    )
    parser.add_argument("--alpha", type=float, default=0.03, help="Base weight for TC Loss (cosine-decayed with warmup).")
    parser.add_argument(
        "--enable_tcif",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable masked-center Temporal Context Innovation Filtering.",
    )
    parser.add_argument(
        "--tcif_context_radius",
        type=int,
        default=1,
        help="Number of neighboring clips per temporal direction.",
    )
    parser.add_argument(
        "--tcif_context_mode",
        type=str,
        default="neighbors",
        choices=["neighbors", "shuffled"],
        help="Use true same-video neighbors or a cross-video counterfactual control.",
    )
    parser.add_argument("--tcif_latent_dim", type=int, default=128)
    parser.add_argument("--tcif_ablation", choices=TCIF_ABLATIONS, default="full",
                        help="Paper ablation; saved with each checkpoint for reconstruction")
    parser.add_argument(
        "--tcif_output_mode",
        type=str,
        default="posterior",
        choices=["local", "context", "posterior"],
    )
    parser.add_argument("--tcif_context_temperature", type=float, default=1.0)
    parser.add_argument(
        "--tcif_enable_transition_gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Route between context-posterior and local experts with a learned continuation gate.",
    )
    parser.add_argument("--tcif_transition_gate_hidden_dim", type=int, default=64)
    parser.add_argument("--tcif_transition_gate_init_bias", type=float, default=2.0)
    parser.add_argument("--tcif_transition_gate_loss_weight", type=float, default=0.0)
    parser.add_argument("--tcif_transition_gate_tau", type=float, default=0.75)
    parser.add_argument(
        "--tcif_transition_gate_conflict_target",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tcif_context_aux_weight",
        type=float,
        default=0.0,
        help="Weight for the masked-center context-only prediction objective.",
    )
    parser.add_argument("--tcif_context_aux_cls7_weight", type=float, default=0.5)
    parser.add_argument("--tokenizer_path", type=str, default="/path/to/models/AI-ModelScope_roberta-base")
    parser.add_argument(
        "--bert_backbone_path",
        type=str,
        default="/path/to/models/AI-ModelScope_roberta-base",
        help="Local Hugging Face BERT/RoBERTa model directory.",
    )
    parser.add_argument("--vision_backbone_type", type=str, default="vit", choices=["resnet18_temporal", "vit"])
    parser.add_argument("--vision_backbone_path", type=str, default="/path/to/models/resnet-18")
    parser.add_argument("--vit_backbone_path", type=str, default="/path/to/models/vit-base-patch16-224-in21k")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--frame_policy", type=str, default="middle", choices=["first", "middle", "last"])
    parser.add_argument("--hubert_model_path", type=str, default="/path/to/models/hubert-base-ls960")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--vit_context_ratio", type=float, default=0.0)
    parser.add_argument("--vit_context_lambda_mode", type=str, default="static", choices=["static", "adaptive", "attention"])
    parser.add_argument("--vit_context_lambda_init", type=str, default="")
    parser.add_argument("--vit_context_attention_dim", type=int, default=64)
    parser.add_argument("--audio_sample_rate", type=int, default=16000)
    parser.add_argument("--audio_max_seconds", type=float, default=6.0)
    parser.add_argument("--audio_frame_ms", type=float, default=25.0)
    parser.add_argument("--audio_hop_ms", type=float, default=10.0)
    parser.add_argument("--disable_audio_denoise", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_tensorboard", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default=os.path.join(STRUCTV7_REMOTE_ROOT, "train", "tensorboard"))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--experiment_git_commit", type=str, default="")
    parser.add_argument("--protocol_version", type=str, default="legacy")
    parser.add_argument("--coverage_audit_path", type=str, default="")
    parser.add_argument("--factor_code", type=str, default="")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--alpha_warmup_epochs", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "plateau", "cosine"])
    parser.add_argument("--plateau_patience", type=int, default=4)
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--lr_min_factor", type=float, default=0.03, help="Final LR factor for cosine decay; set negative to fall back to --min_lr.")
    parser.add_argument("--lr_warmup_epochs", type=int, default=1, help="Number of initial epochs for LR warmup when --scheduler cosine.")
    parser.add_argument("--lr_warmup_start_factor", type=float, default=0.0, help="Initial LR factor for cosine warmup, in [0, 1].")
    parser.add_argument("--early_stop_metric", type=str, default="val_mae", choices=["val_loss", "val_acc", "val_macro_f1", "val_mae"])
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--backbone_lr_ratio", type=float, default=0.05)
    parser.add_argument("--unfreeze_bert_last_layer", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unfreeze_bert_last_n_layers", type=int, default=0)
    parser.add_argument(
        "--use_text_cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use cached text embeddings; explicitly disabled for last-layer unfreezing experiments.",
    )
    parser.add_argument("--bert_last_layer_lr_ratio", type=float, default=0.5, help="LR ratio for the unfrozen BERT final encoder layer, relative to --lr.")
    parser.add_argument("--loss_type", type=str, default="weighted_ce", choices=["weighted_ce", "focal"])
    parser.add_argument("--intensity_loss_type", type=str, default="mse", choices=["mse", "smooth_l1"])
    parser.add_argument("--lambda_polarity", type=float, default=1.0)
    parser.add_argument("--output_head_mode", type=str, default="legacy", choices=["legacy", "signed_reg_cls7"])
    parser.add_argument("--reg_loss_type", type=str, default="smooth_l1", choices=["mse", "smooth_l1"])
    parser.add_argument("--cls7_loss_type", type=str, default="soft_ce", choices=["hard_ce", "soft_ce"])
    parser.add_argument("--cls7_loss_weight", type=float, default=0.5)
    parser.add_argument("--cls7_class_weight_mode", type=str, default="none", choices=["none", "balanced", "sqrt_balanced"])
    parser.add_argument("--cls7_class_weight_max", type=float, default=0.0)
    parser.add_argument("--cls7_low_abs_sample_weight_threshold", type=float, default=0.0)
    parser.add_argument("--cls7_low_abs_sample_weight", type=float, default=1.0)
    parser.add_argument("--cls7_soft_tau", type=float, default=0.5)
    parser.add_argument("--cls7_head_type", type=str, default="flat", choices=["flat", "hier_sign_mag", "cumulative", "hybrid_cumulative"])
    parser.add_argument("--cumulative_p7_mix", type=float, default=0.3)
    parser.add_argument("--cumulative_loss_weight", type=float, default=0.0)
    parser.add_argument("--hier_sign_loss_weight", type=float, default=0.0)
    parser.add_argument("--hier_mag_loss_weight", type=float, default=0.0)
    parser.add_argument("--sign_struct_warmup_ratio", type=float, default=0.0)
    parser.add_argument("--sign_marginal_loss_type", type=str, default="none", choices=["none", "focal", "soft_f1"])
    parser.add_argument("--sign_marginal_loss_weight", type=float, default=0.0)
    parser.add_argument("--sign_marginal_focal_gamma", type=float, default=1.5)
    parser.add_argument("--signed_neutral_band_weight", type=float, default=0.0)
    parser.add_argument("--signed_neutral_band_margin", type=float, default=1.0 / 6.0)
    parser.add_argument("--signed_neutral_band_upper", type=float, default=0.45)
    parser.add_argument("--signed_neutral_band_temperature", type=float, default=0.05)
    parser.add_argument("--signed_neutral_band_eta", type=float, default=0.4)
    parser.add_argument("--zero_sign_margin_weight", type=float, default=0.0)
    parser.add_argument("--zero_sign_margin", type=float, default=0.02)
    parser.add_argument("--zero_sign_margin_temperature", type=float, default=0.02)
    parser.add_argument("--reg_cls_mag_consistency_weight", type=float, default=0.0)
    parser.add_argument("--reg_cls_mag_consistency_boundary_margin", type=float, default=0.1)
    parser.add_argument("--reg_cls_mag_consistency_smooth_l1_beta", type=float, default=0.1)
    parser.add_argument(
        "--reg_cls_mag_consistency_mode",
        type=str,
        default=REG_CLS_MAG_CONSISTENCY_MODE,
        choices=[
            REG_CLS_MAG_CONSISTENCY_MODE,
            REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE,
        ],
    )
    parser.add_argument("--final_pred_eta", type=float, default=0.4)
    parser.add_argument("--enable_sign_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--final_pred_sign_beta", type=float, default=0.0)
    parser.add_argument("--neutral_positive_gate_threshold", type=float, default=None)
    parser.add_argument("--sign_aux_weight", type=float, default=0.0)
    parser.add_argument("--sign_aux_loss_type", type=str, default="bce", choices=["bce", "focal_bce"])
    parser.add_argument("--sign_aux_focal_gamma", type=float, default=2.0)
    parser.add_argument("--sign_aux_low_abs_weight_threshold", type=float, default=0.0)
    parser.add_argument("--sign_aux_low_abs_weight", type=float, default=1.0)
    parser.add_argument("--sign_aux_zero_weight", type=float, default=1.0)
    parser.add_argument("--sign_aux_nonzero_weight", type=float, default=1.0)
    parser.add_argument("--sign_aux_positive_weight", type=float, default=1.0)
    parser.add_argument("--sign_aux_negative_weight", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_weight", type=float, default=0.0)
    parser.add_argument("--score_sign_aux_temperature", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_loss_type", type=str, default="bce", choices=["bce", "focal_bce"])
    parser.add_argument("--score_sign_aux_focal_gamma", type=float, default=2.0)
    parser.add_argument("--score_sign_aux_low_abs_weight_threshold", type=float, default=0.0)
    parser.add_argument("--score_sign_aux_low_abs_weight", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_zero_weight", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_nonzero_weight", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_positive_weight", type=float, default=1.0)
    parser.add_argument("--score_sign_aux_negative_weight", type=float, default=1.0)
    parser.add_argument("--enable_oacr", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lambda_oacr", type=float, default=0.0)
    parser.add_argument("--oacr_tau", type=float, default=0.15)
    parser.add_argument("--oacr_sigma_y", type=float, default=0.7)
    parser.add_argument("--oacr_alpha_c", type=float, default=0.2)
    parser.add_argument("--contrast_proj_dim", type=int, default=128)
    parser.add_argument("--clamp_regression_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_epoch_checkpoints", action="store_true", default=False)
    parser.add_argument("--save_latest_checkpoint", action="store_true", default=False)
    parser.add_argument(
        "--checkpoint_selection_split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Split used to promote best Acc7/MAE checkpoints; val is required for leakage-safe new experiments.",
    )
    parser.add_argument(
        "--checkpoint_metrics",
        type=str,
        default="dual",
        choices=["dual", "acc7"],
        help="Save both Acc7/MAE checkpoints or only Acc7. Single-checkpoint campaigns avoid duplicate eta sweeps.",
    )
    parser.add_argument(
        "--evaluate_test_each_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy diagnostic mode. Disable for validation-selected experiments and evaluate test only post-training.",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--attention_dropout", type=float, default=0.0)

    preliminary, _ = parser.parse_known_args()
    if preliminary.training_config:
        load_training_defaults(parser, preliminary.training_config)
    args = parser.parse_args()
    if args.checkpoint_selection_split == "test" and not args.evaluate_test_each_epoch:
        raise ValueError("--checkpoint_selection_split test requires --evaluate_test_each_epoch")
    if args.cls7_soft_tau <= 0:
        raise ValueError("--cls7_soft_tau must be positive")
    if args.temporal_label_gate_beta <= 0:
        raise ValueError("--temporal_label_gate_beta must be positive")
    if not (0.0 <= args.temporal_zero_bridge_weight <= 1.0):
        raise ValueError("--temporal_zero_bridge_weight must be in [0, 1]")
    if not (0.0 <= args.final_pred_eta <= 1.0):
        raise ValueError("--final_pred_eta must be in [0, 1]")
    if not (0.0 <= args.final_pred_sign_beta <= 1.0):
        raise ValueError("--final_pred_sign_beta must be in [0, 1]")
    if args.neutral_positive_gate_threshold is not None and not (
        -1.0 <= args.neutral_positive_gate_threshold <= 1.0
    ):
        raise ValueError("--neutral_positive_gate_threshold must be in [-1, 1]")
    if args.neutral_positive_gate_threshold is not None and not args.enable_sign_head:
        raise ValueError("--neutral_positive_gate_threshold requires --enable_sign_head")
    if (
        args.neutral_positive_gate_threshold is not None
        and args.output_head_mode != "signed_reg_cls7"
    ):
        raise ValueError(
            "--neutral_positive_gate_threshold requires --output_head_mode signed_reg_cls7"
        )
    if not (0.0 <= args.cumulative_p7_mix <= 1.0):
        raise ValueError("--cumulative_p7_mix must be in [0, 1]")
    if not (0.0 <= args.sign_struct_warmup_ratio <= 1.0):
        raise ValueError("--sign_struct_warmup_ratio must be in [0, 1]")
    for name in [
        "hier_sign_loss_weight",
        "hier_mag_loss_weight",
        "cumulative_loss_weight",
        "sign_marginal_loss_weight",
        "sign_marginal_focal_gamma",
        "signed_neutral_band_weight",
        "zero_sign_margin_weight",
        "zero_sign_margin",
        "reg_cls_mag_consistency_weight",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if not (0.0 < args.signed_neutral_band_margin < args.signed_neutral_band_upper < 0.5):
        raise ValueError("signed neutral band requires 0 < margin < upper < 0.5")
    if args.signed_neutral_band_temperature <= 0:
        raise ValueError("--signed_neutral_band_temperature must be positive")
    if not (0.0 <= args.signed_neutral_band_eta <= 1.0):
        raise ValueError("--signed_neutral_band_eta must be in [0, 1]")
    if not (0.0 <= args.zero_sign_margin < 0.5):
        raise ValueError("--zero_sign_margin must be in [0, 0.5)")
    if args.zero_sign_margin_temperature <= 0:
        raise ValueError("--zero_sign_margin_temperature must be positive")
    if not (0.0 <= args.reg_cls_mag_consistency_boundary_margin < 0.5):
        raise ValueError("--reg_cls_mag_consistency_boundary_margin must be in [0, 0.5)")
    if args.reg_cls_mag_consistency_smooth_l1_beta <= 0:
        raise ValueError("--reg_cls_mag_consistency_smooth_l1_beta must be positive")
    if args.sign_aux_weight < 0:
        raise ValueError("--sign_aux_weight must be non-negative")
    if args.sign_aux_focal_gamma < 0:
        raise ValueError("--sign_aux_focal_gamma must be non-negative")
    if args.sign_aux_low_abs_weight_threshold < 0:
        raise ValueError("--sign_aux_low_abs_weight_threshold must be non-negative")
    for name in [
        "sign_aux_low_abs_weight",
        "sign_aux_zero_weight",
        "sign_aux_nonzero_weight",
        "sign_aux_positive_weight",
        "sign_aux_negative_weight",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.score_sign_aux_weight < 0:
        raise ValueError("--score_sign_aux_weight must be non-negative")
    if args.score_sign_aux_temperature <= 0:
        raise ValueError("--score_sign_aux_temperature must be positive")
    if args.score_sign_aux_focal_gamma < 0:
        raise ValueError("--score_sign_aux_focal_gamma must be non-negative")
    if args.score_sign_aux_low_abs_weight_threshold < 0:
        raise ValueError("--score_sign_aux_low_abs_weight_threshold must be non-negative")
    for name in [
        "score_sign_aux_low_abs_weight",
        "score_sign_aux_zero_weight",
        "score_sign_aux_nonzero_weight",
        "score_sign_aux_positive_weight",
        "score_sign_aux_negative_weight",
    ]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.enable_sign_head and args.output_head_mode != "signed_reg_cls7":
        raise ValueError("--enable_sign_head requires --output_head_mode signed_reg_cls7")
    if args.score_sign_aux_weight > 0 and args.output_head_mode != "signed_reg_cls7":
        raise ValueError("--score_sign_aux_weight requires --output_head_mode signed_reg_cls7")
    if args.cls7_head_type != "flat" and args.output_head_mode != "signed_reg_cls7":
        raise ValueError("--cls7_head_type other than flat requires --output_head_mode signed_reg_cls7")
    if (
        args.hier_sign_loss_weight > 0
        or args.hier_mag_loss_weight > 0
        or args.sign_marginal_loss_weight > 0
        or args.signed_neutral_band_weight > 0
        or args.zero_sign_margin_weight > 0
        or args.reg_cls_mag_consistency_weight > 0
    ) and args.output_head_mode != "signed_reg_cls7":
        raise ValueError("structured cls7 losses require --output_head_mode signed_reg_cls7")
    if (args.hier_sign_loss_weight > 0 or args.hier_mag_loss_weight > 0) and args.cls7_head_type != "hier_sign_mag":
        raise ValueError("hier sign/mag losses require --cls7_head_type hier_sign_mag")
    if args.cumulative_loss_weight > 0 and args.cls7_head_type not in {"cumulative", "hybrid_cumulative"}:
        raise ValueError("--cumulative_loss_weight requires --cls7_head_type cumulative or hybrid_cumulative")
    if args.cls7_class_weight_max < 0:
        raise ValueError("--cls7_class_weight_max must be non-negative")
    if args.lambda_oacr < 0:
        raise ValueError("--lambda_oacr must be non-negative")
    if args.enable_oacr and args.output_head_mode != "signed_reg_cls7":
        raise ValueError("OACR requires --output_head_mode signed_reg_cls7")
    if args.oacr_tau <= 0:
        raise ValueError("--oacr_tau must be positive")
    if args.oacr_sigma_y <= 0:
        raise ValueError("--oacr_sigma_y must be positive")
    if args.contrast_proj_dim <= 0:
        raise ValueError("--contrast_proj_dim must be positive")
    if args.vit_context_attention_dim <= 0:
        raise ValueError("--vit_context_attention_dim must be positive")
    if args.unfreeze_bert_last_layer and args.unfreeze_bert_last_n_layers <= 0:
        args.unfreeze_bert_last_n_layers = 1
    if args.unfreeze_bert_last_n_layers > 0:
        args.unfreeze_bert_last_layer = True
    if args.unfreeze_bert_last_n_layers < 0:
        raise ValueError("--unfreeze_bert_last_n_layers must be non-negative")
    args.use_text_cache = _resolve_use_text_cache(args)
    if args.bert_last_layer_lr_ratio <= 0:
        raise ValueError("--bert_last_layer_lr_ratio must be positive")
    if args.tcif_transition_gate_lr is None:
        args.tcif_transition_gate_lr = args.lr
    if args.tcif_transition_gate_lr <= 0:
        raise ValueError("--tcif_transition_gate_lr must be positive")
    if args.num_temporal_contrast_experts < 0:
        raise ValueError("--num_temporal_contrast_experts must be non-negative")
    if args.enable_temporal_contrast_experts and args.num_temporal_contrast_experts <= 0:
        raise ValueError("--enable_temporal_contrast_experts requires --num_temporal_contrast_experts > 0")
    if args.temporal_embedding_dim <= 0:
        raise ValueError("--temporal_embedding_dim must be positive")
    if args.temporal_contrast_weight < 0:
        raise ValueError("--temporal_contrast_weight must be non-negative")
    if args.temporal_contrast_temperature <= 0:
        raise ValueError("--temporal_contrast_temperature must be positive")
    if args.temporal_decay_tau <= 0:
        raise ValueError("--temporal_decay_tau must be positive")
    if args.temporal_positive_radius < 0:
        raise ValueError("--temporal_positive_radius must be non-negative")
    if args.temporal_weak_positive_radius < args.temporal_positive_radius:
        raise ValueError("--temporal_weak_positive_radius must be >= --temporal_positive_radius")
    if args.temporal_min_positive_weight < 0:
        raise ValueError("--temporal_min_positive_weight must be non-negative")
    if args.temporal_batch_window < 0:
        raise ValueError("--temporal_batch_window must be non-negative")
    if args.target_sampler != "none" and args.temporal_batch_mode != "shuffle":
        raise ValueError("--target_sampler currently requires --temporal_batch_mode shuffle")
    if args.tcif_context_radius < 0:
        raise ValueError("--tcif_context_radius must be non-negative")
    validate_training_ablation(args)
    if args.tcif_latent_dim <= 0:
        raise ValueError("--tcif_latent_dim must be positive")
    if args.tcif_context_temperature <= 0:
        raise ValueError("--tcif_context_temperature must be positive")
    if args.tcif_context_aux_weight < 0:
        raise ValueError("--tcif_context_aux_weight must be non-negative")
    if args.tcif_context_aux_cls7_weight < 0:
        raise ValueError("--tcif_context_aux_cls7_weight must be non-negative")
    if args.tcif_transition_gate_hidden_dim <= 0:
        raise ValueError("--tcif_transition_gate_hidden_dim must be positive")
    if args.tcif_transition_gate_loss_weight < 0:
        raise ValueError("--tcif_transition_gate_loss_weight must be non-negative")
    if args.tcif_transition_gate_tau <= 0:
        raise ValueError("--tcif_transition_gate_tau must be positive")
    if not (0.0 <= args.tcif_transition_gate_conflict_target <= 1.0):
        raise ValueError(
            "--tcif_transition_gate_conflict_target must be in [0, 1]"
        )
    if args.enable_tcif:
        if args.dataset != "cmumosei":
            raise ValueError("TCIF stage-1 currently supports --dataset cmumosei")
        if args.output_head_mode != "signed_reg_cls7":
            raise ValueError("--enable_tcif requires --output_head_mode signed_reg_cls7")
        if args.cls7_head_type != "flat":
            raise ValueError("--enable_tcif currently requires --cls7_head_type flat")
        if args.tcif_context_radius <= 0:
            raise ValueError("--enable_tcif requires --tcif_context_radius > 0")
        if args.tcif_enable_transition_gate is False and args.tcif_transition_gate_loss_weight > 0:
            raise ValueError(
                "--tcif_transition_gate_loss_weight requires "
                "--tcif_enable_transition_gate"
            )
    elif args.tcif_context_aux_weight > 0 or args.tcif_enable_transition_gate:
        raise ValueError("TCIF auxiliary/gate options require --enable_tcif")
    os.makedirs(args.save_dir, exist_ok=True)
    train(args)
