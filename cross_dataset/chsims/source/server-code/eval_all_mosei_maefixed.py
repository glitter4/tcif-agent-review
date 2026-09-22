import argparse
import csv
import json
import math
import os
import re
import time
import traceback

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.emotion_dataset import (
    CHSIMSProcessDataset,
    CMUMOSEIProcessDataset,
    CMUMOSIProcessDataset,
    _valence_to_binary,
)
from models.models_emotion import EmotionM4OE
from head_utils import compute_final_prediction, continuous_to_cls7_hard, get_cls7_centers
from output_layout import ensure_structv5_root


def normalize_path(path: str) -> str:
    p = str(path).strip().replace("\\", "/")
    if p.startswith("/"):
        return p
    if p.startswith("data/"):
        return "/" + p
    if p.startswith("data"):
        return "/" + p
    return p


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model_state_dict" in checkpoint_obj:
            return checkpoint_obj["model_state_dict"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
    return checkpoint_obj


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
        return (
            0.0,
            0.0,
            [0.0 for _ in range(num_classes)],
            [0.0 for _ in range(num_classes)],
        )
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


def mae_fixed(preds, targets):
    assert len(preds) == len(targets), "preds/targets length mismatch"
    if len(preds) == 0:
        return 0.0
    return sum(abs(float(p) - float(t)) for p, t in zip(preds, targets)) / len(preds)


def parse_value(text, cast=float):
    try:
        return cast(text)
    except Exception:
        return None


def parse_hparams_from_dirname(dirname: str):
    hp = {"parsed_from_dirname": True}
    patterns = {
        "lr": r"(?:^|_)lr([0-9]+(?:\.[0-9]+)?(?:e-?[0-9]+)?)",
        "batch_size": r"(?:^|_)bs([0-9]+)",
        "router_temperature": r"(?:^|_)T([0-9]+(?:\.[0-9]+)?)",
        "alpha": r"(?:^|_)a([0-9]+(?:\.[0-9]+)?)",
        "lambda_polarity": r"(?:^|_)lambda([0-9]+(?:\.[0-9]+)?)",
        "num_experts_mtoe": r"(?:^|_)mtoe([0-9]+)",
        "num_experts_msoe": r"(?:^|_)msoe([0-9]+)",
        "num_shared_experts": r"(?:^|_)(?:shared|ns)([0-9]+)",
        "num_text_specific_experts": r"(?:^|_)(?:ts|textspec)([0-9]+)",
        "num_audio_specific_experts": r"(?:^|_)(?:as|audiospec)([0-9]+)",
        "num_vision_specific_experts": r"(?:^|_)(?:vs|visionspec)([0-9]+)",
        "num_frames": r"(?:^|_)nf([0-9]+)",
    }
    for k, p in patterns.items():
        m = re.search(p, dirname)
        if not m:
            continue
        if k in {
            "batch_size",
            "num_experts_mtoe",
            "num_experts_msoe",
            "num_shared_experts",
            "num_text_specific_experts",
            "num_audio_specific_experts",
            "num_vision_specific_experts",
            "num_frames",
        }:
            hp[k] = parse_value(m.group(1), int)
        else:
            hp[k] = parse_value(m.group(1), float)
    if "vit" in dirname.lower():
        hp["vision_backbone_type"] = "vit"
    return hp


def discover_models(checkpoints_root: str):
    targets = {"best_mae_model.pth", "best_acc7_model.pth", "best_model.pth", "best_guarded_acc2_model.pth", "best_dev_argmax_acc5_model.pth", "best_dev_argmax_mae_model.pth"}
    found = []
    for root, _, files in os.walk(checkpoints_root):
        for fname in files:
            if fname not in targets:
                continue
            model_path = os.path.join(root, fname)
            stem = os.path.splitext(fname)[0]
            same_json = os.path.join(root, f"{stem}.json")
            hparams = {}
            hparam_source = "dirname"
            if os.path.exists(same_json):
                with open(same_json, "r", encoding="utf-8") as f:
                    hparams = json.load(f)
                hparam_source = "same_name_json"
            else:
                hparams = parse_hparams_from_dirname(os.path.basename(root))
            found.append(
                {
                    "model_path": model_path,
                    "model_name": fname,
                    "model_stem": stem,
                    "model_dir": root,
                    "hparams": hparams,
                    "hparam_source": hparam_source,
                    "same_name_json": same_json if os.path.exists(same_json) else None,
                }
            )
    found.sort(key=lambda x: x["model_path"])
    return found


def build_dataset(
    dataset: str,
    dataset_root: str,
    split: str,
    tokenizer_path: str,
    local_files_only: bool,
    frame_policy: str,
    num_frames: int,
    vit_context_ratio: float,
    embedding_cache_root: str = "",
    temporal_context_radius: int = 0,
    temporal_context_mode: str = "neighbors",
    temporal_context_seed: int = 0,
    chsims_valence_scale: float = 1.0,
):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if dataset == "cmumosi":
        return CMUMOSIProcessDataset(
            root=dataset_root,
            split=split,
            transform=transform,
            tokenizer_path=tokenizer_path,
            local_files_only=local_files_only,
            frame_policy=frame_policy,
            num_frames=num_frames,
            vit_context_ratio=vit_context_ratio,
            embedding_cache_root=embedding_cache_root or None,
            temporal_context_radius=temporal_context_radius,
            temporal_context_mode=temporal_context_mode,
            temporal_context_seed=temporal_context_seed,
        )
    if dataset == "chsims":
        return CHSIMSProcessDataset(
            root=dataset_root,
            split=split,
            transform=transform,
            tokenizer_path=tokenizer_path,
            local_files_only=local_files_only,
            frame_policy=frame_policy,
            num_frames=num_frames,
            vit_context_ratio=vit_context_ratio,
            embedding_cache_root=embedding_cache_root or None,
            valence_scale=chsims_valence_scale,
            temporal_context_radius=temporal_context_radius,
            temporal_context_mode=temporal_context_mode,
            temporal_context_seed=temporal_context_seed,
        )
    return CMUMOSEIProcessDataset(
        root=dataset_root,
        split=split,
        transform=transform,
        tokenizer_path=tokenizer_path,
        local_files_only=local_files_only,
        frame_policy=frame_policy,
        num_frames=num_frames,
        vit_context_ratio=vit_context_ratio,
        embedding_cache_root=embedding_cache_root or None,
        temporal_context_radius=temporal_context_radius,
        temporal_context_mode=temporal_context_mode,
        temporal_context_seed=temporal_context_seed,
    )


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


def evaluate_split(
    model,
    loader,
    device,
    final_pred_eta=0.0,
    final_pred_sign_beta=0.0,
    neutral_positive_gate_threshold=None,
    clamp_regression_eval=True,
    use_text_cache=True,
    signed_class_count=7,
    label_min=-3.0,
    label_max=3.0,
    classification_readout="expected",
):
    model.eval()
    ids = []
    raw_vals = []
    pred_vals = []
    pred_bin = []
    gt_bin = []
    pred_7 = []
    gt_7 = []
    tcif_context_vals = []
    tcif_has_context = []
    tcif_reg_innovation = []
    tcif_cls_innovation = []
    tcif_reg_gate = []
    tcif_cls_gate = []
    model.reset_expert_load_stats()
    with torch.no_grad():
        for batch in loader:
            batch_tcif_context = None
            batch_tcif_has_context = None
            batch_tcif_reg_innovation = None
            batch_tcif_cls_innovation = None
            batch_tcif_reg_gate = None
            batch_tcif_cls_gate = None
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            raw_valence = batch["raw_valence"].to(device)
            audio_values = batch["audio_values"].to(device)
            audio_attention_mask = batch["audio_attention_mask"].to(device)
            model_out = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                **_embedding_cache_kwargs(batch, device, use_text_cache=use_text_cache),
                **_tcif_context_kwargs(batch, device, use_text_cache=use_text_cache),
            )
            if getattr(model, "output_head_mode", "legacy") == "legacy":
                out_intensity, out_polarity, _ = model_out
                polarity_preds = torch.argmax(out_polarity, dim=1)
                signs = torch.where(
                    polarity_preds == 0,
                    -torch.ones_like(out_intensity),
                    torch.ones_like(out_intensity),
                )
                final_scores = out_intensity * signs
            else:
                y_reg = model_out["y_reg"]
                cls7_logits = model_out["cls7_logits"]
                sign_logits = model_out.get("sign_logits", None)
                centers = get_cls7_centers(
                    device=cls7_logits.device,
                    dtype=cls7_logits.dtype,
                    num_classes=signed_class_count,
                    label_min=label_min,
                    label_max=label_max,
                )
                y_reg_eval = (
                    y_reg.clamp(float(label_min), float(label_max))
                    if clamp_regression_eval
                    else y_reg
                )
                final_scores = compute_final_prediction(
                    y_reg_eval,
                    cls7_logits,
                    eta=final_pred_eta,
                    centers=centers,
                    classification_readout=classification_readout,
                    sign_logits=sign_logits,
                    sign_beta=final_pred_sign_beta,
                    neutral_positive_gate_threshold=neutral_positive_gate_threshold,
                )
                tcif_payload = model_out.get("extras", {}).get("tcif")
                if isinstance(tcif_payload, dict):
                    context_y_reg = tcif_payload["context_y_reg"]
                    if clamp_regression_eval:
                        context_y_reg = context_y_reg.clamp(
                            float(label_min),
                            float(label_max),
                        )
                    batch_tcif_context = compute_final_prediction(
                        context_y_reg,
                        tcif_payload["context_cls7_logits"],
                        eta=final_pred_eta,
                        centers=centers,
                        classification_readout=classification_readout,
                    )
                    batch_tcif_has_context = tcif_payload["has_context"]
                    batch_tcif_reg_innovation = tcif_payload[
                        "regression_innovation_norm"
                    ]
                    batch_tcif_cls_innovation = tcif_payload[
                        "ordinal_innovation_norm"
                    ]
                    batch_tcif_reg_gate = tcif_payload[
                        "regression_continuation_gate"
                    ]
                    batch_tcif_cls_gate = tcif_payload[
                        "ordinal_continuation_gate"
                    ]
            batch_ids = batch.get("id", None)
            if batch_ids is None:
                batch_ids = [
                    f"idx_{len(ids) + i}" for i in range(final_scores.shape[0])
                ]
            batch_raw = raw_valence.detach().cpu().numpy().tolist()
            batch_pred = final_scores.detach().cpu().numpy().tolist()
            ids.extend(list(batch_ids))
            raw_vals.extend([float(v) for v in batch_raw])
            pred_vals.extend([float(v) for v in batch_pred])
            if batch_tcif_context is not None:
                tcif_context_vals.extend(
                    [
                        float(v)
                        for v in batch_tcif_context.detach().cpu().numpy().tolist()
                    ]
                )
                tcif_has_context.extend(
                    [
                        bool(v)
                        for v in batch_tcif_has_context.detach().cpu().numpy().tolist()
                    ]
                )
                tcif_reg_innovation.extend(
                    [
                        float(v)
                        for v in batch_tcif_reg_innovation.detach().cpu().numpy().tolist()
                    ]
                )
                tcif_cls_innovation.extend(
                    [
                        float(v)
                        for v in batch_tcif_cls_innovation.detach().cpu().numpy().tolist()
                    ]
                )
                tcif_reg_gate.extend(
                    [
                        float(v)
                        for v in batch_tcif_reg_gate.detach().cpu().numpy().tolist()
                    ]
                )
                tcif_cls_gate.extend(
                    [
                        float(v)
                        for v in batch_tcif_cls_gate.detach().cpu().numpy().tolist()
                    ]
                )
    assert len(ids) == len(raw_vals) == len(pred_vals), "id/label/pred length mismatch"
    metric_centers = get_cls7_centers(
        num_classes=signed_class_count,
        label_min=label_min,
        label_max=label_max,
    )
    pred_signed = continuous_to_cls7_hard(
        torch.tensor(pred_vals, dtype=torch.float32), centers=metric_centers
    ).tolist()
    gt_signed = continuous_to_cls7_hard(
        torch.tensor(raw_vals, dtype=torch.float32), centers=metric_centers
    ).tolist()
    for pv, rv, p7, g7 in zip(pred_vals, raw_vals, pred_signed, gt_signed):
        pb = _valence_to_binary(pv)
        gb = _valence_to_binary(rv)
        pred_bin.append(pb)
        gt_bin.append(gb)
        pred_7.append(p7)
        gt_7.append(g7)
    acc2 = accuracy_from_lists(gt_bin, pred_bin)
    acc7, f1_7, _, _ = metrics_from_lists(
        gt_7, pred_7, int(signed_class_count)
    )
    _, f1_2, _, _ = metrics_from_lists(gt_bin, pred_bin, 2)
    rmse, mae_reg, pearson, spearman = regression_metrics(pred_vals, raw_vals)
    mae = mae_fixed(pred_vals, raw_vals)
    assert abs(mae - mae_reg) < 1e-10, "MAE fixed mismatch against regression_metrics"
    details = []
    abs_sum = 0.0
    for row_idx, (i, t, p) in enumerate(zip(ids, raw_vals, pred_vals)):
        err = float(p - t)
        abs_sum += abs(err)
        row = {
            "id": str(i),
            "true_value": float(t),
            "pred_value": float(p),
            "error": err,
        }
        if tcif_context_vals:
            row.update(
                {
                    "tcif_has_context": bool(tcif_has_context[row_idx]),
                    "tcif_context_pred_value": float(tcif_context_vals[row_idx]),
                    "tcif_reg_innovation_norm": float(
                        tcif_reg_innovation[row_idx]
                    ),
                    "tcif_cls_innovation_norm": float(
                        tcif_cls_innovation[row_idx]
                    ),
                    "tcif_regression_continuation_gate": float(
                        tcif_reg_gate[row_idx]
                    ),
                    "tcif_ordinal_continuation_gate": float(
                        tcif_cls_gate[row_idx]
                    ),
                }
            )
        details.append(row)
    mae_from_details = abs_sum / max(1, len(details))
    assert abs(mae_from_details - mae) < 1e-10, "MAE fixed mismatch against details"
    metrics = {
        "num_samples": len(details),
        "acc2": acc2,
        "acc7": acc7,
        "binary_f1": f1_2,
        "acc7_macro_f1": f1_7,
        "signed_class_count": int(signed_class_count),
        "signed_accuracy_name": f"Acc{int(signed_class_count)}",
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
    }
    if tcif_context_vals:
        valid_indices = [
            idx for idx, is_valid in enumerate(tcif_has_context) if is_valid
        ]
        context_preds = [tcif_context_vals[idx] for idx in valid_indices]
        context_targets = [raw_vals[idx] for idx in valid_indices]
        context_pred_7 = continuous_to_cls7_hard(
            torch.tensor(context_preds, dtype=torch.float32),
            centers=metric_centers,
        ).tolist()
        context_gt_7 = continuous_to_cls7_hard(
            torch.tensor(context_targets, dtype=torch.float32),
            centers=metric_centers,
        ).tolist()
        context_pred_2 = [_valence_to_binary(v) for v in context_preds]
        context_gt_2 = [_valence_to_binary(v) for v in context_targets]
        metrics.update(
            {
                "tcif_context_num_samples": len(valid_indices),
                "tcif_context_coverage": len(valid_indices) / max(1, len(raw_vals)),
                "tcif_context_mae": mae_fixed(context_preds, context_targets),
                "tcif_context_acc7": accuracy_from_lists(
                    context_gt_7,
                    context_pred_7,
                ),
                "tcif_context_acc2": accuracy_from_lists(
                    context_gt_2,
                    context_pred_2,
                ),
                "tcif_regression_continuation_gate_mean": float(
                    sum(tcif_reg_gate[idx] for idx in valid_indices)
                    / max(1, len(valid_indices))
                ),
                "tcif_ordinal_continuation_gate_mean": float(
                    sum(tcif_cls_gate[idx] for idx in valid_indices)
                    / max(1, len(valid_indices))
                ),
            }
        )
    router_stats = model.get_expert_load_summary(reset=True)
    return metrics, details, router_stats


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_details_csv(path, details):
    with open(path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["id", "true_value", "pred_value", "error"]
        if details and "tcif_has_context" in details[0]:
            fieldnames.extend(
                [
                    "tcif_has_context",
                    "tcif_context_pred_value",
                    "tcif_reg_innovation_norm",
                    "tcif_cls_innovation_norm",
                    "tcif_regression_continuation_gate",
                    "tcif_ordinal_continuation_gate",
                ]
            )
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for row in details:
            writer.writerow(row)


def safe_name(s: str):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def resolve_final_pred_eta(hparams, cli_final_pred_eta):
    if cli_final_pred_eta is not None:
        return float(cli_final_pred_eta), "cli"
    if "final_pred_eta" in hparams:
        return float(hparams["final_pred_eta"]), "hparams"
    return 0.0, "default"


def resolve_final_pred_sign_beta(hparams, cli_final_pred_sign_beta):
    if cli_final_pred_sign_beta is not None:
        return float(cli_final_pred_sign_beta), "cli"
    if "final_pred_sign_beta" in hparams:
        return float(hparams["final_pred_sign_beta"]), "hparams"
    return 0.0, "default"


def resolve_neutral_positive_gate_threshold(hparams, cli_threshold, cli_disabled=False):
    if cli_disabled:
        return None, "cli_disabled"
    if cli_threshold is not None:
        value, source = float(cli_threshold), "cli"
    elif hparams.get("neutral_positive_gate_threshold") is not None:
        value, source = float(hparams["neutral_positive_gate_threshold"]), "hparams"
    else:
        return None, "disabled"
    if not (-1.0 <= value <= 1.0):
        raise ValueError("neutral_positive_gate_threshold must be in [-1, 1]")
    return value, source


def main():
    ensure_structv5_root()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_root",
        type=str,
        default="/path/to/outputs/StructV7.0/train/checkpoints",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="/path/to/outputs/StructV7.0/eval_all_mosei_maefixed",
    )
    parser.add_argument(
        "--default_dataset",
        type=str,
        default="cmumosei",
        choices=["cmumosei", "cmumosi", "chsims"],
    )
    parser.add_argument(
        "--default_dataset_root",
        type=str,
        default="/path/to/datasets/cmumosei-process-complete-20260408",
    )
    parser.add_argument(
        "--default_tokenizer_path",
        type=str,
        default="/path/to/models/google-bert_bert-base-uncased",
    )
    parser.add_argument(
        "--default_vit_path",
        type=str,
        default="/path/to/models/vit-base-patch16-224-in21k",
    )
    parser.add_argument(
        "--default_bert_path",
        type=str,
        default="/path/to/models/google-bert_bert-base-uncased",
    )
    parser.add_argument(
        "--default_hubert_path",
        type=str,
        default="/path/to/models/hubert-base-ls960",
    )
    parser.add_argument(
        "--default_vision_backbone_path",
        type=str,
        default="/path/to/models/resnet-18",
    )
    parser.add_argument("--default_batch_size", type=int, default=16)
    parser.add_argument(
        "--output_head_mode",
        type=str,
        default="legacy",
        choices=["legacy", "signed_reg_cls7"],
    )
    parser.add_argument(
        "--final_pred_eta",
        type=float,
        default=None,
        help="Override checkpoint hparams final_pred_eta when explicitly provided.",
    )
    parser.add_argument(
        "--classification_readout",
        type=str,
        default="expected",
        choices=["expected", "argmax"],
        help="Classification endpoint used by eta fusion; argmax is an ablation.",
    )
    parser.add_argument(
        "--final_pred_sign_beta",
        type=float,
        default=None,
        help="Override checkpoint hparams final_pred_sign_beta when explicitly provided.",
    )
    neutral_gate_group = parser.add_mutually_exclusive_group()
    neutral_gate_group.add_argument(
        "--neutral_positive_gate_threshold",
        type=float,
        default=None,
        help=(
            "Optionally mirror negative predictions inside the neutral Acc7 bin when "
            "the sign-head unit score exceeds this threshold."
        ),
    )
    neutral_gate_group.add_argument(
        "--disable_neutral_positive_gate",
        action="store_true",
        help="Disable a neutral-positive gate recorded in checkpoint hparams.",
    )
    parser.add_argument(
        "--clamp_regression_eval", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--test_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅评估 test 集，不评估 val 集",
    )
    parser.add_argument(
        "--train_only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Evaluate only the train split. This is intended for fitting frozen "
            "post-model components and never changes checkpoint selection metadata."
        ),
    )
    parser.add_argument(
        "--enable_expert_load_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否输出专家负载统计（expert_load_val/test.json与run.log摘要）",
    )
    args = parser.parse_args()
    if args.train_only and args.test_only:
        parser.error("--train_only and --test_only are mutually exclusive")
    primary_split = "train" if args.train_only else "test"
    include_val = not args.train_only and not args.test_only

    checkpoints_root = normalize_path(args.checkpoints_root)
    results_root = normalize_path(args.results_root)
    os.makedirs(results_root, exist_ok=True)
    os.makedirs(os.path.join(results_root, "error_logs"), exist_ok=True)
    run_log_path = os.path.join(results_root, "run.log")

    def log(msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    start_time = time.time()
    log("job_start")
    log(f"checkpoints_root={checkpoints_root}")
    log(f"results_root={results_root}")

    discovered = discover_models(checkpoints_root)
    mapping_path = os.path.join(results_root, "model_hparams_mapping.json")
    write_json(mapping_path, discovered)
    log(f"discovered_models={len(discovered)}")

    if len(discovered) == 0:
        log("no_models_found")
        end_time = time.time()
        log(f"job_end elapsed_sec={end_time - start_time:.3f} qc_passed=false")
        return

    summary = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    for item in discovered:
        per_model_start = time.time()
        model_path = item["model_path"]
        model_name = item["model_name"]
        model_stem = item["model_stem"]
        hparams = dict(item.get("hparams", {}))
        rel_model_path = os.path.relpath(model_path, checkpoints_root)
        rel_parent = os.path.dirname(rel_model_path)
        out_dir = os.path.join(results_root, rel_parent, model_stem)
        os.makedirs(out_dir, exist_ok=True)

        dataset = str(hparams.get("dataset", args.default_dataset)).lower()
        if dataset not in {"cmumosei", "cmumosi", "chsims"}:
            dataset = args.default_dataset
        dataset_root = normalize_path(
            hparams.get("dataset_root", args.default_dataset_root)
        )
        tokenizer_path = hparams.get(
            "tokenizer_path",
            hparams.get("bert_backbone_path", args.default_tokenizer_path),
        )
        local_files_only = bool(hparams.get("local_files_only", True))
        frame_policy = str(hparams.get("frame_policy", "middle"))
        num_frames = int(hparams.get("num_frames", 4))
        vit_context_ratio = float(hparams.get("vit_context_ratio", 0.0))
        vit_context_lambda_mode = str(hparams.get("vit_context_lambda_mode", "static"))
        vit_context_lambda_init = hparams.get("vit_context_lambda_init", "")
        vit_context_attention_dim = int(hparams.get("vit_context_attention_dim", 64))
        embedding_cache_root = normalize_path(hparams.get("embedding_cache_root", ""))
        signed_class_count = int(
            hparams.get("signed_class_count", 5 if dataset == "chsims" else 7)
        )
        label_min = float(hparams.get("label_min", -1.0 if dataset == "chsims" else -3.0))
        label_max = float(hparams.get("label_max", 1.0 if dataset == "chsims" else 3.0))
        chsims_valence_scale = float(hparams.get("chsims_valence_scale", 1.0))
        batch_size = int(hparams.get("batch_size", args.default_batch_size))
        vision_backbone_type = str(hparams.get("vision_backbone_type", "vit"))
        num_experts_msoe = int(hparams.get("num_experts_msoe", 16))
        num_experts_mtoe = int(hparams.get("num_experts_mtoe", 32))
        num_shared_experts = int(hparams.get("num_shared_experts", 2))
        num_text_specific_experts = int(hparams.get("num_text_specific_experts", 4))
        num_audio_specific_experts = int(hparams.get("num_audio_specific_experts", 4))
        num_vision_specific_experts = int(hparams.get("num_vision_specific_experts", 4))
        expert_mlp_ratio = float(hparams.get("expert_mlp_ratio", 1.0))
        num_temporal_contrast_experts = int(
            hparams.get("num_temporal_contrast_experts", 0)
        )
        enable_text_shared_experts = bool(
            hparams.get("enable_text_shared_experts", True)
        )
        enable_temporal_contrast_experts = bool(
            hparams.get("enable_temporal_contrast_experts", False)
        )
        enable_temporal_contrast_loss = bool(
            hparams.get("enable_temporal_contrast_loss", False)
        )
        temporal_embedding_dim = int(hparams.get("temporal_embedding_dim", 128))
        temporal_detach_task2 = bool(hparams.get("temporal_detach_task2", False))
        enable_tcif = bool(hparams.get("enable_tcif", False))
        tcif_context_radius = int(
            hparams.get("tcif_context_radius", 1 if enable_tcif else 0)
        )
        tcif_context_mode = str(hparams.get("tcif_context_mode", "neighbors"))
        tcif_context_seed = int(hparams.get("seed", 0))
        tcif_latent_dim = int(hparams.get("tcif_latent_dim", 128))
        tcif_output_mode = str(hparams.get("tcif_output_mode", "posterior"))
        tcif_context_temperature = float(
            hparams.get("tcif_context_temperature", 1.0)
        )
        tcif_enable_transition_gate = bool(
            hparams.get("tcif_enable_transition_gate", False)
        )
        tcif_transition_gate_hidden_dim = int(
            hparams.get("tcif_transition_gate_hidden_dim", 64)
        )
        tcif_transition_gate_init_bias = float(
            hparams.get("tcif_transition_gate_init_bias", 2.0)
        )
        tcif_transition_gate_lr = float(
            hparams.get("tcif_transition_gate_lr")
            or hparams.get("lr", 3e-5)
        )
        if enable_tcif and dataset not in {"cmumosei", "cmumosi", "chsims"}:
            raise ValueError(
                "TCIF evaluation requires a process-format temporal dataset: "
                "cmumosei, cmumosi, or chsims"
            )
        router_temperature = float(hparams.get("router_temperature", 0.1))
        dropout = float(hparams.get("dropout", 0.3))
        enable_expert_load_output = bool(
            hparams.get("enable_expert_load_output", args.enable_expert_load_output)
        )
        output_head_mode = str(hparams.get("output_head_mode", args.output_head_mode))
        cls7_head_type = str(hparams.get("cls7_head_type", "flat"))
        cumulative_p7_mix = float(hparams.get("cumulative_p7_mix", 0.3))
        final_pred_eta, final_pred_eta_source = resolve_final_pred_eta(
            hparams, args.final_pred_eta
        )
        final_pred_sign_beta, final_pred_sign_beta_source = (
            resolve_final_pred_sign_beta(hparams, args.final_pred_sign_beta)
        )
        neutral_positive_gate_threshold, neutral_positive_gate_threshold_source = (
            resolve_neutral_positive_gate_threshold(
                hparams,
                args.neutral_positive_gate_threshold,
                args.disable_neutral_positive_gate,
            )
        )
        if (
            neutral_positive_gate_threshold is not None
            and output_head_mode != "signed_reg_cls7"
        ):
            raise ValueError(
                "neutral positive gate requires output_head_mode='signed_reg_cls7'"
            )
        clamp_regression_eval = bool(
            hparams.get("clamp_regression_eval", args.clamp_regression_eval)
        )
        enable_sign_head = bool(hparams.get("enable_sign_head", False))
        unfreeze_text = (
            bool(hparams.get("unfreeze_bert_last_layer", False))
            or int(hparams.get("unfreeze_bert_last_n_layers", 0) or 0) > 0
        )
        explicit_use_text_cache = hparams.get("use_text_cache")
        if explicit_use_text_cache is None:
            use_text_cache = not unfreeze_text
        else:
            use_text_cache = bool(explicit_use_text_cache)
            if unfreeze_text and use_text_cache:
                raise ValueError(
                    f"Invalid checkpoint metadata for {model_path}: unfrozen text layers require use_text_cache=false"
                )
        vit_model_path = hparams.get("vit_model_path", args.default_vit_path)
        bert_model_path = hparams.get(
            "bert_model_path", hparams.get("bert_backbone_path", args.default_bert_path)
        )
        hubert_model_path = hparams.get("hubert_model_path", args.default_hubert_path)
        vision_backbone_path = hparams.get(
            "vision_backbone_path", args.default_vision_backbone_path
        )

        merged_hparams = dict(hparams)
        merged_hparams.update(
            {
                "dataset": dataset,
                "dataset_root": dataset_root,
                "tokenizer_path": tokenizer_path,
                "local_files_only": local_files_only,
                "frame_policy": frame_policy,
                "num_frames": num_frames,
                "vit_context_ratio": vit_context_ratio,
                "vit_context_lambda_mode": vit_context_lambda_mode,
                "vit_context_lambda_init": vit_context_lambda_init,
                "vit_context_attention_dim": vit_context_attention_dim,
                "embedding_cache_root": embedding_cache_root,
                "signed_class_count": signed_class_count,
                "label_min": label_min,
                "label_max": label_max,
                "chsims_valence_scale": chsims_valence_scale,
                "batch_size": batch_size,
                "vision_backbone_type": vision_backbone_type,
                "num_experts_msoe": num_experts_msoe,
                "num_experts_mtoe": num_experts_mtoe,
                "num_shared_experts": num_shared_experts,
                "num_text_specific_experts": num_text_specific_experts,
                "num_audio_specific_experts": num_audio_specific_experts,
                "num_vision_specific_experts": num_vision_specific_experts,
                "expert_mlp_ratio": expert_mlp_ratio,
                "num_temporal_contrast_experts": num_temporal_contrast_experts,
                "enable_text_shared_experts": enable_text_shared_experts,
                "enable_temporal_contrast_experts": enable_temporal_contrast_experts,
                "enable_temporal_contrast_loss": enable_temporal_contrast_loss,
                "temporal_embedding_dim": temporal_embedding_dim,
                "temporal_detach_task2": temporal_detach_task2,
                "enable_tcif": enable_tcif,
                "tcif_context_radius": tcif_context_radius,
                "tcif_context_mode": tcif_context_mode,
                "tcif_context_seed": tcif_context_seed,
                "tcif_latent_dim": tcif_latent_dim,
                "tcif_output_mode": tcif_output_mode,
                "tcif_context_temperature": tcif_context_temperature,
                "tcif_enable_transition_gate": tcif_enable_transition_gate,
                "tcif_transition_gate_hidden_dim": tcif_transition_gate_hidden_dim,
                "tcif_transition_gate_init_bias": tcif_transition_gate_init_bias,
                "tcif_transition_gate_lr": tcif_transition_gate_lr,
                "router_temperature": router_temperature,
                "dropout": dropout,
                "enable_expert_load_output": enable_expert_load_output,
                "output_head_mode": output_head_mode,
                "cls7_head_type": cls7_head_type,
                "cumulative_p7_mix": cumulative_p7_mix,
                "final_pred_eta": final_pred_eta,
                "final_pred_eta_source": final_pred_eta_source,
                "classification_readout": args.classification_readout,
                "enable_sign_head": enable_sign_head,
                "final_pred_sign_beta": final_pred_sign_beta,
                "final_pred_sign_beta_source": final_pred_sign_beta_source,
                "neutral_positive_gate_threshold": neutral_positive_gate_threshold,
                "neutral_positive_gate_threshold_source": neutral_positive_gate_threshold_source,
                "clamp_regression_eval": clamp_regression_eval,
                "use_text_cache": use_text_cache,
                "vit_model_path": vit_model_path,
                "bert_model_path": bert_model_path,
                "hubert_model_path": hubert_model_path,
                "vision_backbone_path": vision_backbone_path,
                "hparam_source": item.get("hparam_source"),
                "same_name_json": item.get("same_name_json"),
                "model_path": model_path,
                "evaluation_primary_split": primary_split,
                "evaluation_train_only": bool(args.train_only),
            }
        )
        write_json(os.path.join(out_dir, "hyperparams.json"), merged_hparams)

        try:
            log(f"model_start path={model_path}")
            primary_ds = build_dataset(
                dataset,
                dataset_root,
                primary_split,
                tokenizer_path,
                local_files_only,
                frame_policy,
                num_frames,
                vit_context_ratio,
                embedding_cache_root,
                tcif_context_radius if enable_tcif else 0,
                tcif_context_mode,
                tcif_context_seed,
                chsims_valence_scale,
            )
            primary_loader = DataLoader(
                primary_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
            val_loader = None
            if include_val:
                val_ds = build_dataset(
                    dataset,
                    dataset_root,
                    "val",
                    tokenizer_path,
                    local_files_only,
                    frame_policy,
                    num_frames,
                    vit_context_ratio,
                    embedding_cache_root,
                    tcif_context_radius if enable_tcif else 0,
                    tcif_context_mode,
                    tcif_context_seed,
                    chsims_valence_scale,
                )
                val_loader = DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                )

            model = EmotionM4OE(
                num_classes=2,
                num_aux_classes=2,
                embed_dim=768,
                depth_msoe=2,
                depth_mtoe=1,
                num_experts_msoe=num_experts_msoe,
                num_experts_mtoe=num_experts_mtoe,
                num_shared_experts=num_shared_experts,
                num_text_specific_experts=num_text_specific_experts,
                num_audio_specific_experts=num_audio_specific_experts,
                num_vision_specific_experts=num_vision_specific_experts,
                num_temporal_contrast_experts=num_temporal_contrast_experts,
                enable_text_shared_experts=enable_text_shared_experts,
                enable_temporal_contrast_experts=enable_temporal_contrast_experts,
                enable_temporal_contrast_loss=enable_temporal_contrast_loss,
                temporal_embedding_dim=temporal_embedding_dim,
                temporal_detach_task2=temporal_detach_task2,
                vision_backbone_type=vision_backbone_type,
                vision_backbone_path=vision_backbone_path,
                vit_model_path=vit_model_path,
                bert_model_path=bert_model_path,
                hubert_model_path=hubert_model_path,
                local_files_only=local_files_only,
                dropout=dropout,
                expert_mlp_ratio=expert_mlp_ratio,
                router_temperature=router_temperature,
                enable_expert_load_output=enable_expert_load_output,
                vit_context_ratio=vit_context_ratio,
                vit_context_lambda_mode=vit_context_lambda_mode,
                vit_context_lambda_init=vit_context_lambda_init,
                vit_context_attention_dim=vit_context_attention_dim,
                output_head_mode=output_head_mode,
                final_pred_eta=final_pred_eta,
                signed_class_count=signed_class_count,
                label_min=label_min,
                label_max=label_max,
                enable_sign_head=enable_sign_head,
                final_pred_sign_beta=final_pred_sign_beta,
                neutral_positive_gate_threshold=neutral_positive_gate_threshold,
                cls7_head_type=cls7_head_type,
                cumulative_p7_mix=cumulative_p7_mix,
                enable_tcif=enable_tcif,
                tcif_latent_dim=tcif_latent_dim,
                tcif_output_mode=tcif_output_mode,
                tcif_context_temperature=tcif_context_temperature,
                tcif_enable_transition_gate=tcif_enable_transition_gate,
                tcif_transition_gate_hidden_dim=tcif_transition_gate_hidden_dim,
                tcif_transition_gate_init_bias=tcif_transition_gate_init_bias,
            ).to(device)
            try:
                state = torch.load(model_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=device)
            model.load_state_dict(_extract_state_dict(state), strict=True)

            val_metrics = {
                "num_samples": 0,
                "acc2": None,
                "acc7": None,
                "binary_f1": None,
                "acc7_macro_f1": None,
                "mae": None,
                "rmse": None,
                "pearson": None,
                "spearman": None,
            }
            val_details = []
            val_router_stats = {}
            if val_loader is not None:
                val_metrics, val_details, val_router_stats = evaluate_split(
                    model,
                    val_loader,
                    device,
                    final_pred_eta=final_pred_eta,
                    final_pred_sign_beta=final_pred_sign_beta,
                    neutral_positive_gate_threshold=neutral_positive_gate_threshold,
                    clamp_regression_eval=clamp_regression_eval,
                    use_text_cache=use_text_cache,
                    signed_class_count=signed_class_count,
                    label_min=label_min,
                    label_max=label_max,
                    classification_readout=args.classification_readout,
                )
            primary_metrics, primary_details, primary_router_stats = evaluate_split(
                model,
                primary_loader,
                device,
                final_pred_eta=final_pred_eta,
                final_pred_sign_beta=final_pred_sign_beta,
                neutral_positive_gate_threshold=neutral_positive_gate_threshold,
                clamp_regression_eval=clamp_regression_eval,
                use_text_cache=use_text_cache,
                signed_class_count=signed_class_count,
                label_min=label_min,
                label_max=label_max,
                classification_readout=args.classification_readout,
            )

            model_type = model_stem
            val_json_name = f"{model_type}_val_results.json"
            primary_json_name = f"{model_type}_{primary_split}_results.json"
            val_csv_name = f"{model_type}_val_details.csv"
            primary_csv_name = f"{model_type}_{primary_split}_details.csv"
            val_result = {
                "model_path": model_path,
                "model_name": model_name,
                "split": "val",
                "metrics": val_metrics,
            }
            primary_result = {
                "model_path": model_path,
                "model_name": model_name,
                "split": primary_split,
                "metrics": primary_metrics,
            }
            if not args.train_only:
                write_json(os.path.join(out_dir, val_json_name), val_result)
                write_json(os.path.join(out_dir, "val_results.json"), val_result)
                write_json(
                    os.path.join(out_dir, "expert_load_val.json"), val_router_stats
                )
                write_details_csv(os.path.join(out_dir, val_csv_name), val_details)
                write_details_csv(os.path.join(out_dir, "val_details.csv"), val_details)
            write_json(os.path.join(out_dir, primary_json_name), primary_result)
            write_json(
                os.path.join(out_dir, f"{primary_split}_results.json"), primary_result
            )
            write_json(
                os.path.join(out_dir, f"expert_load_{primary_split}.json"),
                primary_router_stats,
            )
            write_details_csv(os.path.join(out_dir, primary_csv_name), primary_details)
            write_details_csv(
                os.path.join(out_dir, f"{primary_split}_details.csv"),
                primary_details,
            )

            elapsed = time.time() - per_model_start
            summary_row = {
                "model_name": model_name,
                "model_stem": model_stem,
                "model_path": model_path,
                "relative_model_path": rel_model_path,
                "output_dir": out_dir,
                "hparams": merged_hparams,
                f"{primary_split}_mae": primary_metrics["mae"],
                f"{primary_split}_acc7": primary_metrics["acc7"],
                f"{primary_split}_acc2": primary_metrics["acc2"],
                f"{primary_split}_binary_f1": primary_metrics["binary_f1"],
                "duration_sec": elapsed,
                "status": "ok",
            }
            if not args.train_only:
                summary_row.update(
                    {
                        "val_mae": val_metrics["mae"],
                        "val_acc7": val_metrics["acc7"],
                        "val_acc2": val_metrics["acc2"],
                        "val_binary_f1": val_metrics["binary_f1"],
                    }
                )
            summary.append(summary_row)
            if val_loader is not None:
                log(
                    f"model_done path={model_path} elapsed_sec={elapsed:.3f} "
                    f"val_mae={val_metrics['mae']:.6f} {primary_split}_mae={primary_metrics['mae']:.6f} "
                    f"val_acc7={val_metrics['acc7']:.6f} {primary_split}_acc7={primary_metrics['acc7']:.6f} "
                    f"val_acc2={val_metrics['acc2']:.6f} {primary_split}_acc2={primary_metrics['acc2']:.6f} "
                    f"val_f1={val_metrics['binary_f1']:.6f} {primary_split}_f1={primary_metrics['binary_f1']:.6f}"
                )
                log(
                    f"expert_load_val path={model_path} {model.expert_load_tracker.brief(val_router_stats)}"
                )
            else:
                log(
                    f"model_done path={model_path} elapsed_sec={elapsed:.3f} "
                    f"{primary_split}_mae={primary_metrics['mae']:.6f} "
                    f"{primary_split}_acc7={primary_metrics['acc7']:.6f} "
                    f"{primary_split}_acc2={primary_metrics['acc2']:.6f} "
                    f"{primary_split}_f1={primary_metrics['binary_f1']:.6f} "
                    f"{primary_split}_only=true"
                )
            log(
                f"expert_load_{primary_split} path={model_path} "
                f"{model.expert_load_tracker.brief(primary_router_stats)}"
            )
        except Exception as e:
            elapsed = time.time() - per_model_start
            key = safe_name(rel_model_path.replace(os.sep, "__").replace(".pth", ""))
            err_path = os.path.join(results_root, "error_logs", f"{key}_error.log")
            with open(err_path, "w", encoding="utf-8") as ef:
                ef.write(f"model_path: {model_path}\n")
                ef.write(f"elapsed_sec: {elapsed:.6f}\n")
                ef.write(f"error: {repr(e)}\n")
                ef.write(traceback.format_exc())
            summary.append(
                {
                    "model_name": model_name,
                    "model_stem": model_stem,
                    "model_path": model_path,
                    "relative_model_path": rel_model_path,
                    "output_dir": out_dir,
                    "hparams": merged_hparams,
                    "status": "error",
                    "error_log": err_path,
                    "duration_sec": elapsed,
                }
            )
            log(
                f"model_error path={model_path} elapsed_sec={elapsed:.3f} error_log={err_path} error={repr(e)}"
            )

    summary_path = os.path.join(results_root, "all_models_summary.json")
    write_json(summary_path, summary)

    ok_cnt = sum(1 for x in summary if x.get("status") == "ok")
    err_cnt = sum(1 for x in summary if x.get("status") == "error")
    qc_passed = err_cnt == 0
    elapsed_total = time.time() - start_time
    log(
        f"job_end elapsed_sec={elapsed_total:.3f} ok_models={ok_cnt} error_models={err_cnt} qc_passed={str(qc_passed).lower()}"
    )


if __name__ == "__main__":
    main()
