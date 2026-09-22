#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

SERVER_CODE = Path(__file__).resolve().parents[1]
if str(SERVER_CODE) not in sys.path:
    sys.path.insert(0, str(SERVER_CODE))

from datasets.emotion_dataset import CMUMOSEIProcessDataset
from models.models_emotion import EmotionM4OE
from test_emotion import _embedding_cache_kwargs, _extract_state_dict

MODALITIES = ("vision", "text", "audio")
POLARITIES = ("all", "negative", "neutral", "positive")


def import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def load_hparams(checkpoint_path, hparams_path=None):
    ckpt = Path(checkpoint_path)
    candidates = []
    if hparams_path:
        candidates.append(Path(hparams_path))
    candidates.append(ckpt.with_suffix(".json"))
    candidates.append(ckpt.parent / "checkpoint_hparams.json")
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
    raise FileNotFoundError(f"Could not find hparams JSON for {checkpoint_path}")


def hget(hparams, key, default=None, *aliases):
    for name in (key,) + aliases:
        if name in hparams and hparams[name] is not None:
            return hparams[name]
    return default


def build_dataset(hparams, sample_limit=0):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = CMUMOSEIProcessDataset(
        root=hget(hparams, "dataset_root", "/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408"),
        split="test",
        transform=transform,
        tokenizer_path=hget(
            hparams,
            "tokenizer_path",
            "/path/to/user/models/AI-ModelScope_roberta-base",
            "bert_backbone_path",
            "bert_model_path",
        ),
        local_files_only=bool(hget(hparams, "local_files_only", True)),
        max_length=int(hget(hparams, "max_length", 128)),
        frame_policy=hget(hparams, "frame_policy", "middle"),
        audio_sample_rate=int(hget(hparams, "audio_sample_rate", 16000)),
        audio_max_seconds=float(hget(hparams, "audio_max_seconds", 6.0)),
        audio_frame_ms=float(hget(hparams, "audio_frame_ms", 25.0)),
        audio_hop_ms=float(hget(hparams, "audio_hop_ms", 10.0)),
        audio_denoise=not bool(hget(hparams, "disable_audio_denoise", False)),
        num_frames=int(hget(hparams, "num_frames", 4)),
        vit_context_ratio=float(hget(hparams, "vit_context_ratio", 0.0)),
        embedding_cache_root=hget(hparams, "embedding_cache_root", ""),
    )
    if sample_limit and sample_limit > 0:
        return Subset(dataset, list(range(min(int(sample_limit), len(dataset)))))
    return dataset


def build_model(hparams, device):
    kwargs = {
        "num_classes": 2,
        "num_aux_classes": 2,
        "embed_dim": int(hget(hparams, "embed_dim", 768)),
        "depth_msoe": int(hget(hparams, "depth_msoe", 2)),
        "depth_mtoe": int(hget(hparams, "depth_mtoe", 1)),
        "num_experts_msoe": int(hget(hparams, "num_experts_msoe", 16)),
        "num_experts_mtoe": int(hget(hparams, "num_experts_mtoe", 32)),
        "num_shared_experts": int(hget(hparams, "num_shared_experts", 4)),
        "num_text_specific_experts": int(hget(hparams, "num_text_specific_experts", 4)),
        "num_audio_specific_experts": int(hget(hparams, "num_audio_specific_experts", 4)),
        "num_vision_specific_experts": int(hget(hparams, "num_vision_specific_experts", 4)),
        "num_temporal_contrast_experts": int(hget(hparams, "num_temporal_contrast_experts", 0)),
        "enable_text_shared_experts": bool(hget(hparams, "enable_text_shared_experts", True)),
        "enable_temporal_contrast_experts": bool(hget(hparams, "enable_temporal_contrast_experts", False)),
        "enable_temporal_contrast_loss": bool(hget(hparams, "enable_temporal_contrast_loss", False)),
        "temporal_embedding_dim": int(hget(hparams, "temporal_embedding_dim", 128)),
        "vision_backbone_type": hget(hparams, "vision_backbone_type", "vit"),
        "vision_backbone_path": hget(hparams, "vision_backbone_path", "/path/to/user/models/resnet-18"),
        "vit_model_path": hget(
            hparams,
            "vit_model_path",
            "/path/to/user/models/vit-base-patch16-224-in21k",
            "vit_backbone_path",
        ),
        "bert_model_path": hget(
            hparams,
            "bert_model_path",
            "/path/to/user/models/AI-ModelScope_roberta-base",
            "bert_backbone_path",
        ),
        "hubert_model_path": hget(hparams, "hubert_model_path", "/path/to/user/models/hubert-base-ls960"),
        "local_files_only": bool(hget(hparams, "local_files_only", True)),
        "dropout": float(hget(hparams, "dropout", 0.3)),
        "attention_dropout": float(hget(hparams, "attention_dropout", 0.0)),
        "expert_mlp_ratio": float(hget(hparams, "expert_mlp_ratio", 4.0)),
        "alignment_num_heads": int(hget(hparams, "alignment_num_heads", 4)),
        "router_temperature": float(hget(hparams, "router_temperature", 0.1)),
        "enable_expert_load_output": False,
        "vit_context_ratio": float(hget(hparams, "vit_context_ratio", 0.0)),
        "vit_context_lambda_mode": hget(hparams, "vit_context_lambda_mode", "static"),
        "vit_context_lambda_init": hget(hparams, "vit_context_lambda_init", None),
        "vit_context_attention_dim": int(hget(hparams, "vit_context_attention_dim", 64)),
        "output_head_mode": hget(hparams, "output_head_mode", "signed_reg_cls7"),
        "final_pred_eta": float(hget(hparams, "final_pred_eta", 0.0)),
        "signed_class_count": int(hget(hparams, "signed_class_count", 7)),
        "label_min": float(hget(hparams, "label_min", -3.0)),
        "label_max": float(hget(hparams, "label_max", 3.0)),
    }
    accepted = set(inspect.signature(EmotionM4OE.__init__).parameters)
    kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    model = EmotionM4OE(**kwargs).to(device)
    return model


def list_from_batch(batch, key):
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def polarity_name(value):
    value = float(value)
    if value < -0.05:
        return "negative"
    if value > 0.05:
        return "positive"
    return "neutral"


def safe_row_normalize(x):
    denom = x.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return x / denom


def collect_checkpoint(name, checkpoint_path, hparams, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = build_dataset(hparams, sample_limit=args.sample_limit)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    model = build_model(hparams, device)
    state = _extract_state_dict(torch.load(checkpoint_path, map_location=device))
    model.load_state_dict(state)
    model.eval()

    rows = []
    embeddings = []
    attn_reg = []
    attn_ord = []
    routing = {modality: [] for modality in MODALITIES}
    layout = None

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            audio_values = batch["audio_values"].to(device, non_blocking=True)
            audio_attention_mask = batch["audio_attention_mask"].to(device, non_blocking=True)
            out = model(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                return_router_stats=True,
                return_analysis=True,
                **_embedding_cache_kwargs(batch, device),
            )
            extras = out.get("extras", {})
            affective = extras["affective_repr"].detach().cpu().float().numpy()
            embeddings.append(affective)
            query_attn = extras["task_query_attention"]
            attn_reg.append(query_attn["regression"].detach().cpu().float().numpy())
            attn_ord.append(query_attn["ordinal"].detach().cpu().float().numpy())

            router_stats = out.get("router_stats", {})
            layout = router_stats.get("expert_group_layout", layout)
            for modality in MODALITIES:
                masses = router_stats["msoe"][modality]["dispatch_expert_mass_per_sample"]
                routing[modality].append(safe_row_normalize(masses.detach().cpu().float().numpy()))

            ids = list_from_batch(batch, "id")
            base_ids = list_from_batch(batch, "base_id")
            clip_idxs = list_from_batch(batch, "clip_idx")
            positions = list_from_batch(batch, "temporal_pos")
            valences = list_from_batch(batch, "raw_valence")
            for sample_id, base_id, clip_idx, pos, val in zip(ids, base_ids, clip_idxs, positions, valences):
                rows.append({
                    "variant": name,
                    "id": str(sample_id),
                    "base_id": str(base_id),
                    "clip_idx": int(clip_idx) if str(clip_idx) != "None" else -1,
                    "temporal_pos": float(pos),
                    "raw_valence": float(val),
                    "polarity": polarity_name(val),
                })

    return {
        "name": name,
        "rows": rows,
        "embeddings": np.concatenate(embeddings, axis=0),
        "attn_reg": np.concatenate(attn_reg, axis=0),
        "attn_ord": np.concatenate(attn_ord, axis=0),
        "routing": {k: np.concatenate(v, axis=0) for k, v in routing.items()},
        "layout": layout or {},
        "checkpoint": str(checkpoint_path),
    }


def fit_reducer(without_data, full_data, method, seed, components):
    x = np.concatenate([without_data["embeddings"], full_data["embeddings"]], axis=0)
    reducer_name = method
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(
                n_components=components,
                n_neighbors=30,
                min_dist=0.12,
                metric="cosine",
                random_state=seed,
            )
            xy = reducer.fit_transform(x)
        except Exception as exc:
            print(f"UMAP unavailable ({exc}); falling back to PCA.", flush=True)
            reducer_name = "pca"
            from sklearn.decomposition import PCA
            xy = PCA(n_components=components, random_state=seed).fit_transform(x)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=components, init="pca", learning_rate="auto", perplexity=40, random_state=seed).fit_transform(x)
    else:
        reducer_name = "pca"
        from sklearn.decomposition import PCA
        xy = PCA(n_components=components, random_state=seed).fit_transform(x)
    n0 = len(without_data["rows"])
    without_data["xy"] = xy[:n0]
    full_data["xy"] = xy[n0:]
    return reducer_name


def plot_embedding(data, title, path):
    plt = import_matplotlib()
    rows = data["rows"]
    xy = data["xy"]
    valence = np.array([r["raw_valence"] for r in rows], dtype=float)
    is_3d = xy.shape[1] == 3
    if is_3d:
        fig = plt.figure(figsize=(7.5, 6.4), dpi=180)
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=22, azim=-58)
    else:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=180)

    by_base = defaultdict(list)
    for idx, row in enumerate(rows):
        by_base[row["base_id"]].append(idx)
    drawn = 0
    for indices in by_base.values():
        if len(indices) < 2:
            continue
        indices = sorted(indices, key=lambda i: (rows[i]["temporal_pos"], rows[i]["clip_idx"]))
        if drawn >= 350:
            break
        pts = xy[indices]
        if is_3d:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#222222", alpha=0.08, linewidth=0.45, zorder=1)
        else:
            ax.plot(pts[:, 0], pts[:, 1], color="#222222", alpha=0.08, linewidth=0.45, zorder=1)
        drawn += 1

    scatter_kwargs = {
        "c": valence,
        "s": 7,
        "cmap": "coolwarm",
        "vmin": -3,
        "vmax": 3,
        "alpha": 0.78,
        "linewidths": 0,
        "zorder": 2,
    }
    if is_3d:
        sc = ax.scatter(xy[:, 0], xy[:, 1], xy[:, 2], depthshade=False, **scatter_kwargs)
    else:
        sc = ax.scatter(xy[:, 0], xy[:, 1], **scatter_kwargs)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Raw valence")
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    if is_3d:
        ax.set_zlabel("Dimension 3")
    ax.grid(True, alpha=0.16, linewidth=0.4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def group_indices(rows):
    out = {"all": np.arange(len(rows))}
    for group in ("negative", "neutral", "positive"):
        out[group] = np.array([i for i, row in enumerate(rows) if row["polarity"] == group], dtype=int)
    return out


def mean_or_zeros(array, indices):
    if indices.size == 0:
        return np.zeros(array.shape[1], dtype=float)
    return array[indices].mean(axis=0)


def group_share(vector, layout, group_name):
    group = layout.get(group_name, {}) if isinstance(layout, dict) else {}
    indices = [int(i) for i in group.get("indices", []) if 0 <= int(i) < len(vector)]
    return float(vector[indices].sum()) if indices else 0.0


def modality_specific_group(modality):
    return {
        "vision": "vision_specific",
        "text": "text_specific",
        "audio": "audio_specific",
    }[modality]


def plot_expert_heatmap(data, path):
    plt = import_matplotlib()
    usage = np.stack([data["routing"][m].mean(axis=0) for m in MODALITIES], axis=0)
    fig, ax = plt.subplots(figsize=(8.6, 3.7), dpi=180)
    im = ax.imshow(usage, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(MODALITIES)))
    ax.set_yticklabels(MODALITIES)
    ax.set_xticks(np.arange(usage.shape[1]))
    ax.set_xticklabels([f"E{i}" for i in range(usage.shape[1])], rotation=90)
    ax.set_title("Full HTCM expert usage")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Average routing probability")
    layout = data["layout"]
    for group_name in ("text_specific", "audio_specific", "vision_specific", "shared", "temporal_contrast"):
        group = layout.get(group_name, {})
        end = group.get("end")
        if isinstance(end, int) and 0 < end < usage.shape[1]:
            ax.axvline(end - 0.5, color="white", linewidth=1.2, alpha=0.9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_shared_specific_ratio(data, path):
    plt = import_matplotlib()
    layout = data["layout"]
    indices_by_group = group_indices(data["rows"])
    colors = {
        "specific": "#4c78a8",
        "shared": "#54a24b",
        "temporal": "#e45756",
        "other": "#b7b7b7",
    }
    labels = []
    stacks = {key: [] for key in colors}
    for modality in MODALITIES:
        for polarity in POLARITIES:
            vector = mean_or_zeros(data["routing"][modality], indices_by_group[polarity])
            specific = group_share(vector, layout, modality_specific_group(modality))
            shared = group_share(vector, layout, "shared")
            temporal = group_share(vector, layout, "temporal_contrast")
            other = max(0.0, 1.0 - specific - shared - temporal)
            labels.append(f"{modality}\n{polarity}")
            stacks["specific"].append(specific)
            stacks["shared"].append(shared)
            stacks["temporal"].append(temporal)
            stacks["other"].append(other)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11.0, 4.2), dpi=180)
    bottom = np.zeros(len(labels), dtype=float)
    for key in ("specific", "shared", "temporal", "other"):
        values = np.array(stacks[key], dtype=float)
        ax.bar(x, values, bottom=bottom, label=key, color=colors[key], width=0.72)
        bottom += values
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Routing probability share")
    ax.set_title("Full HTCM shared/specific routing ratio by polarity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_task_query_attention(data, path):
    plt = import_matplotlib()
    groups = group_indices(data["rows"])
    fig, axes = plt.subplots(1, 4, figsize=(12.2, 3.6), dpi=180, sharey=True)
    x = np.arange(len(MODALITIES))
    width = 0.36
    for ax, group_name in zip(axes, POLARITIES):
        idx = groups[group_name]
        reg = mean_or_zeros(data["attn_reg"], idx)
        ordn = mean_or_zeros(data["attn_ord"], idx)
        ax.bar(x - width / 2, reg, width=width, label="regression", color="#4c78a8")
        ax.bar(x + width / 2, ordn, width=width, label="ordinal", color="#f58518")
        ax.set_title(group_name)
        ax.set_xticks(x)
        ax.set_xticklabels(MODALITIES, rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.18, linewidth=0.5)
    axes[0].set_ylabel("Attention mass")
    axes[-1].legend(loc="upper right", frameon=False)
    fig.suptitle("Full HTCM task-query modality attention", y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(path)
    plt.close(fig)


def write_routing_csv(data, path):
    layout = data["layout"]
    groups = group_indices(data["rows"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "modality", "polarity", "expert", "probability", "expert_group"],
        )
        writer.writeheader()
        for modality in MODALITIES:
            for polarity in POLARITIES:
                vector = mean_or_zeros(data["routing"][modality], groups[polarity])
                for expert_idx, value in enumerate(vector):
                    group_name = "unknown"
                    for candidate, info in layout.items():
                        if isinstance(info, dict) and expert_idx in info.get("indices", []):
                            group_name = candidate
                            break
                    writer.writerow({
                        "variant": data["name"],
                        "modality": modality,
                        "polarity": polarity,
                        "expert": expert_idx,
                        "probability": float(value),
                        "expert_group": group_name,
                    })


def write_attention_csv(data, path):
    groups = group_indices(data["rows"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["variant", "query", "polarity", "modality", "attention"],
        )
        writer.writeheader()
        for query, matrix in (("regression", data["attn_reg"]), ("ordinal", data["attn_ord"])):
            for polarity in POLARITIES:
                vector = mean_or_zeros(matrix, groups[polarity])
                for modality, value in zip(MODALITIES, vector):
                    writer.writerow({
                        "variant": data["name"],
                        "query": query,
                        "polarity": polarity,
                        "modality": modality,
                        "attention": float(value),
                    })


def knn_valence_smoothness(embeddings, rows, k=10):
    from sklearn.neighbors import NearestNeighbors
    valence = np.array([r["raw_valence"] for r in rows], dtype=float)
    n_neighbors = min(k + 1, len(rows))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    diffs = []
    for i, neigh in enumerate(indices):
        neigh = [j for j in neigh if j != i]
        if neigh:
            diffs.append(np.abs(valence[neigh] - valence[i]).mean())
    return float(np.mean(diffs)) if diffs else float("nan")


def temporal_continuity_distance(embeddings, rows):
    by_base = defaultdict(list)
    for idx, row in enumerate(rows):
        by_base[row["base_id"]].append(idx)
    distances = []
    for indices in by_base.values():
        if len(indices) < 2:
            continue
        indices = sorted(indices, key=lambda i: (rows[i]["temporal_pos"], rows[i]["clip_idx"]))
        for left, right in zip(indices[:-1], indices[1:]):
            distances.append(float(np.linalg.norm(embeddings[left] - embeddings[right])))
    return float(np.mean(distances)) if distances else float("nan")


def polarity_silhouette(embeddings, rows):
    from sklearn.metrics import silhouette_score
    labels = np.array([{"negative": 0, "neutral": 1, "positive": 2}[r["polarity"]] for r in rows], dtype=int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def top_experts_text(data, topk=3):
    parts = []
    layout = data["layout"]
    for modality in MODALITIES:
        vector = data["routing"][modality].mean(axis=0)
        pairs = sorted(enumerate(vector), key=lambda item: float(item[1]), reverse=True)[:topk]
        labels = []
        for idx, value in pairs:
            group_name = "unknown"
            for candidate, info in layout.items():
                if isinstance(info, dict) and idx in info.get("indices", []):
                    group_name = candidate
                    break
            labels.append(f"E{idx} ({group_name}, {float(value):.3f})")
        parts.append(f"- {modality}: " + ", ".join(labels))
    return "\n".join(parts)


def analysis_direction(without_value, full_value, lower_is_better):
    if math.isnan(without_value) or math.isnan(full_value):
        return "inconclusive"
    improved = full_value < without_value if lower_is_better else full_value > without_value
    return "improved" if improved else "not improved"


def write_qualitative(without_data, full_data, reducer_name, path):
    metrics = {
        without_data["name"]: {
            "knn_valence_abs_diff": knn_valence_smoothness(without_data["embeddings"], without_data["rows"]),
            "temporal_continuity_distance": temporal_continuity_distance(without_data["embeddings"], without_data["rows"]),
            "polarity_silhouette": polarity_silhouette(without_data["embeddings"], without_data["rows"]),
        },
        full_data["name"]: {
            "knn_valence_abs_diff": knn_valence_smoothness(full_data["embeddings"], full_data["rows"]),
            "temporal_continuity_distance": temporal_continuity_distance(full_data["embeddings"], full_data["rows"]),
            "polarity_silhouette": polarity_silhouette(full_data["embeddings"], full_data["rows"]),
        },
    }
    w = metrics[without_data["name"]]
    f = metrics[full_data["name"]]
    valence_result = analysis_direction(w["knn_valence_abs_diff"], f["knn_valence_abs_diff"], lower_is_better=True)
    temporal_result = analysis_direction(w["temporal_continuity_distance"], f["temporal_continuity_distance"], lower_is_better=True)
    silhouette_result = analysis_direction(w["polarity_silhouette"], f["polarity_silhouette"], lower_is_better=False)

    groups = group_indices(full_data["rows"])
    full_reg = mean_or_zeros(full_data["attn_reg"], groups["all"])
    full_ord = mean_or_zeros(full_data["attn_ord"], groups["all"])
    reg_vision = float(full_reg[0])
    reg_text = float(full_reg[1])
    reg_audio = float(full_reg[2])
    ord_vision = float(full_ord[0])
    ord_text = float(full_ord[1])
    ord_audio = float(full_ord[2])
    if reg_audio > ord_audio:
        audio_query_conclusion = "Regression assigns more attention to audio than ordinal."
    elif reg_audio < ord_audio:
        audio_query_conclusion = "Ordinal assigns more attention to audio than regression."
    else:
        audio_query_conclusion = "Regression and ordinal assign identical audio attention."
    if ord_vision > reg_vision:
        vision_query_conclusion = "Ordinal assigns more attention to vision than regression."
    elif ord_vision < reg_vision:
        vision_query_conclusion = "Regression assigns more attention to vision than ordinal."
    else:
        vision_query_conclusion = "Regression and ordinal assign identical vision attention."
    if ord_text > reg_text:
        text_query_conclusion = "Ordinal assigns more attention to text than regression."
    elif ord_text < reg_text:
        text_query_conclusion = "Regression assigns more attention to text than ordinal."
    else:
        text_query_conclusion = "Regression and ordinal assign identical text attention."
    text_min = min(reg_text, ord_text)
    text_max = max(reg_text, ord_text)
    layout = full_data["layout"]
    shared = {
        modality: group_share(full_data["routing"][modality].mean(axis=0), layout, "shared")
        for modality in MODALITIES
    }
    temporal = {
        modality: group_share(full_data["routing"][modality].mean(axis=0), layout, "temporal_contrast")
        for modality in MODALITIES
    }
    payload = {
        "reducer": reducer_name,
        "metrics": metrics,
        "full_attention_all": {
            "regression": dict(zip(MODALITIES, [float(x) for x in full_reg])),
            "ordinal": dict(zip(MODALITIES, [float(x) for x in full_ord])),
        },
        "full_shared_ratio": shared,
        "full_temporal_ratio": temporal,
        "task_query_conclusion": {
            "vision": vision_query_conclusion,
            "audio": audio_query_conclusion,
            "text": text_query_conclusion,
        },
        "checkpoints": {
            without_data["name"]: without_data["checkpoint"],
            full_data["name"]: full_data["checkpoint"],
        },
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    text = f"""# StructV7.5.4 qualitative visualization analysis

Reducer: `{reducer_name}`. Dataset: CMU-MOSEI test split.

## Embedding space

- kNN valence smoothness, lower is more locally ordered: `{without_data['name']}` = {w['knn_valence_abs_diff']:.4f}, `{full_data['name']}` = {f['knn_valence_abs_diff']:.4f} ({valence_result}).
- Consecutive-segment distance, lower is more temporally continuous: `{without_data['name']}` = {w['temporal_continuity_distance']:.4f}, `{full_data['name']}` = {f['temporal_continuity_distance']:.4f} ({temporal_result}).
- Polarity silhouette, higher means clearer affective regions: `{without_data['name']}` = {w['polarity_silhouette']:.4f}, `{full_data['name']}` = {f['polarity_silhouette']:.4f} ({silhouette_result}).

Full HTCM should be described as forming a more continuous and ordered affective manifold only if the first two metrics improve and the UMAP plot shows smoother valence transitions without isolated polarity islands.

## Expert routing

Top full-HTCM experts by modality:

{top_experts_text(full_data)}

Shared routing ratios in Full HTCM: {json.dumps(shared, ensure_ascii=False)}

Temporal expert ratios in Full HTCM: {json.dumps(temporal, ensure_ascii=False)}

Experts with high modality-specific mass support specialization; experts in the shared group with non-trivial mass across modalities are the strongest evidence for shared affective semantics.

## Task query attention

- Regression query all-sample modality attention: {dict(zip(MODALITIES, [round(float(x), 4) for x in full_reg]))}
- Ordinal query all-sample modality attention: {dict(zip(MODALITIES, [round(float(x), 4) for x in full_ord]))}

Measured conclusion: {vision_query_conclusion} {audio_query_conclusion} {text_query_conclusion} Both task queries are dominated by vision, with text attention in the {text_min:.1%}-{text_max:.1%} range, so this visualization does not support the claim that the ordinal query is text-polarity focused. Text can still enter the prediction through the pooled text representation concatenated into each task head, so low task-query text attention should not be interpreted as the whole model ignoring text.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--without_checkpoint", required=True)
    parser.add_argument("--without_hparams", default="")
    parser.add_argument("--full_checkpoint", required=True)
    parser.add_argument("--full_hparams", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--components", type=int, choices=[2, 3], default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    without_hparams, without_hparams_path = load_hparams(args.without_checkpoint, args.without_hparams)
    full_hparams, full_hparams_path = load_hparams(args.full_checkpoint, args.full_hparams)
    print(f"Loaded without hparams: {without_hparams_path}", flush=True)
    print(f"Loaded full hparams: {full_hparams_path}", flush=True)

    without_data = collect_checkpoint("without_ltemp", args.without_checkpoint, without_hparams, args)
    full_data = collect_checkpoint("full_htcm", args.full_checkpoint, full_hparams, args)
    reducer_name = fit_reducer(without_data, full_data, args.method, args.seed, args.components)

    plot_embedding(without_data, "w/o Ltemp affective representation", out_dir / "figure_umap_without_ltemp.png")
    plot_embedding(full_data, "Full HTCM affective representation", out_dir / "figure_umap_full_htcm.png")
    plot_expert_heatmap(full_data, out_dir / "figure_expert_usage_heatmap.png")
    plot_shared_specific_ratio(full_data, out_dir / "figure_shared_specific_ratio.png")
    plot_task_query_attention(full_data, out_dir / "figure_task_query_attention.png")
    write_routing_csv(full_data, out_dir / "expert_routing_summary_full_htcm.csv")
    write_attention_csv(full_data, out_dir / "task_query_attention_summary_full_htcm.csv")
    write_qualitative(without_data, full_data, reducer_name, out_dir / "qualitative_analysis.md")

    manifest = {
        "output_dir": str(out_dir),
        "figures": [
            "figure_umap_without_ltemp.png",
            "figure_umap_full_htcm.png",
            "figure_expert_usage_heatmap.png",
            "figure_shared_specific_ratio.png",
            "figure_task_query_attention.png",
        ],
        "without_checkpoint": args.without_checkpoint,
        "full_checkpoint": args.full_checkpoint,
        "sample_limit": args.sample_limit,
        "reducer": reducer_name,
        "components": args.components,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
