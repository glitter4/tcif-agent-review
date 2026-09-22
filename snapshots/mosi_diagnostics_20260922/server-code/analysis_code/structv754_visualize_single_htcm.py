#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from structv754_visualize_htcm import (  # noqa: E402
    MODALITIES,
    collect_checkpoint,
    fit_reducer,
    group_indices,
    group_share,
    knn_valence_smoothness,
    load_hparams,
    mean_or_zeros,
    plot_embedding,
    plot_expert_heatmap,
    plot_shared_specific_ratio,
    plot_task_query_attention,
    polarity_silhouette,
    temporal_continuity_distance,
    top_experts_text,
    write_attention_csv,
    write_routing_csv,
)


def write_single_qualitative(data, reducer_name, components, path):
    rows = data["rows"]
    metrics = {
        "knn_valence_abs_diff": knn_valence_smoothness(data["embeddings"], rows),
        "temporal_continuity_distance": temporal_continuity_distance(data["embeddings"], rows),
        "polarity_silhouette": polarity_silhouette(data["embeddings"], rows),
    }
    groups = group_indices(rows)
    reg = mean_or_zeros(data["attn_reg"], groups["all"])
    ordn = mean_or_zeros(data["attn_ord"], groups["all"])
    layout = data["layout"]
    shared = {
        modality: group_share(data["routing"][modality].mean(axis=0), layout, "shared")
        for modality in MODALITIES
    }
    temporal = {
        modality: group_share(data["routing"][modality].mean(axis=0), layout, "temporal_contrast")
        for modality in MODALITIES
    }
    payload = {
        "reducer": reducer_name,
        "components": components,
        "metrics": metrics,
        "attention_all": {
            "regression": dict(zip(MODALITIES, [float(x) for x in reg])),
            "ordinal": dict(zip(MODALITIES, [float(x) for x in ordn])),
        },
        "shared_ratio": shared,
        "temporal_ratio": temporal,
        "checkpoint": data["checkpoint"],
    }
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    text_min = min(float(reg[1]), float(ordn[1]))
    text_max = max(float(reg[1]), float(ordn[1]))
    text = f"""# StructV7.5.4 single-model visualization analysis

Reducer: `{reducer_name}` with `{components}` components. Dataset: CMU-MOSEI test split.

Checkpoint: `{data['checkpoint']}`

## Embedding space

- kNN valence smoothness, lower is more locally ordered: {metrics['knn_valence_abs_diff']:.4f}.
- Consecutive-segment distance, lower is more temporally continuous: {metrics['temporal_continuity_distance']:.4f}.
- Polarity silhouette, higher means clearer affective regions: {metrics['polarity_silhouette']:.4f}.

These are descriptive single-model geometry diagnostics; without a paired control in the same reducer fit, they should not be used as evidence of improvement over w/o Ltemp.

## Expert routing

Top experts by modality:

{top_experts_text(data)}

Shared routing ratios: {json.dumps(shared, ensure_ascii=False)}

Temporal expert ratios: {json.dumps(temporal, ensure_ascii=False)}

In the no-text-shared setting, text shared ratio should be exactly 0.0. Audio/vision shared mass reflects the non-text modalities' use of shared affective experts.

## Task query attention

- Regression query all-sample modality attention: {dict(zip(MODALITIES, [round(float(x), 4) for x in reg]))}
- Ordinal query all-sample modality attention: {dict(zip(MODALITIES, [round(float(x), 4) for x in ordn]))}

Both task queries remain vision-dominant, with text attention in the {text_min:.1%}-{text_max:.1%} range.
"""
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hparams", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--components", type=int, choices=[2, 3], default=3)
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

    hparams, hparams_path = load_hparams(args.checkpoint, args.hparams)
    print(f"Loaded hparams: {hparams_path}", flush=True)
    data = collect_checkpoint("full_htcm", args.checkpoint, hparams, args)
    dummy = {
        "embeddings": np.empty((0, data["embeddings"].shape[1]), dtype=data["embeddings"].dtype),
        "rows": [],
    }
    reducer_name = fit_reducer(dummy, data, args.method, args.seed, args.components)

    plot_embedding(data, "Full HTCM affective representation", out_dir / "figure_umap3d_full_htcm.png")
    plot_expert_heatmap(data, out_dir / "figure_expert_usage_heatmap.png")
    plot_shared_specific_ratio(data, out_dir / "figure_shared_specific_ratio.png")
    plot_task_query_attention(data, out_dir / "figure_task_query_attention.png")
    write_routing_csv(data, out_dir / "expert_routing_summary_full_htcm.csv")
    write_attention_csv(data, out_dir / "task_query_attention_summary_full_htcm.csv")
    write_single_qualitative(data, reducer_name, args.components, out_dir / "qualitative_analysis.md")

    manifest = {
        "output_dir": str(out_dir),
        "figures": [
            "figure_umap3d_full_htcm.png",
            "figure_expert_usage_heatmap.png",
            "figure_shared_specific_ratio.png",
            "figure_task_query_attention.png",
        ],
        "checkpoint": args.checkpoint,
        "hparams": str(hparams_path),
        "sample_limit": args.sample_limit,
        "reducer": reducer_name,
        "components": args.components,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
