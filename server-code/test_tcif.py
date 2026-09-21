import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from datasets.emotion_dataset import build_temporal_context_index
from models.models_emotion import EmotionM4OE
from models.tcif import TemporalContextInnovationFilter
from test_signed_reg_cls7_head import build_tiny_model, tiny_batch
from train_emotion import (
    _compute_tcif_context_aux_loss,
    _compute_tcif_transition_gate_loss,
)


def _toy_temporal_metadata():
    ids = ["video_a_0", "video_a_1", "video_a_2", "video_b_0", "video_b_1"]
    metadata = {
        "video_a_0": {"temporal_group_id": 0, "temporal_pos": 0.0},
        "video_a_1": {"temporal_group_id": 0, "temporal_pos": 1.0},
        "video_a_2": {"temporal_group_id": 0, "temporal_pos": 3.0},
        "video_b_0": {"temporal_group_id": 1, "temporal_pos": 0.0},
        "video_b_1": {"temporal_group_id": 1, "temporal_pos": 2.0},
    }
    return ids, metadata


def test_backbone_loader_falls_back_from_metadata_less_safetensors_to_bin():
    marker = torch.nn.Identity()
    with tempfile.TemporaryDirectory() as directory:
        Path(directory, "pytorch_model.bin").touch()
        with patch(
            "models.models_emotion.AutoModel.from_pretrained",
            side_effect=[
                AttributeError("metadata-less safetensors"),
                AttributeError("'NoneType' object has no attribute 'get'"),
                marker,
            ],
        ) as loader:
            result = EmotionM4OE._load_backbone(
                directory,
                local_files_only=True,
                fallback_model=torch.nn.Identity(),
            )
        assert result is marker
        assert loader.call_count == 3
        assert loader.call_args.kwargs["use_safetensors"] is False


def _tiny_tcif_context(batch, context_count=2):
    context = {
        "tcif_context_images": torch.stack(
            [batch["image"] + float(i + 1) for i in range(context_count)],
            dim=1,
        ),
        "tcif_context_input_ids": torch.stack(
            [batch["input_ids"] + i + 1 for i in range(context_count)],
            dim=1,
        ),
        "tcif_context_attention_mask": torch.stack(
            [batch["attention_mask"] for _ in range(context_count)],
            dim=1,
        ),
        "tcif_context_audio_values": torch.stack(
            [batch["audio_values"] + float(i + 1) for i in range(context_count)],
            dim=1,
        ),
        "tcif_context_audio_attention_mask": torch.stack(
            [batch["audio_attention_mask"] for _ in range(context_count)],
            dim=1,
        ),
        "tcif_context_valid_mask": torch.tensor(
            [[True, True], [True, False]],
            dtype=torch.bool,
        ),
        "tcif_context_relative_pos": torch.tensor(
            [[-1.0, 1.0], [-1.0, 0.0]],
            dtype=torch.float32,
        ),
    }
    return context


def test_temporal_context_index_masks_center_and_boundaries():
    ids, metadata = _toy_temporal_metadata()
    index = build_temporal_context_index(ids, metadata, radius=1, mode="neighbors")
    first = index[0]
    middle = index[1]
    last = index[2]
    assert [row["valid"] for row in first] == [False, True]
    assert [row["index"] for row in first] == [0, 1]
    assert [row["valid"] for row in middle] == [True, True]
    assert [row["index"] for row in middle] == [0, 2]
    assert [row["relative_pos"] for row in middle] == [-1.0, 2.0]
    assert [row["valid"] for row in last] == [True, False]
    for center_idx, rows in index.items():
        for row in rows:
            if row["valid"]:
                assert row["index"] != center_idx


def test_shuffled_context_is_deterministic_and_cross_group():
    ids, metadata = _toy_temporal_metadata()
    neighbors = build_temporal_context_index(
        ids,
        metadata,
        radius=1,
        mode="neighbors",
        seed=9,
    )
    shuffled_a = build_temporal_context_index(
        ids,
        metadata,
        radius=1,
        mode="shuffled",
        seed=9,
    )
    shuffled_b = build_temporal_context_index(
        ids,
        metadata,
        radius=1,
        mode="shuffled",
        seed=9,
    )
    assert shuffled_a == shuffled_b
    for center_idx, rows in shuffled_a.items():
        center_group = metadata[ids[center_idx]]["temporal_group_id"]
        assert [row["valid"] for row in rows] == [
            row["valid"] for row in neighbors[center_idx]
        ]
        assert [row["relative_pos"] for row in rows] == [
            row["relative_pos"] for row in neighbors[center_idx]
        ]
        for row in rows:
            if row["valid"]:
                context_group = metadata[ids[row["index"]]]["temporal_group_id"]
                assert context_group != center_group


def test_tcif_prior_is_center_blind_and_zero_init_is_exact():
    torch.manual_seed(5)
    layer = TemporalContextInnovationFilter(feature_dim=8, latent_dim=4)
    context = torch.randn(2, 2, 8)
    valid = torch.tensor([[True, True], [True, False]])
    relative_pos = torch.tensor([[-1.0, 1.0], [-1.0, 0.0]])
    local_a = torch.randn(2, 8)
    local_b = torch.randn(2, 8) * 4.0
    out_a = layer(local_a, context, valid, relative_pos, output_mode="posterior")
    out_b = layer(local_b, context, valid, relative_pos, output_mode="posterior")
    assert torch.equal(out_a["prior_mean"], out_b["prior_mean"])
    assert torch.equal(out_a["context_attention"], out_b["context_attention"])
    assert torch.equal(out_a["feature"], local_a)
    context_mode = layer(local_a, context, valid, relative_pos, output_mode="context")
    assert torch.equal(context_mode["feature"], local_a)
    assert torch.isfinite(out_a["posterior_mean"]).all()
    assert torch.isfinite(out_a["posterior_variance"]).all()


def test_tcif_enabled_model_starts_from_identical_prediction_function():
    torch.manual_seed(123)
    baseline = build_tiny_model("signed_reg_cls7")
    torch.manual_seed(123)
    tcif_model = build_tiny_model(
        "signed_reg_cls7",
        enable_tcif=True,
        tcif_latent_dim=4,
        tcif_output_mode="posterior",
    )
    assert not any(key.startswith("tcif_") for key in baseline.state_dict())
    common_tcif_state = {
        key: value
        for key, value in tcif_model.state_dict().items()
        if key in baseline.state_dict()
    }
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, common_tcif_state[key]), key

    baseline.eval()
    tcif_model.eval()
    batch = tiny_batch()
    context = _tiny_tcif_context(batch)
    with torch.no_grad():
        baseline_out = baseline(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["audio_values"],
            batch["audio_attention_mask"],
        )
        tcif_out = tcif_model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["audio_values"],
            batch["audio_attention_mask"],
            **context,
        )
    assert torch.equal(tcif_out["y_reg"], baseline_out["y_reg"])
    assert torch.equal(tcif_out["cls7_logits"], baseline_out["cls7_logits"])
    payload = tcif_out["extras"]["tcif"]
    assert tuple(payload["context_y_reg"].shape) == (2,)
    assert tuple(payload["context_cls7_logits"].shape) == (2, 7)
    assert tuple(payload["regression_attention"].shape) == (2, 2)


def test_tcif_posterior_and_context_auxiliary_receive_gradients():
    torch.manual_seed(321)
    model = build_tiny_model(
        "signed_reg_cls7",
        enable_tcif=True,
        tcif_latent_dim=4,
        tcif_output_mode="posterior",
    )
    model.train()
    batch = tiny_batch()
    out = model(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
        batch["audio_values"],
        batch["audio_attention_mask"],
        **_tiny_tcif_context(batch),
    )
    args = SimpleNamespace(
        reg_loss_type="smooth_l1",
        tcif_context_aux_cls7_weight=0.5,
    )
    context_total, context_reg, context_cls7, valid_count = (
        _compute_tcif_context_aux_loss(out, batch["raw_valence"], args)
    )
    loss = (
        torch.nn.functional.smooth_l1_loss(out["y_reg"], batch["raw_valence"])
        + torch.nn.functional.cross_entropy(
            out["cls7_logits"],
            torch.tensor([1, 4], dtype=torch.long),
        )
        + 0.1 * context_total
    )
    loss.backward()
    assert valid_count == 2
    assert torch.isfinite(context_reg)
    assert torch.isfinite(context_cls7)
    assert model.tcif_regression.output_projection.weight.grad is not None
    assert model.tcif_regression.output_projection.weight.grad.abs().sum() > 0
    assert model.tcif_context_reg_head.weight.grad is not None
    assert model.tcif_context_reg_head.weight.grad.abs().sum() > 0
    assert model.tcif_context_cls7_head.weight.grad is not None
    assert model.tcif_context_cls7_head.weight.grad.abs().sum() > 0


def test_transition_gate_routes_between_local_and_posterior_experts():
    torch.manual_seed(99)
    layer = TemporalContextInnovationFilter(
        feature_dim=8,
        latent_dim=4,
        enable_transition_gate=True,
        transition_gate_hidden_dim=6,
        transition_gate_init_bias=2.0,
    )
    torch.nn.init.normal_(layer.output_projection.weight, std=0.1)
    torch.nn.init.zeros_(layer.output_projection.bias)
    local = torch.randn(2, 8)
    context = torch.randn(2, 2, 8)
    valid = torch.tensor([[True, True], [False, False]])
    relative_pos = torch.tensor([[-1.0, 1.0], [0.0, 0.0]])

    torch.nn.init.zeros_(layer.transition_gate[-1].weight)
    torch.nn.init.constant_(layer.transition_gate[-1].bias, -20.0)
    transition = layer(local, context, valid, relative_pos)
    assert torch.allclose(transition["feature"], local, atol=1e-6)
    assert transition["continuation_gate"][0] < 1e-6
    assert transition["continuation_gate"][1] == 0.0

    torch.nn.init.constant_(layer.transition_gate[-1].bias, 20.0)
    continuation = layer(local, context, valid, relative_pos)
    assert continuation["continuation_gate"][0] > 1.0 - 1e-6
    assert not torch.allclose(continuation["feature"][0], local[0])
    assert torch.equal(continuation["feature"][1], local[1])


def test_transition_gate_supervision_marks_sign_conflict_as_transition():
    torch.manual_seed(1001)
    model = build_tiny_model(
        "signed_reg_cls7",
        enable_tcif=True,
        tcif_latent_dim=4,
        tcif_output_mode="posterior",
        tcif_enable_transition_gate=True,
        tcif_transition_gate_hidden_dim=6,
    )
    model.train()
    batch = tiny_batch()
    context = _tiny_tcif_context(batch)
    output = model(
        batch["image"],
        batch["input_ids"],
        batch["attention_mask"],
        batch["audio_values"],
        batch["audio_attention_mask"],
        **context,
    )
    gate_batch = {
        "tcif_context_valid_mask": context["tcif_context_valid_mask"],
        "tcif_context_raw_valence": torch.tensor(
            [[-2.0, -1.5], [-0.5, 1.0]],
            dtype=torch.float32,
        ),
    }
    args = SimpleNamespace(
        tcif_transition_gate_tau=0.75,
        tcif_transition_gate_conflict_target=0.0,
    )
    loss, stats = _compute_tcif_transition_gate_loss(
        output,
        gate_batch,
        batch["raw_valence"],
        args,
    )
    assert torch.isfinite(loss)
    assert stats["valid_count"] == 2
    assert stats["sign_conflict_count"] == 1
    loss.backward()
    gate_grad = model.tcif_regression.transition_gate[-1].weight.grad
    assert gate_grad is not None
    assert gate_grad.abs().sum() > 0


def main():
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"tcif tests passed: {len(tests)}")


if __name__ == "__main__":
    main()
