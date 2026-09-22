import torch
from torch.utils.data import Dataset

from train_emotion import (
    TEMPORAL_KERNEL_CHOICES,
    TemporalGroupWindowBatchSampler,
    compute_soft_temporal_contrastive_loss,
    compute_temporal_positive_weights,
)


class ToyTemporalDataset(Dataset):
    def __init__(self):
        self.temporal_metadata = [
            {"temporal_group_id": 0, "temporal_pos": 0.0},
            {"temporal_group_id": 0, "temporal_pos": 1.0},
            {"temporal_group_id": 0, "temporal_pos": 2.0},
            {"temporal_group_id": 1, "temporal_pos": 0.0},
            {"temporal_group_id": 1, "temporal_pos": 1.0},
            {"temporal_group_id": 2, "temporal_pos": 0.0},
        ]

    def __len__(self):
        return len(self.temporal_metadata)

    def __getitem__(self, idx):
        return idx


def test_temporal_kernels_are_finite_and_masked():
    embeddings = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0],
        ],
        requires_grad=True,
    )
    group_id = torch.tensor([0, 0, 0, 1, 1, 2])
    pos = torch.tensor([0.0, 1.0, 3.0, 0.0, 1.0, 0.0])
    raw_valence = torch.tensor([-1.0, -0.8, 0.5, 1.0, 1.2, 0.0])
    for kernel in TEMPORAL_KERNEL_CHOICES:
        kwargs = (
            {"raw_valence": raw_valence, "label_gate_beta": 1.0}
            if kernel in {"label_exp", "sign_safe_exp", "legacy_sign_safe"}
            else {}
        )
        weights, stats = compute_temporal_positive_weights(
            embeddings,
            group_id,
            pos,
            decay_tau=1.0,
            positive_radius=1.0,
            weak_positive_radius=4.0,
            temporal_kernel=kernel,
            **kwargs,
        )
        assert weights.shape == (6, 6)
        assert torch.all(weights.diag() == 0)
        assert torch.all(weights[group_id.unsqueeze(0) != group_id.unsqueeze(1)] == 0)
        assert stats["positive_pair_count"] > 0
        loss, loss_stats = compute_soft_temporal_contrastive_loss(
            embeddings,
            group_id,
            pos,
            decay_tau=1.0,
            positive_radius=1.0,
            weak_positive_radius=4.0,
            temporal_kernel=kernel,
            **kwargs,
        )
        assert torch.isfinite(loss)
        assert loss_stats["temporal_kernel"] == kernel


def test_semantic_gate_target_has_no_gradient():
    embeddings = torch.randn(5, 4, requires_grad=True)
    group_id = torch.tensor([0, 0, 0, 1, 1])
    pos = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0])
    weights, _ = compute_temporal_positive_weights(
        embeddings,
        group_id,
        pos,
        temporal_kernel="semantic_exp",
        weak_positive_radius=4.0,
    )
    assert not weights.requires_grad


def test_label_gate_targets_have_no_gradient():
    embeddings = torch.randn(5, 4, requires_grad=True)
    raw_valence = torch.tensor([-1.0, -0.5, 0.2, 0.7, 0.0], requires_grad=True)
    group_id = torch.tensor([0, 0, 0, 1, 1])
    pos = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0])
    for kernel in ["label_exp", "sign_safe_exp", "legacy_sign_safe"]:
        weights, _ = compute_temporal_positive_weights(
            embeddings,
            group_id,
            pos,
            temporal_kernel=kernel,
            weak_positive_radius=4.0,
            raw_valence=raw_valence,
            label_gate_beta=1.0,
        )
        assert not weights.requires_grad


def test_legacy_sign_safe_is_legacy_exp_times_explicit_sign_gate():
    embeddings = torch.randn(6, 4)
    group_id = torch.zeros(6, dtype=torch.long)
    pos = torch.arange(6, dtype=torch.float32)
    raw_valence = torch.tensor([-1.0, -0.2, 0.0, 0.0, 0.3, 2.0])
    bridge = 0.25

    legacy, _ = compute_temporal_positive_weights(
        embeddings,
        group_id,
        pos,
        decay_tau=1.5,
        positive_radius=1.0,
        weak_positive_radius=8.0,
        min_positive_weight=0.2,
        temporal_kernel="legacy_exp",
    )
    sign_safe, _ = compute_temporal_positive_weights(
        embeddings,
        group_id,
        pos,
        decay_tau=1.5,
        positive_radius=1.0,
        weak_positive_radius=8.0,
        min_positive_weight=0.2,
        temporal_kernel="legacy_sign_safe",
        raw_valence=raw_valence,
        zero_bridge_weight=bridge,
    )

    zero = raw_valence.abs() <= 1e-12
    both_zero = zero.unsqueeze(1) & zero.unsqueeze(0)
    one_zero = zero.unsqueeze(1) ^ zero.unsqueeze(0)
    same_nonzero_sign = (
        torch.sign(raw_valence).unsqueeze(1).eq(torch.sign(raw_valence).unsqueeze(0))
        & ~zero.unsqueeze(1)
        & ~zero.unsqueeze(0)
    )
    gate = torch.zeros_like(legacy)
    gate[same_nonzero_sign | both_zero] = 1.0
    gate[one_zero] = bridge

    assert torch.allclose(sign_safe, legacy * gate)
    assert torch.allclose(sign_safe[0, 1], legacy[0, 1])  # same non-zero sign
    assert torch.allclose(sign_safe[2, 3], legacy[2, 3])  # zero-zero
    assert torch.allclose(sign_safe[1, 2], legacy[1, 2] * bridge)  # one-zero bridge
    assert sign_safe[1, 4].item() == 0.0  # opposite non-zero signs


def test_legacy_exp_matches_historical_piecewise_kernel():
    embeddings = torch.randn(5, 3)
    group_id = torch.tensor([0, 0, 0, 0, 1])
    pos = torch.tensor([0.0, 0.5, 2.0, 5.0, 0.0])
    tau = 1.25
    positive_radius = 1.0
    weak_positive_radius = 4.0
    min_positive_weight = 0.2

    actual, _ = compute_temporal_positive_weights(
        embeddings,
        group_id,
        pos,
        decay_tau=tau,
        positive_radius=positive_radius,
        weak_positive_radius=weak_positive_radius,
        min_positive_weight=min_positive_weight,
        temporal_kernel="legacy_exp",
    )

    distance = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()
    same_group = group_id.unsqueeze(0).eq(group_id.unsqueeze(1))
    not_self = ~torch.eye(pos.numel(), dtype=torch.bool)
    exp_weights = torch.exp(-distance / tau)
    expected = torch.zeros_like(exp_weights)
    strong = same_group & not_self & (distance <= positive_radius)
    weak = same_group & not_self & (distance > positive_radius) & (distance <= weak_positive_radius)
    expected = torch.where(strong, exp_weights.clamp_min(min_positive_weight), expected)
    expected = torch.where(weak, exp_weights, expected)

    assert torch.allclose(actual, expected)


def test_temporal_group_window_sampler_is_reproducible_and_grouped():
    dataset = ToyTemporalDataset()
    sampler_a = TemporalGroupWindowBatchSampler(dataset, batch_size=3, temporal_window=2.0, seed=123)
    sampler_b = TemporalGroupWindowBatchSampler(dataset, batch_size=3, temporal_window=2.0, seed=123)
    batches_a = list(iter(sampler_a))
    batches_b = list(iter(sampler_b))
    assert batches_a == batches_b
    assert sum(len(batch) for batch in batches_a) == len(dataset)
    grouped_batches = [
        batch
        for batch in batches_a
        if len({dataset.temporal_metadata[idx]["temporal_group_id"] for idx in batch}) == 1
    ]
    assert grouped_batches
    for batch in grouped_batches:
        if len(batch) > 1:
            positions = [dataset.temporal_metadata[idx]["temporal_pos"] for idx in batch]
            assert max(positions) - min(positions) <= 2.0 + 1e-9


def main():
    test_temporal_kernels_are_finite_and_masked()
    test_semantic_gate_target_has_no_gradient()
    test_label_gate_targets_have_no_gradient()
    test_legacy_sign_safe_is_legacy_exp_times_explicit_sign_gate()
    test_legacy_exp_matches_historical_piecewise_kernel()
    test_temporal_group_window_sampler_is_reproducible_and_grouped()
    print("temporal kernel and sampler tests passed")


if __name__ == "__main__":
    main()
