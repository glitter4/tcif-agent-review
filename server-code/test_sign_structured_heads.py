from types import SimpleNamespace

import torch

from head_utils import (
    cls7_expected_value,
    compute_final_prediction,
    continuous_to_cls7_hard,
    cumulative_logits_to_cls7_probs,
    get_cls7_centers,
    hier_sign_mag_to_cls7_probs,
    probs_to_logits,
)
from train_emotion import (
    REG_CLS_MAG_CONSISTENCY_BOUNDARIES,
    REG_CLS_MAG_CONSISTENCY_MODE,
    REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE,
    SlimTensorBoardWriter,
    _compute_hier_sign_mag_losses,
    _compute_reg_cls_mag_consistency_loss,
    _compute_sign_marginal_loss,
    _compute_signed_neutral_band_loss,
    _compute_zero_sign_margin_loss,
    _reg_cls_mag_consistency_decoded_endpoints,
    _sign_struct_scale,
)


def test_hier_sign_mag_reconstructs_valid_cls7_distribution():
    sign_logits = torch.randn(8, 3)
    neg_logits = torch.randn(8, 3)
    pos_logits = torch.randn(8, 3)
    probs = hier_sign_mag_to_cls7_probs(sign_logits, neg_logits, pos_logits)
    assert probs.shape == (8, 7)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)
    assert torch.all(probs >= 0)
    logits = probs_to_logits(probs)
    assert torch.allclose(torch.softmax(logits, dim=-1), probs, atol=1e-6)


def test_cumulative_head_reconstructs_valid_cls7_distribution():
    logits = torch.randn(8, 6)
    probs = cumulative_logits_to_cls7_probs(logits)
    assert probs.shape == (8, 7)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-6)
    assert torch.all(probs >= 0)


def test_structured_losses_are_finite_for_edge_batches():
    for raw in [
        torch.zeros(4),
        torch.tensor([1.0, 2.0, 0.5, 3.0]),
        torch.tensor([-1.0, -2.0, -0.5, -3.0]),
        torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
    ]:
        n = raw.numel()
        y_reg = torch.zeros(n)
        model_out = {
            "y_reg": y_reg,
            "extras": {
                "hier_sign_logits": torch.randn(n, 3, requires_grad=True),
                "hier_mag_neg_logits": torch.randn(n, 3, requires_grad=True),
                "hier_mag_pos_logits": torch.randn(n, 3, requires_grad=True),
            },
        }
        sign_loss, mag_loss = _compute_hier_sign_mag_losses(model_out, raw)
        assert torch.isfinite(sign_loss)
        assert torch.isfinite(mag_loss)


def test_sign_marginal_losses_are_finite_and_nonzero_only():
    args_focal = SimpleNamespace(sign_marginal_loss_type="focal", sign_marginal_focal_gamma=1.5)
    args_softf1 = SimpleNamespace(sign_marginal_loss_type="soft_f1", sign_marginal_focal_gamma=1.5)
    logits = torch.randn(6, 7, requires_grad=True)
    raw = torch.tensor([-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])
    for args in [args_focal, args_softf1]:
        loss = _compute_sign_marginal_loss(logits, raw, args)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
    zero_only = _compute_sign_marginal_loss(logits, torch.zeros(6), args_focal)
    assert torch.isfinite(zero_only)


def _neutral_band_args():
    return SimpleNamespace(
        signed_neutral_band_eta=0.4,
        signed_neutral_band_margin=1.0 / 6.0,
        signed_neutral_band_upper=0.45,
        signed_neutral_band_temperature=0.05,
    )


def _neutral_band_loss(raw, score):
    model_out = {
        "y_reg": score,
        "extras": {"y_cls_expected": score},
    }
    return _compute_signed_neutral_band_loss(model_out, raw, _neutral_band_args())


def test_signed_neutral_band_penalizes_wrong_and_insufficient_margin():
    raw = torch.tensor([-0.2, 0.2])
    wrong = _neutral_band_loss(raw, torch.tensor([0.3, -0.3]))
    insufficient = _neutral_band_loss(raw, torch.tensor([-0.05, 0.05]))
    correct = _neutral_band_loss(raw, torch.tensor([-0.3, 0.3]))

    assert wrong > insufficient
    assert insufficient > correct


def test_signed_neutral_band_penalizes_leaving_the_acc7_neutral_bin():
    raw = torch.tensor([-0.2, 0.2])
    inside = _neutral_band_loss(raw, torch.tensor([-0.3, 0.3]))
    far_outside = _neutral_band_loss(raw, torch.tensor([-1.2, 1.2]))
    assert far_outside > inside


def test_signed_neutral_band_excludes_exact_zero_labels():
    args = _neutral_band_args()
    raw_with_zero = torch.tensor([-0.2, 0.0, 0.2])
    with_zero = _compute_signed_neutral_band_loss(
        {
            "y_reg": torch.tensor([-0.3, 100.0, 0.3]),
            "extras": {"y_cls_expected": torch.tensor([-0.3, -100.0, 0.3])},
        },
        raw_with_zero,
        args,
    )
    without_zero = _compute_signed_neutral_band_loss(
        {
            "y_reg": torch.tensor([-0.3, 0.3]),
            "extras": {"y_cls_expected": torch.tensor([-0.3, 0.3])},
        },
        torch.tensor([-0.2, 0.2]),
        args,
    )

    assert torch.allclose(with_zero, without_zero)


def test_signed_neutral_band_backpropagates_only_through_tiny_nonzero_labels():
    y_reg = torch.zeros(4, requires_grad=True)
    y_cls_expected = torch.zeros(4, requires_grad=True)
    raw = torch.tensor([-0.2, 0.0, 0.2, 0.5])
    loss = _compute_signed_neutral_band_loss(
        {"y_reg": y_reg, "extras": {"y_cls_expected": y_cls_expected}},
        raw,
        _neutral_band_args(),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert y_reg.grad is not None
    assert y_cls_expected.grad is not None
    assert torch.all(y_reg.grad[[0, 2]].abs() > 0)
    assert torch.all(y_cls_expected.grad[[0, 2]].abs() > 0)
    assert torch.equal(y_reg.grad[[1, 3]], torch.zeros(2))
    assert torch.equal(y_cls_expected.grad[[1, 3]], torch.zeros(2))


def _zero_sign_margin_args():
    return SimpleNamespace(
        zero_sign_margin=0.02,
        zero_sign_margin_temperature=0.02,
    )


def test_zero_sign_margin_matches_exact_zero_two_endpoint_formula():
    raw = torch.tensor([-1.0, 0.0, 1e-8, 0.0])
    y_reg = torch.tensor([-0.7, -0.03, -0.2, 0.08], requires_grad=True)
    y_cls_expected = torch.tensor([-0.6, 0.01, 0.3, -0.04], requires_grad=True)
    args = _zero_sign_margin_args()
    loss = _compute_zero_sign_margin_loss(
        {"y_reg": y_reg, "extras": {"y_cls_expected": y_cls_expected}},
        raw,
        args,
    )

    mask = raw == 0.0
    reg_barrier = args.zero_sign_margin_temperature * torch.nn.functional.softplus(
        (args.zero_sign_margin - y_reg[mask]) / args.zero_sign_margin_temperature
    )
    cls_barrier = args.zero_sign_margin_temperature * torch.nn.functional.softplus(
        (args.zero_sign_margin - y_cls_expected[mask]) / args.zero_sign_margin_temperature
    )
    expected = 0.5 * (reg_barrier.mean() + cls_barrier.mean())
    assert torch.allclose(loss, expected, atol=1e-8, rtol=1e-7)

    loss.backward()
    assert torch.equal(y_reg.grad[[0, 2]], torch.zeros(2))
    assert torch.equal(y_cls_expected.grad[[0, 2]], torch.zeros(2))
    assert torch.all(y_reg.grad[[1, 3]] < 0.0)
    assert torch.all(y_cls_expected.grad[[1, 3]] < 0.0)


def test_zero_sign_margin_no_zero_or_missing_cls_endpoint_is_safe():
    args = _zero_sign_margin_args()
    y_reg = torch.tensor([-4.0, 0.0, 4.0], requires_grad=True)
    y_cls_expected = torch.tensor([4.0, 0.0, -4.0], requires_grad=True)
    no_zero_loss = _compute_zero_sign_margin_loss(
        {"y_reg": y_reg, "extras": {"y_cls_expected": y_cls_expected}},
        torch.tensor([-1.0, 1e-8, 1.0]),
        args,
    )
    assert torch.isfinite(no_zero_loss) and no_zero_loss == 0.0
    no_zero_loss.backward()
    assert torch.equal(y_reg.grad, torch.zeros_like(y_reg))
    assert torch.equal(y_cls_expected.grad, torch.zeros_like(y_cls_expected))

    missing_y_reg = torch.tensor([-1.0, 1.0], requires_grad=True)
    missing_loss = _compute_zero_sign_margin_loss(
        {"y_reg": missing_y_reg, "extras": {}},
        torch.zeros(2),
        args,
    )
    assert torch.isfinite(missing_loss) and missing_loss == 0.0
    missing_loss.backward()
    assert torch.equal(missing_y_reg.grad, torch.zeros_like(missing_y_reg))


def test_zero_sign_margin_is_finite_at_extreme_scores_and_reuses_warmup():
    args = _zero_sign_margin_args()
    y_reg = torch.tensor([-4.0, 0.0, 4.0], requires_grad=True)
    y_cls_expected = torch.tensor([4.0, 0.0, -4.0], requires_grad=True)
    loss = _compute_zero_sign_margin_loss(
        {"y_reg": y_reg, "extras": {"y_cls_expected": y_cls_expected}},
        torch.zeros(3),
        args,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(y_reg.grad).all()
    assert torch.isfinite(y_cls_expected.grad).all()

    warmup_args = SimpleNamespace(epochs=10, sign_struct_warmup_ratio=0.3)
    effective_weights = [_sign_struct_scale(warmup_args, epoch) * 0.02 for epoch in range(10)]
    assert effective_weights[:3] == [0.0, 0.0, 0.0]
    assert effective_weights[3:] == [0.02] * 7


def _reg_cls_mag_consistency_args(**overrides):
    values = {
        "reg_cls_mag_consistency_mode": REG_CLS_MAG_CONSISTENCY_MODE,
        "reg_cls_mag_consistency_boundary_margin": 0.1,
        "reg_cls_mag_consistency_smooth_l1_beta": 0.1,
        "final_pred_sign_beta": 0.2,
        "clamp_regression_eval": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reg_cls_mag_consistency_mask_is_conservative():
    assert REG_CLS_MAG_CONSISTENCY_BOUNDARIES == (-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5)
    # First two samples qualify.  The remaining samples respectively violate
    # sign agreement, Acc7-bin agreement, Acc7-boundary safety, sign-boundary
    # safety, and shrink-only direction.
    y_reg = torch.tensor([0.40, -0.40, 0.40, 1.10, 0.60, 0.20, 0.25], requires_grad=True)
    y_cls_expected = torch.tensor(
        [0.25, -0.25, -0.25, 0.40, 0.40, 0.05, 0.40],
        requires_grad=True,
    )
    sign_logits = torch.zeros(7, requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(),
    )

    assert torch.isfinite(loss)
    assert loss > 0
    assert stats["candidate_count"] == 7
    assert stats["selected_count"] == 2
    assert abs(stats["coverage"] - 2.0 / 7.0) < 1e-8
    assert abs(stats["mean_abs_gap"] - 0.12) < 1e-6


def test_reg_cls_mag_consistency_detaches_cls_sign_and_mask():
    y_reg = torch.tensor([0.40, -0.40], requires_grad=True)
    y_cls_expected = torch.tensor([0.25, -0.25], requires_grad=True)
    sign_logits = torch.tensor([0.2, -0.2], requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(),
    )
    loss.backward()

    assert stats["selected_count"] == 2
    assert y_reg.grad is not None
    assert y_reg.grad[0] > 0  # gradient descent shrinks a positive magnitude
    assert y_reg.grad[1] < 0  # gradient descent shrinks a negative magnitude
    assert y_cls_expected.grad is None
    assert sign_logits.grad is None


def test_reg_cls_mag_consistency_sign_balanced_weights_branches_equally():
    y_reg = torch.tensor([0.25, 0.30, 0.35, -0.39], requires_grad=True)
    y_cls_expected = torch.tensor([0.11, 0.11, 0.11, -0.11], requires_grad=True)
    sign_logits = torch.zeros(4, requires_grad=True)
    model_out = {
        "y_reg": y_reg,
        "sign_logits": sign_logits,
        "extras": {"y_cls_expected": y_cls_expected},
    }
    balanced_loss, stats = _compute_reg_cls_mag_consistency_loss(
        model_out,
        _reg_cls_mag_consistency_args(
            reg_cls_mag_consistency_mode=(
                REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
            ),
            final_pred_sign_beta=0.0,
        ),
    )
    per_sample = torch.nn.functional.smooth_l1_loss(
        y_reg.abs(),
        y_cls_expected.abs(),
        reduction="none",
        beta=0.1,
    )
    expected = 0.5 * (per_sample[:3].mean() + per_sample[3:].mean())
    assert torch.allclose(balanced_loss, expected)
    assert stats["selected_count"] == 4
    assert stats["positive_selected_count"] == 3
    assert stats["negative_selected_count"] == 1
    assert stats["sign_balanced_reduction_applied"] == 1
    assert stats["single_branch_fallback"] == 0

    balanced_loss.backward()
    assert torch.allclose(y_reg.grad[:3].abs().sum(), torch.tensor(0.5))
    assert torch.allclose(y_reg.grad[3:].abs().sum(), torch.tensor(0.5))
    assert y_cls_expected.grad is None
    assert sign_logits.grad is None


def test_reg_cls_mag_consistency_sign_balanced_single_branch_falls_back_to_mean():
    y_reg = torch.tensor([0.25, 0.30, 0.35], requires_grad=True)
    y_cls_expected = torch.tensor([0.11, 0.11, 0.11], requires_grad=True)
    sign_logits = torch.zeros(3, requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(
            reg_cls_mag_consistency_mode=(
                REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
            ),
            final_pred_sign_beta=0.0,
        ),
    )
    expected = torch.nn.functional.smooth_l1_loss(
        y_reg.abs(),
        y_cls_expected.abs(),
        reduction="mean",
        beta=0.1,
    )
    assert torch.allclose(loss, expected)
    assert stats["positive_selected_count"] == 3
    assert stats["negative_selected_count"] == 0
    assert stats["sign_balanced_reduction_applied"] == 0
    assert stats["single_branch_fallback"] == 1


def test_reg_cls_mag_consistency_sign_balanced_negative_only_falls_back_to_mean():
    y_reg = torch.tensor([-0.25, -0.30, -0.35], requires_grad=True)
    y_cls_expected = torch.tensor([-0.11, -0.11, -0.11], requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": torch.zeros(3, requires_grad=True),
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(
            reg_cls_mag_consistency_mode=(
                REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
            ),
            final_pred_sign_beta=0.0,
        ),
    )
    expected = torch.nn.functional.smooth_l1_loss(
        y_reg.abs(),
        y_cls_expected.abs(),
        reduction="mean",
        beta=0.1,
    )
    assert torch.allclose(loss, expected)
    assert stats["positive_selected_count"] == 0
    assert stats["negative_selected_count"] == 3
    assert stats["sign_balanced_reduction_applied"] == 0
    assert stats["single_branch_fallback"] == 1


def test_reg_cls_mag_consistency_sign_balanced_production_beta_detaches_teachers():
    y_reg = torch.tensor([0.40, -0.40], requires_grad=True)
    y_cls_expected = torch.tensor([0.25, -0.25], requires_grad=True)
    sign_logits = torch.tensor([0.2, -0.2], requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(
            reg_cls_mag_consistency_mode=(
                REG_CLS_MAG_CONSISTENCY_SIGN_BALANCED_MODE
            )
        ),
    )
    loss.backward()
    assert stats["positive_selected_count"] == 1
    assert stats["negative_selected_count"] == 1
    assert stats["sign_balanced_reduction_applied"] == 1
    assert y_reg.grad[0] > 0
    assert y_reg.grad[1] < 0
    assert y_cls_expected.grad is None
    assert sign_logits.grad is None


def test_reg_cls_mag_consistency_legacy_reduction_remains_unbalanced_mean():
    y_reg = torch.tensor([0.25, 0.30, 0.35, -0.39], requires_grad=True)
    y_cls_expected = torch.tensor([0.11, 0.11, 0.11, -0.11], requires_grad=True)
    loss, _ = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": torch.zeros(4, requires_grad=True),
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(final_pred_sign_beta=0.0),
    )
    expected = torch.nn.functional.smooth_l1_loss(
        y_reg.abs(),
        y_cls_expected.abs(),
        reduction="mean",
        beta=0.1,
    )
    assert torch.allclose(loss, expected)


def test_reg_cls_mag_consistency_zero_mask_is_safe():
    y_reg = torch.tensor([0.4], requires_grad=True)
    y_cls_expected = torch.tensor([-0.25], requires_grad=True)
    sign_logits = torch.zeros(1, requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(),
    )
    loss.backward()

    assert loss.item() == 0.0
    assert stats["selected_count"] == 0
    assert torch.equal(y_reg.grad, torch.zeros_like(y_reg))
    assert y_cls_expected.grad is None
    assert sign_logits.grad is None


def test_reg_cls_mag_consistency_straight_through_clamp_has_inward_gradient():
    y_reg = torch.tensor([4.0, -4.0], requires_grad=True)
    y_cls_expected = torch.tensor([2.7, -2.7], requires_grad=True)
    sign_logits = torch.zeros(2, requires_grad=True)
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        {
            "y_reg": y_reg,
            "sign_logits": sign_logits,
            "extras": {"y_cls_expected": y_cls_expected},
        },
        _reg_cls_mag_consistency_args(reg_cls_mag_consistency_boundary_margin=0.05),
    )
    loss.backward()

    assert stats["selected_count"] == 2
    assert loss > 0
    assert y_reg.grad is not None
    assert y_reg.grad[0] > 0
    assert y_reg.grad[1] < 0
    assert y_cls_expected.grad is None
    assert sign_logits.grad is None


def test_reg_cls_mag_consistency_endpoints_match_production_decoder():
    torch.manual_seed(7)
    y_reg = torch.tensor([-4.0, -1.2, 0.2, 1.4, 4.0], requires_grad=True)
    cls7_logits = torch.randn(5, 7, requires_grad=True)
    sign_logits = torch.randn(5, requires_grad=True)
    centers = get_cls7_centers(device=cls7_logits.device, dtype=cls7_logits.dtype)
    y_cls_expected = cls7_expected_value(cls7_logits, centers=centers)
    model_out = {
        "y_reg": y_reg,
        "cls7_logits": cls7_logits,
        "sign_logits": sign_logits,
        # Deliberately wrong: endpoint reconstruction must not consume y_final.
        "extras": {"y_cls_expected": y_cls_expected, "y_final": torch.full((5,), 123.0)},
    }
    for sign_beta in (0.0, 0.2, 1.0):
        args = _reg_cls_mag_consistency_args(final_pred_sign_beta=sign_beta)
        decoded_reg, decoded_cls = _reg_cls_mag_consistency_decoded_endpoints(model_out, args)
        production_reg = compute_final_prediction(
            y_reg.clamp(-3.0, 3.0),
            cls7_logits,
            eta=0.0,
            centers=centers,
            sign_logits=sign_logits,
            sign_beta=sign_beta,
        )
        production_cls = compute_final_prediction(
            y_reg.clamp(-3.0, 3.0),
            cls7_logits,
            eta=1.0,
            centers=centers,
            sign_logits=sign_logits,
            sign_beta=sign_beta,
        )
        assert torch.allclose(decoded_reg, production_reg, atol=1e-7, rtol=1e-6)
        assert torch.allclose(decoded_cls, production_cls, atol=1e-7, rtol=1e-6)


def test_reg_cls_mag_consistency_uses_half_away_acc7_boundaries():
    eps = 1e-6
    boundaries_and_zero = torch.tensor([-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5])
    exact_expected_with_zero = torch.tensor([0, 1, 2, 3, 4, 5, 6])
    boundaries = boundaries_and_zero[boundaries_and_zero != 0.0]
    exact_expected = exact_expected_with_zero[boundaries_and_zero != 0.0]
    toward_zero_expected = torch.tensor([1, 2, 3, 3, 4, 5])
    away_from_zero_expected = exact_expected

    assert torch.equal(continuous_to_cls7_hard(boundaries_and_zero), exact_expected_with_zero)
    toward_zero = boundaries + torch.tensor([eps, eps, eps, -eps, -eps, -eps])
    away_from_zero = boundaries + torch.tensor([-eps, -eps, -eps, eps, eps, eps])
    assert torch.equal(continuous_to_cls7_hard(toward_zero), toward_zero_expected)
    assert torch.equal(continuous_to_cls7_hard(away_from_zero), away_from_zero_expected)
    around_half = torch.tensor([-0.5 - eps, -0.5, -0.5 + eps, 0.5 - eps, 0.5, 0.5 + eps])
    assert torch.equal(continuous_to_cls7_hard(around_half), torch.tensor([2, 2, 3, 3, 4, 4]))


def test_reg_cls_mag_consistency_toy_shared_path_has_only_direct_reg_teacher_gradients():
    class ToyHeads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = torch.nn.Linear(1, 1, bias=False)
            self.reg = torch.nn.Linear(1, 1, bias=False)
            self.cls = torch.nn.Linear(1, 1, bias=False)
            self.sign = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                self.shared.weight.fill_(1.0)
                self.reg.weight.fill_(0.40)
                self.cls.weight.fill_(0.25)
                self.sign.weight.fill_(0.20)

        def forward(self, x):
            shared = self.shared(x)
            y_reg = self.reg(shared).squeeze(-1)
            y_cls_expected = self.cls(shared).squeeze(-1)
            sign_logits = self.sign(shared).squeeze(-1)
            return {
                "y_reg": y_reg,
                "sign_logits": sign_logits,
                "extras": {"y_cls_expected": y_cls_expected},
            }

    model = ToyHeads()
    x = torch.ones(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    cls_weight_before = model.cls.weight.detach().clone()
    sign_weight_before = model.sign.weight.detach().clone()
    cls_output_before = model(x)["extras"]["y_cls_expected"].detach().clone()
    loss, stats = _compute_reg_cls_mag_consistency_loss(
        model(x),
        _reg_cls_mag_consistency_args(),
    )
    loss.backward()

    assert stats["selected_count"] == 1
    assert model.shared.weight.grad is not None and model.shared.weight.grad.abs().sum() > 0
    assert model.reg.weight.grad is not None and model.reg.weight.grad.abs().sum() > 0
    assert model.cls.weight.grad is None
    assert model.sign.weight.grad is None

    optimizer.step()
    cls_output_after = model(x)["extras"]["y_cls_expected"].detach()
    assert torch.equal(model.cls.weight.detach(), cls_weight_before)
    assert torch.equal(model.sign.weight.detach(), sign_weight_before)
    assert not torch.allclose(cls_output_after, cls_output_before)


def test_reg_cls_mag_consistency_reuses_sign_struct_warmup():
    args = SimpleNamespace(epochs=10, sign_struct_warmup_ratio=0.3)
    effective_weights = [_sign_struct_scale(args, epoch) * 0.02 for epoch in range(10)]
    assert effective_weights[:3] == [0.0, 0.0, 0.0]
    assert effective_weights[3:] == [0.02] * 7


def test_slim_tensorboard_forwards_reg_cls_mag_consistency_diagnostics():
    class FakeWriter:
        def __init__(self):
            self.scalars = []
            self.texts = []

        def add_scalar(self, tag, value, *args, **kwargs):
            self.scalars.append((tag, value, args, kwargs))

        def add_text(self, tag, value, *args, **kwargs):
            self.texts.append((tag, value, args, kwargs))

    fake = FakeWriter()
    writer = SlimTensorBoardWriter(fake)
    tags = [
        "train/zero_sign_margin_loss",
        "train/weighted_zero_sign_margin_loss",
        "epoch/val_zero_sign_margin_loss",
        "epoch/val_weighted_zero_sign_margin_loss",
        "train/reg_cls_mag_consistency_loss",
        "train/weighted_reg_cls_mag_consistency_loss",
        "train/reg_cls_mag_consistency_selected_count",
        "train/reg_cls_mag_consistency_coverage",
        "train/reg_cls_mag_consistency_mean_abs_gap",
        "epoch/val_reg_cls_mag_consistency_loss",
        "epoch/val_weighted_reg_cls_mag_consistency_loss",
        "epoch/val_reg_cls_mag_consistency_selected_count",
        "epoch/val_reg_cls_mag_consistency_coverage",
        "epoch/val_reg_cls_mag_consistency_mean_abs_gap",
    ]
    for index, tag in enumerate(tags):
        writer.add_scalar(tag, float(index), 7)
    config_tags = [
        "config/zero_sign_margin_weight",
        "config/zero_sign_margin",
        "config/zero_sign_margin_temperature",
        "config/reg_cls_mag_consistency_weight",
        "config/reg_cls_mag_consistency_boundary_margin",
        "config/reg_cls_mag_consistency_smooth_l1_beta",
        "config/reg_cls_mag_consistency_mode",
    ]
    for tag in config_tags:
        writer.add_text(tag, "configured", 0)

    forwarded = [row[0] for row in fake.scalars]
    assert len(forwarded) == len(tags)
    assert "loss/train_step_zero_sign_margin" in forwarded
    assert "loss/train_step_weighted_zero_sign_margin" in forwarded
    assert "loss/val_zero_sign_margin" in forwarded
    assert "loss/val_weighted_zero_sign_margin" in forwarded
    assert "loss/train_step_reg_cls_mag_consistency" in forwarded
    assert "loss/train_step_weighted_reg_cls_mag_consistency" in forwarded
    assert "diagnostics/train_step_reg_cls_mag_consistency_selected_count" in forwarded
    assert "diagnostics/train_step_reg_cls_mag_consistency_coverage" in forwarded
    assert "diagnostics/train_step_reg_cls_mag_consistency_mean_abs_gap" in forwarded
    assert "loss/val_reg_cls_mag_consistency" in forwarded
    assert "loss/val_weighted_reg_cls_mag_consistency" in forwarded
    assert "diagnostics/val_reg_cls_mag_consistency_selected_count" in forwarded
    assert "diagnostics/val_reg_cls_mag_consistency_coverage" in forwarded
    assert "diagnostics/val_reg_cls_mag_consistency_mean_abs_gap" in forwarded
    assert [row[0] for row in fake.texts] == config_tags


def main():
    test_hier_sign_mag_reconstructs_valid_cls7_distribution()
    test_cumulative_head_reconstructs_valid_cls7_distribution()
    test_structured_losses_are_finite_for_edge_batches()
    test_sign_marginal_losses_are_finite_and_nonzero_only()
    test_signed_neutral_band_penalizes_wrong_and_insufficient_margin()
    test_signed_neutral_band_penalizes_leaving_the_acc7_neutral_bin()
    test_signed_neutral_band_excludes_exact_zero_labels()
    test_signed_neutral_band_backpropagates_only_through_tiny_nonzero_labels()
    test_zero_sign_margin_matches_exact_zero_two_endpoint_formula()
    test_zero_sign_margin_no_zero_or_missing_cls_endpoint_is_safe()
    test_zero_sign_margin_is_finite_at_extreme_scores_and_reuses_warmup()
    test_reg_cls_mag_consistency_mask_is_conservative()
    test_reg_cls_mag_consistency_detaches_cls_sign_and_mask()
    test_reg_cls_mag_consistency_sign_balanced_weights_branches_equally()
    test_reg_cls_mag_consistency_sign_balanced_single_branch_falls_back_to_mean()
    test_reg_cls_mag_consistency_sign_balanced_negative_only_falls_back_to_mean()
    test_reg_cls_mag_consistency_sign_balanced_production_beta_detaches_teachers()
    test_reg_cls_mag_consistency_legacy_reduction_remains_unbalanced_mean()
    test_reg_cls_mag_consistency_zero_mask_is_safe()
    test_reg_cls_mag_consistency_straight_through_clamp_has_inward_gradient()
    test_reg_cls_mag_consistency_endpoints_match_production_decoder()
    test_reg_cls_mag_consistency_uses_half_away_acc7_boundaries()
    test_reg_cls_mag_consistency_toy_shared_path_has_only_direct_reg_teacher_gradients()
    test_reg_cls_mag_consistency_reuses_sign_struct_warmup()
    test_slim_tensorboard_forwards_reg_cls_mag_consistency_diagnostics()
    print("sign structured head tests passed")


if __name__ == "__main__":
    main()
