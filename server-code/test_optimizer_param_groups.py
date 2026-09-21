from types import SimpleNamespace

import torch

from train_emotion import _build_optimizer_param_groups


class _ToyOptimizerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(3, 2)
        self.vit = torch.nn.Linear(3, 2)
        self.soft_moe = torch.nn.Module()
        self.soft_moe.phi = torch.nn.Parameter(torch.ones(2, 2))
        self.soft_moe.scale = torch.nn.Parameter(torch.ones(2))
        self.tcif_regression = torch.nn.Module()
        self.tcif_regression.transition_gate = torch.nn.Linear(3, 1)
        self.tcif_ordinal = torch.nn.Module()
        self.tcif_ordinal.transition_gate = torch.nn.Linear(3, 1)


def test_tcif_transition_gate_has_dedicated_lr_without_changing_parameter_set():
    model = _ToyOptimizerModel()
    args = SimpleNamespace(
        lr=6e-5,
        router_lr=4e-4,
        tcif_transition_gate_lr=1.2e-4,
        unfreeze_bert_last_n_layers=0,
        bert_last_layer_lr_ratio=0.5,
        backbone_lr_ratio=0.1,
    )
    groups = _build_optimizer_param_groups(model, None, args)
    by_name = {group["name"]: group for group in groups}

    assert by_name["head"]["lr"] == args.lr
    assert by_name["router"]["lr"] == args.router_lr
    assert by_name["tcif_transition_gate"]["lr"] == args.tcif_transition_gate_lr
    assert by_name["backbone"]["lr"] == args.lr * args.backbone_lr_ratio

    named = dict(model.named_parameters())
    gate_ids = {
        id(parameter)
        for name, parameter in named.items()
        if ".transition_gate." in name
    }
    dedicated_ids = {
        id(parameter) for parameter in by_name["tcif_transition_gate"]["params"]
    }
    assert dedicated_ids == gate_ids
    assert not gate_ids.intersection(
        id(parameter) for parameter in by_name["router"]["params"]
    )

    grouped_ids = [
        id(parameter) for group in groups for parameter in group["params"]
    ]
    expected_ids = {id(parameter) for parameter in model.parameters()}
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == expected_ids


def test_base_lr_transition_gate_preserves_historical_numeric_lr():
    model = _ToyOptimizerModel()
    args = SimpleNamespace(
        lr=6e-5,
        router_lr=4e-4,
        tcif_transition_gate_lr=6e-5,
        unfreeze_bert_last_n_layers=0,
        bert_last_layer_lr_ratio=0.5,
        backbone_lr_ratio=0.1,
    )
    groups = _build_optimizer_param_groups(model, None, args)
    by_name = {group["name"]: group for group in groups}
    assert by_name["tcif_transition_gate"]["lr"] == by_name["head"]["lr"]
