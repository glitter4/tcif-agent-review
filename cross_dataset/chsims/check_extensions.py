"""CPU checks for loss isolation, scheduling and guarded checkpoint semantics."""
import ast
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import torch.optim as optim

WORK = Path(__file__).resolve().parent
CODE = WORK / "source/server-code"
sys.path.insert(0, str(CODE))
from head_utils import get_cls7_centers, compute_final_prediction, continuous_to_cls7_hard
from chsims_followup_support import guarded_metrics, update_guarded, passes_guard
import metric_audit


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(found) == len(names), names
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


ns = extract(CODE / "train_emotion.py", {
    "compute_intensity_loss", "compute_regression_loss", "compute_center_regression_loss",
    "_compute_tcif_context_aux_loss", "_signed_centers", "_build_lr_scheduler"},
    {"torch": torch, "F": F, "optim": optim, "math": math,
     "get_cls7_centers": get_cls7_centers, "continuous_to_cls7_hard": continuous_to_cls7_hard})
args = SimpleNamespace(reg_loss_type="smooth_l1", center_reg_loss_type="l1", signed_class_count=5,
                       label_min=-1., label_max=1., tcif_context_aux_cls7_weight=.5)
y = torch.tensor([-.2, .4])
p = torch.tensor([0., .1], requires_grad=True)
assert torch.allclose(ns["compute_center_regression_loss"](p, y, args), F.l1_loss(p, y))
ns["compute_center_regression_loss"](p, y, args).backward()
assert torch.allclose(p.grad, torch.tensor([.5, -.5]))
payload = {"extras": {"tcif": {"has_context": torch.tensor([True, True]), "context_y_reg": p,
                              "context_cls7_logits": torch.zeros(2, 5)}}}
context = ns["_compute_tcif_context_aux_loss"](payload, y, args)
assert torch.allclose(context[1], F.smooth_l1_loss(p, y))
args.center_reg_loss_type = "inherit"
assert torch.allclose(ns["compute_center_regression_loss"](p, y, args), F.smooth_l1_loss(p, y))

rates = [2.32e-5, 1.16e-5, 1e-4, 2.32e-5, 1.16e-6]
opt = optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr} for lr in rates])
cfg = SimpleNamespace(scheduler="plateau", plateau_factor=.5, plateau_patience=5,
                      plateau_relative_min_factor=.03, min_lr=1e-6)
scheduler = ns["_build_lr_scheduler"](opt, cfg)
for value in [1., .9, .8, .7]:
    scheduler.step(value)
assert [g["lr"] for g in opt.param_groups] == rates
for _ in range(6):
    scheduler.step(.8)
assert all(abs(g["lr"] - r*.5) < 1e-14 for g, r in zip(opt.param_groups, rates))
for _ in range(90):
    scheduler.step(.8)
assert all(abs(g["lr"] - r*.03) < 1e-14 for g, r in zip(opt.param_groups, rates))
legacy_opt = optim.AdamW([{"params": [torch.nn.Parameter(torch.ones(1))], "lr": 2.32e-5}])
cfg.plateau_relative_min_factor = -1
assert ns["_build_lr_scheduler"](legacy_opt, cfg).min_lrs == [1e-6]
training_source = (CODE / "train_emotion.py").read_text(encoding="utf-8")
assert 'scheduler.step(val_mae if args.plateau_monitor == "val_mae" else val_loss)' in training_source

centers = get_cls7_centers(num_classes=5, label_min=-1, label_max=1)
truth = torch.tensor([-1., -.8, -.6, -.4, -.2, 0., .2, .4, .6, .8, 1.])
reg = torch.tensor([-.7, .2, -.5, -.1, 0., -.2, .1, -.5, .4, .9, 1.5])
classes = torch.tensor([0, 0, 1, 2, 2, 2, 2, 0, 3, 4, 4])
logits = F.one_hot(classes, 5).float() * 5
fused = compute_final_prediction(reg.clamp(-1, 1), logits, eta=.8, centers=centers, classification_readout="argmax")
metrics = guarded_metrics(reg.tolist(), classes.tolist(), truth.tolist(), centers)
audit = metric_audit.binary_metrics(truth.tolist(), fused.tolist(), 0.)
for key in ["Acc2", "Acc2non0", "F1", "F1non0"]:
    assert abs(metrics[key] - audit[key]) < 1e-9, (key, metrics[key], audit[key])
assert abs(metrics["MAE"] - float((fused.double()-truth.double()).abs().mean())) < 1e-8

with tempfile.TemporaryDirectory(prefix="chsims-guard-check-") as temporary:
    folder = Path(temporary) / "checkpoints"
    folder.mkdir()
    cfg = SimpleNamespace(enable_guarded_checkpoint=True, dataset="chsims",
                          checkpoint_selection_split="test", save_dir=str(folder))
    records = {}
    model = torch.nn.Linear(1, 1)
    def sidecar(path, args, checkpoint_epoch):
        Path(path).with_suffix(".json").write_text(json.dumps({"checkpoint_epoch": checkpoint_epoch}))
    bad = {"Acc5": 48., "MAE": .39, "F1non0": 82., "Acc2": 77., "Acc2non0": 84.}
    update_guarded(model, cfg, 1, bad, bad, records, sidecar)
    assert records["guarded_acc2"]["checkpoint"] is None
    assert not list(folder.glob("*.pth"))
    good = {**bad, "Acc5": 49.5}
    update_guarded(model, cfg, 2, good, good, records, sidecar)
    update_guarded(model, cfg, 3, good, good, records, sidecar)
    assert records["guarded_acc2"]["best"]["epoch"] == 2
    better = {**good, "MAE": .38}
    update_guarded(model, cfg, 4, better, better, records, sidecar)
    assert records["guarded_acc2"]["best"]["epoch"] == 4
    assert records["guarded_acc2"]["eligible_epoch_count"] == 3
    assert not passes_guard({**good, "Acc2": 76.})
print("EXTENSION_CHECKS_PASSED: center/context loss, relative floors, metrics, null/tie guarded checkpoint", flush=True)

# Coefficient changes center values and gradients only, leaving context unchanged.
for coefficient in [.75, 1., 1.25]:
    args.center_reg_loss_type = "l1"
    args.center_reg_loss_weight = coefficient
    sample = torch.tensor([0., .1], requires_grad=True)
    loss = ns["compute_center_regression_loss"](sample, y, args)
    assert torch.allclose(loss, coefficient * F.l1_loss(sample, y))
    loss.backward()
    assert torch.allclose(sample.grad, coefficient * torch.tensor([.5, -.5]))
    assert torch.allclose(ns["_compute_tcif_context_aux_loss"](payload, y, args)[1], F.smooth_l1_loss(p, y))
print("CENTER_SCALE_CHECKS_PASSED")

# Validation-only selection must ignore an opposing test trend and preserve tie order.
from chsims_followup_support import update_dev_aligned
with tempfile.TemporaryDirectory(prefix="chsims-dev-selection-") as temporary:
    args = SimpleNamespace(save_dir=temporary)
    records = {}
    model = torch.nn.Linear(1, 1)
    def sidecar(path, args, checkpoint_epoch):
        Path(path).with_suffix('.json').write_text(json.dumps({'checkpoint_epoch':checkpoint_epoch}))
    update_dev_aligned(model,args,1,{'Acc5':50.,'MAE':.4},{'Acc5':90.,'MAE':.1},records,sidecar)
    update_dev_aligned(model,args,2,{'Acc5':51.,'MAE':.41},{'Acc5':20.,'MAE':.9},records,sidecar)
    assert records['dev_argmax_acc5']['epoch']==2
    assert records['dev_argmax_mae']['epoch']==1
    update_dev_aligned(model,args,3,{'Acc5':51.,'MAE':.41},{'Acc5':99.,'MAE':.01},records,sidecar)
    assert records['dev_argmax_acc5']['epoch']==2
    update_dev_aligned(model,args,4,{'Acc5':49.,'MAE':.39},{'Acc5':10.,'MAE':1.},records,sidecar)
    assert records['dev_argmax_mae']['epoch']==4
    for path in Path(temporary).glob('*.pth'):
        assert json.loads(path.with_suffix('.json').read_text())['checkpoint_selection_split']=='val'

gate_ns = extract(CODE/'train_emotion.py', {'_compute_tcif_transition_gate_loss'}, {'torch':torch,'F':F})
center=torch.tensor([.8,.8])
batch={'tcif_context_raw_valence':torch.tensor([[.4],[-.4]]), 'tcif_context_valid_mask':torch.ones(2,1,dtype=torch.bool)}
logits=torch.zeros(2,requires_grad=True)
payload={'extras':{'tcif':{'regression_continuation_gate_logits':logits,'ordinal_continuation_gate_logits':logits,
                         'regression_continuation_gate':logits.sigmoid(),'ordinal_continuation_gate':logits.sigmoid()}}}
means=[]
for tau in [.75,.4,.25]:
    cfg=SimpleNamespace(tcif_transition_gate_tau=tau,tcif_transition_gate_conflict_target=0.)
    loss,stats=gate_ns['_compute_tcif_transition_gate_loss'](payload,batch,center,cfg)
    assert stats['sign_conflict_count']==1
    assert abs(stats['target_mean']-math.exp(-.4/tau)/2)<1e-6
    means.append(stats['target_mean'])
assert means[0]>means[1]>means[2]
print('DEV_SELECTION_AND_GATE_SCALE_CHECKS_PASSED',flush=True)
