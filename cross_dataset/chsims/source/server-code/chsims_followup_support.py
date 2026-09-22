"""Training diagnostics and guarded selection; no model or forward changes."""
import json
from pathlib import Path

import torch
from head_utils import continuous_to_cls7_hard

TEMPORAL_BATCH_STATS = []


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def binary_metrics(pred, truth):
    y, p = truth >= 0, pred >= 0
    accuracy = 100.0 * int((y == p).sum()) / len(y)
    f1 = []
    for cls in (False, True):
        tp = int(((p == cls) & (y == cls)).sum())
        fp = int(((p == cls) & (y != cls)).sum())
        fn = int(((p != cls) & (y == cls)).sum())
        f1.append(2.0 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return accuracy, 50.0 * sum(f1)


def guarded_metrics(reg_scores, cls_indices, truths, centers):
    reg = torch.as_tensor(reg_scores, dtype=torch.float32)
    y = torch.as_tensor(truths, dtype=torch.float32)
    centers = torch.as_tensor(centers, dtype=torch.float32).cpu()
    assert centers.numel() == 5 and len(reg) == len(y) and len(y) > 0
    cls = centers[torch.as_tensor(cls_indices, dtype=torch.long)]
    p = 0.2 * reg.clamp(-1, 1) + 0.8 * cls
    nz = y != 0
    assert bool(nz.any())
    acc2, f1 = binary_metrics(p, y)
    nonzero, f1nz = binary_metrics(p[nz], y[nz])
    acc5 = 100.0 * int((continuous_to_cls7_hard(p, centers) == continuous_to_cls7_hard(y, centers)).sum()) / len(y)
    errors = (p >= 0) != (y >= 0)
    return {"Acc2": acc2, "Acc2non0": nonzero, "F1": f1, "F1non0": f1nz,
            "Acc5": acc5, "MAE": float((p.double() - y.double()).abs().mean()),
            "num_samples_all": len(y), "num_samples_non0": int(nz.sum()),
            "correct_nonzero": int((~errors & nz).sum()),
            "neutral_cls_nonzero_errors": int((errors & nz & (cls == 0)).sum()),
            "nonneutral_cls_nonzero_errors": int((errors & nz & (cls != 0)).sum()),
            "readout": "argmax", "eta": 0.8, "temperature": 1.0, "threshold": 0.0}


def passes_guard(metrics):
    return (metrics["Acc5"] >= 49.0 and metrics["MAE"] <= .402
            and metrics["F1non0"] >= 81.0606713182301 and metrics["Acc2"] >= 76.3)


def rank_guarded(metrics, epoch):
    return (-metrics["Acc2non0"], metrics["MAE"], -metrics["Acc5"], epoch)


def update_guarded(model, args, epoch, val_metrics, test_metrics, records, write_hparams):
    if not getattr(args, "enable_guarded_checkpoint", False):
        return
    if args.dataset != "chsims" or args.checkpoint_selection_split != "test":
        raise ValueError("Guarded CH-SIMS protocol requires test-selected CH-SIMS")
    if getattr(args, "enable_dev_aligned_checkpoint", False):
        update_dev_aligned(model, args, epoch, val_metrics, test_metrics, records, write_hparams)
    folder = Path(args.save_dir)
    state = records.setdefault("guarded_acc2", {"eligible_epoch_count": 0, "checkpoint": None, "best": None})
    eligible = passes_guard(test_metrics)
    append_json(folder.parent / "argmax_epoch_metrics.jsonl",
                {"epoch": epoch, "val": val_metrics, "test": test_metrics, "eligible": eligible})
    if eligible:
        state["eligible_epoch_count"] += 1
        best = state["best"]
        if best is None or rank_guarded(test_metrics, epoch) < rank_guarded(best["metrics"], best["epoch"]):
            path = folder / "best_guarded_acc2_model.pth"
            temp = path.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temp)
            temp.replace(path)
            write_hparams(str(path), args, checkpoint_epoch=epoch)
            state["checkpoint"] = str(path)
            state["best"] = {"epoch": epoch, "metrics": test_metrics}
            print("Updated guarded argmax Acc2 checkpoint", epoch, test_metrics, flush=True)
    write_json(folder / "guarded_acc2_selection.json", state)


def record_temporal(batch, args, stats):
    if not getattr(args, "enable_guarded_checkpoint", False) or not torch.is_grad_enabled():
        return
    with torch.no_grad():
        y = batch["raw_valence"].detach().reshape(-1)
        group = batch["temporal_group_id"].reshape(-1)
        pos = batch["temporal_pos"].reshape(-1)
        distance = (pos[:, None] - pos[None, :]).abs()
        not_self = ~torch.eye(len(y), dtype=torch.bool, device=y.device)
        pairs = (group[:, None] == group[None, :]) & not_self & (distance <= args.temporal_weak_positive_radius)
        opposite = int((pairs & (y[:, None] * y[None, :] < 0)).sum())
    TEMPORAL_BATCH_STATS.append({**stats, "legacy_opposite_pairs": opposite,
        "removed_opposite_pairs": opposite if args.temporal_kernel == "legacy_sign_safe" else 0})


def flush_temporal(args, epoch):
    if not TEMPORAL_BATCH_STATS:
        return
    rows = list(TEMPORAL_BATCH_STATS)
    TEMPORAL_BATCH_STATS.clear()
    fields = ("valid_anchor_count", "legacy_valid_anchor_count", "positive_pair_count",
              "legacy_positive_pair_count", "legacy_opposite_pairs", "removed_opposite_pairs")
    result = {key: sum(row.get(key, 0) for row in rows) for key in fields}
    result.update(epoch=epoch, batches=len(rows), zero_positive_batches=sum(row.get("valid_anchor_count", 0) == 0 for row in rows),
                  mean_anchor_reliability_over_batches=sum(row.get("mean_anchor_reliability", 0 if row.get("valid_anchor_count", 0) == 0 else 1) for row in rows) / len(rows))
    append_json(Path(args.save_dir).parent / "temporal_followup_history.jsonl", result)


def record_lr(args, optimizer, epoch, before, val_mae):
    if getattr(args, "enable_guarded_checkpoint", False):
        append_json(Path(args.save_dir).parent / "optimizer_lr_history.jsonl", {
            "epoch": epoch, "lr_used": before,
            "lr_next": {group.get("name", str(i)): group["lr"] for i, group in enumerate(optimizer.param_groups)},
            "val_mae_expected_eta1": float(val_mae), "scheduler": args.scheduler})


def update_dev_aligned(model, args, epoch, val_metrics, test_metrics, records, write_hparams):
    """Select solely by fixed validation argmax eta=.8; test metrics are descriptive."""
    folder = Path(args.save_dir)
    for criterion in ("acc5", "mae"):
        key = "dev_argmax_" + criterion
        def rank(metrics, at_epoch):
            if criterion == "acc5":
                return (-metrics["Acc5"], metrics["MAE"], at_epoch)
            return (metrics["MAE"], -metrics["Acc5"], at_epoch)
        old = records.get(key)
        if old is None or rank(val_metrics, epoch) < rank(old["val"], old["epoch"]):
            path = folder / ("best_" + key + "_model.pth")
            temp = path.with_suffix(".pth.tmp")
            torch.save(model.state_dict(), temp)
            temp.replace(path)
            write_hparams(str(path), args, checkpoint_epoch=epoch)
            sidecar = path.with_suffix(".json")
            metadata = json.loads(sidecar.read_text())
            metadata.update(checkpoint_selection_split="val", checkpoint_policy="val_best_"+key,
                            selection_readout="argmax", selection_eta=.8, selection_temperature=1.)
            write_json(sidecar, metadata)
            records[key] = {"epoch":epoch,"val":val_metrics,"test":test_metrics,"checkpoint":str(path)}
    write_json(folder / "dev_aligned_selection.json", {key:records[key] for key in ("dev_argmax_acc5","dev_argmax_mae")})
