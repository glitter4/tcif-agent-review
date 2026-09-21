"""Plan, train, and evaluate the selected single-model setting (one seed only).

No scheduler submission is performed. Execute training/evaluation inside an
appropriate GPU allocation. Plan and summarize require only the standard library.
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "server-code"
sys.path.insert(0, str(CODE))
from tcif_ablation_config import TCIF_ABLATIONS, validate_training_ablation
from types import SimpleNamespace

ETAS = (0., .2, .4, .6, .8, .9, 1.)
CHECKPOINTS = ("best_acc7_model", "best_mae_model")
SELECTED_ETA = .8


def configuration(variant, output):
    cfg = json.loads((ROOT / "configs/tcif_paper_single_model.json").read_text(encoding="utf-8"))
    if variant not in TCIF_ABLATIONS:
        raise ValueError(variant)
    cfg.update(tcif_ablation=variant, run_name=f"tcif_ablation_{variant}", factor_code=variant,
               save_dir=str(output / variant / "checkpoints"),
               log_dir=str(output / variant / "tensorboard"))
    if variant in ("no_gate", "standard_context"):
        cfg.update(tcif_enable_transition_gate=False, tcif_transition_gate_loss_weight=0.)
    # Keep the original context auxiliary objective in the ordinary fusion
    # baseline: its masked pooled-context projection has the same two heads.
    validate_training_ablation(SimpleNamespace(**cfg))
    locked = dict(seed=40, tcif_latent_dim=128, tcif_context_radius=1,
                  final_pred_eta=.4, batch_size=64, grad_accum_steps=4,
                  epochs=10, lr=6e-5, router_lr=4e-4,
                  checkpoint_selection_split="test", checkpoint_metrics="dual")
    for key, value in locked.items():
        if cfg[key] != value:
            raise ValueError(f"Selected setting drifted: {key}={cfg[key]!r}")
    return cfg


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def train_command(python, config_path):
    return [python, "-u", str(CODE / "train_emotion.py"), "--training_config", str(config_path)]


def eval_command(python, cfg, destination, eta):
    return [python, "-u", str(CODE / "eval_all_mosei_maefixed.py"),
            "--checkpoints_root", cfg["save_dir"], "--results_root", str(destination),
            "--default_dataset", "cmumosei", "--default_dataset_root", cfg["dataset_root"],
            "--default_tokenizer_path", cfg["tokenizer_path"],
            "--default_bert_path", cfg["bert_backbone_path"],
            "--default_vit_path", cfg["vit_backbone_path"],
            "--default_hubert_path", cfg["hubert_model_path"],
            "--default_vision_backbone_path", cfg["vision_backbone_path"],
            "--output_head_mode", "signed_reg_cls7", "--final_pred_eta", str(eta),
            "--default_batch_size", "64", "--num_workers", "2"]


def eta_dir(output, variant, eta):
    return output / variant / "eta_sweep" / ("eta_" + f"{eta:.1f}".replace(".", "p"))


def validate_summary(path, variant, eta):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != 2 or {r.get("model_stem") for r in rows} != set(CHECKPOINTS):
        raise RuntimeError(f"Expected both selected checkpoints in {path}")
    if any(r.get("status") != "ok" or
           r.get("hparams", {}).get("tcif_ablation", "full") != variant or
           r.get("hparams", {}).get("final_pred_eta") != eta for r in rows):
        raise RuntimeError(f"Failed evaluation or incorrect ablation identity in {path}")
    return rows


def score_details(path):
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty predictions: {path}")
    truth = [float(r["true_value"]) for r in rows]
    pred = [float(r["pred_value"]) for r in rows]
    if not all(math.isfinite(v) for v in truth + pred):
        raise ValueError(f"Nonfinite predictions: {path}")
    def cls(v):
        return max(-3, min(3, math.floor(v+.5) if v >= 0 else math.ceil(v-.5)))
    nonzero = [(int(y >= 0), int(p >= 0)) for y, p in zip(truth, pred) if y != 0]
    f1s = []
    for c in (0, 1):
        tp = sum(y == c and p == c for y, p in nonzero)
        fp = sum(y != c and p == c for y, p in nonzero)
        fn = sum(y == c and p != c for y, p in nonzero)
        f1s.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(samples=len(rows), nonzero_samples=len(nonzero),
                acc7=100*sum(cls(y) == cls(p) for y,p in zip(truth,pred))/len(rows),
                mae=sum(abs(y-p) for y,p in zip(truth,pred))/len(rows),
                acc2non0=100*sum(y == p for y,p in nonzero)/len(nonzero) if nonzero else None,
                f1non0=50*sum(f1s) if nonzero else None)


def summarize(output):
    report = []
    for variant in TCIF_ABLATIONS:
        candidates = {name: [] for name in CHECKPOINTS}
        for eta in ETAS:
            directory = eta_dir(output, variant, eta)
            for row in validate_summary(directory / "all_models_summary.json", variant, eta):
                details = Path(row["output_dir"]) / f'{row["model_stem"]}_test_details.csv'
                candidates[row["model_stem"]].append(dict(eta=eta, **score_details(details)))
        for checkpoint, points in candidates.items():
            criterion = (lambda p: (-p["acc7"], p["mae"], p["eta"])) if checkpoint == CHECKPOINTS[0] else (
                lambda p: (p["mae"], -p["acc7"], p["eta"]))
            report.append(dict(variant=variant, checkpoint=checkpoint,
                               selected_eta=next(p for p in points if p["eta"] == SELECTED_ETA),
                               swept_best=min(points, key=criterion), sweep=points))
    write_json(output / "ablation_results.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "train", "evaluate", "summarize"))
    parser.add_argument("--variant", choices=TCIF_ABLATIONS)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tcif_paper_ablation")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if args.action == "plan":
        plans = []
        for variant in TCIF_ABLATIONS:
            cfg = configuration(variant, output)
            plans.append(dict(variant=variant, config=cfg,
                train=train_command(args.python, output / variant / "config.json"),
                evaluate=[eval_command(args.python, cfg, eta_dir(output, variant, eta), eta) for eta in ETAS]))
        print(json.dumps(dict(selected_seed=40, selected_filter_width=128, report_eta=.8,
                              configurations=plans), indent=2))
        return
    if args.action == "summarize":
        summarize(output)
        return
    if args.variant is None:
        parser.error("train/evaluate requires --variant")
    cfg = configuration(args.variant, output)
    if args.action == "train":
        config_path = output / args.variant / "config.json"
        if config_path.exists() or Path(cfg["save_dir"]).exists():
            raise FileExistsError("Run already exists; choose a fresh output root (no implicit overwrite)")
        # Prevent accidental long CPU training outside the intended allocation.
        subprocess.run([args.python, "-c", "import torch; assert torch.cuda.is_available(), 'GPU allocation required'"], check=True)
        write_json(config_path, cfg)
        subprocess.run(train_command(args.python, config_path), cwd=CODE, check=True)
    else:
        for checkpoint in CHECKPOINTS:
            stem = Path(cfg["save_dir"]) / checkpoint
            if not stem.with_suffix(".pth").is_file():
                raise FileNotFoundError(stem.with_suffix(".pth"))
            saved = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            for key, value in cfg.items():
                if key in ("run_name", "factor_code", "save_dir", "log_dir", "protocol_version"):
                    continue
                if saved.get(key, "full" if key == "tcif_ablation" else None) != value:
                    raise ValueError(f"Checkpoint setting mismatch: {key}")
        for eta in ETAS:
            destination = eta_dir(output, args.variant, eta)
            summary = destination / "all_models_summary.json"
            if summary.exists():
                validate_summary(summary, args.variant, eta)
                continue
            subprocess.run(eval_command(args.python, cfg, destination, eta), cwd=CODE, check=True)
            validate_summary(summary, args.variant, eta)


if __name__ == "__main__":
    main()
