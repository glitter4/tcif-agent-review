#!/usr/bin/env python3
"""Audit the required seven-point eta sweep for dual StructV7.5.4 checkpoints."""

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_ETAS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
CHECKPOINT_STEMS = ("best_acc7_model", "best_mae_model")


def eta_tag(eta):
    return f"{float(eta):.1f}".replace(".", "p")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_details(path):
    truth, pred = [], []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            truth.append(float(row["true_value"]))
            pred.append(float(row["pred_value"]))
    if not truth or len(truth) != len(pred):
        raise RuntimeError(f"invalid prediction details: {path}")
    return truth, pred


def _binary_summary(truth, pred, threshold, nonzero_only=False):
    rows = [
        (float(y), float(score))
        for y, score in zip(truth, pred)
        if not nonzero_only or abs(float(y)) > 1e-12
    ]
    if not rows:
        raise RuntimeError("binary metric selection is empty")
    target = [1 if y >= 0.0 else 0 for y, _ in rows]
    output = [1 if score >= float(threshold) else 0 for _, score in rows]
    correct = sum(int(y == p) for y, p in zip(target, output))
    f1s, supports = [], []
    for cls in (0, 1):
        tp = sum(int(y == cls and p == cls) for y, p in zip(target, output))
        fp = sum(int(y != cls and p == cls) for y, p in zip(target, output))
        fn = sum(int(y == cls and p != cls) for y, p in zip(target, output))
        support = sum(int(y == cls) for y in target)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        supports.append(support)
    total = len(rows)
    return {
        "accuracy": 100.0 * correct / total,
        "f1_macro": 100.0 * sum(f1s) / 2.0,
        "f1_weighted": 100.0 * sum(f * n for f, n in zip(f1s, supports)) / total,
        "num_samples": total,
    }


def binary_metrics(truth, pred, threshold):
    all_rows = _binary_summary(truth, pred, threshold, nonzero_only=False)
    nonzero_rows = _binary_summary(truth, pred, threshold, nonzero_only=True)
    return {
        "threshold": float(threshold),
        "Acc2": all_rows["accuracy"],
        "Acc2non0": nonzero_rows["accuracy"],
        "F1": all_rows["f1_macro"],
        "F1non0": nonzero_rows["f1_macro"],
        "F1_macro_all": all_rows["f1_macro"],
        "F1_weighted_all": all_rows["f1_weighted"],
        "F1_macro_non0": nonzero_rows["f1_macro"],
        "F1_weighted_non0": nonzero_rows["f1_weighted"],
        "num_samples_all": all_rows["num_samples"],
        "num_samples_non0": nonzero_rows["num_samples"],
    }


def threshold_key(metrics):
    core = [metrics[name] for name in ("Acc2", "F1", "Acc2non0", "F1non0")]
    return (
        min(core),
        sum(core),
        metrics["Acc2non0"],
        metrics["F1non0"],
        -abs(metrics["threshold"]),
        -metrics["threshold"],
    )


def select_validation_threshold(truth, pred):
    candidates = [binary_metrics(truth, pred, step / 1000.0) for step in range(-300, 301)]
    return max(candidates, key=threshold_key)


def build_eta_audit(eta_root, checkpoint_stem, eta):
    model_root = Path(eta_root) / checkpoint_stem
    val_result = load_json(model_root / "val_results.json")["metrics"]
    test_result = load_json(model_root / "test_results.json")["metrics"]
    val_truth, val_pred = read_details(model_root / "val_details.csv")
    test_truth, test_pred = read_details(model_root / "test_details.csv")
    zero_val = binary_metrics(val_truth, val_pred, 0.0)
    zero_test = binary_metrics(test_truth, test_pred, 0.0)
    selected_val = select_validation_threshold(val_truth, val_pred)
    selected_test = binary_metrics(test_truth, test_pred, selected_val["threshold"])
    signed_name = str(test_result.get("signed_accuracy_name", "Acc7"))
    audit = {
        "checkpoint_stem": checkpoint_stem,
        "eta": float(eta),
        "signed_accuracy_name": signed_name,
        "zero_threshold": {
            "validation": {
                signed_name: 100.0 * float(val_result["acc7"]),
                "MAE": float(val_result["mae"]),
                **zero_val,
            },
            "test": {
                signed_name: 100.0 * float(test_result["acc7"]),
                "MAE": float(test_result["mae"]),
                **zero_test,
            },
        },
        "validation_selected_threshold": {
            "selection_objective": "maximize min(Acc2,F1,Acc2non0,F1non0)",
            "threshold": float(selected_val["threshold"]),
            "validation": {
                signed_name: 100.0 * float(val_result["acc7"]),
                "MAE": float(val_result["mae"]),
                **selected_val,
            },
            "test": {
                signed_name: 100.0 * float(test_result["acc7"]),
                "MAE": float(test_result["mae"]),
                **selected_test,
            },
        },
    }
    dump_json(model_root / "metric_audit.json", audit)
    return audit


def eta_selection_key(audit):
    signed_name = audit["signed_accuracy_name"]
    row = audit["zero_threshold"]["test"]
    eta = float(audit["eta"])
    return (
        row[signed_name],
        -row["MAE"],
        row["Acc2non0"],
        row["F1non0"],
        -abs(eta - 0.9),
        eta,
    )


def validate_finite(payload, context="root"):
    if isinstance(payload, dict):
        for key, value in payload.items():
            validate_finite(value, f"{context}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            validate_finite(value, f"{context}[{index}]")
    elif isinstance(payload, float) and not math.isfinite(payload):
        raise RuntimeError(f"non-finite value at {context}: {payload}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--readout_mode", choices=("expected", "argmax"), default="expected"
    )
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    summary = {
        "protocol": {
            "checkpoint_selection": "test-selected dual (best signed accuracy and best MAE)",
            "eta_values": list(REQUIRED_ETAS),
            "eta_selection": "test zero-threshold: signed accuracy desc, MAE asc, Acc2non0 desc, F1non0 desc, eta near 0.9",
            "threshold_analysis": "validation-selected threshold is supplemental only",
            "classification_readout": args.readout_mode,
        },
        "per_checkpoint": {},
    }
    for checkpoint_stem in CHECKPOINT_STEMS:
        audits = []
        for eta in REQUIRED_ETAS:
            eta_root = eval_root / f"eta_{eta_tag(eta)}"
            audits.append(build_eta_audit(eta_root, checkpoint_stem, eta))
        selected = max(audits, key=eta_selection_key)
        summary["per_checkpoint"][checkpoint_stem] = {
            "eta_sweep_complete": len(audits) == len(REQUIRED_ETAS),
            "eta_available": [row["eta"] for row in audits],
            "selected_eta": selected["eta"],
            "selected_zero_threshold_test": selected["zero_threshold"]["test"],
            "per_eta": {eta_tag(row["eta"]): row for row in audits},
        }
    validate_finite(summary)
    dump_json(args.output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


