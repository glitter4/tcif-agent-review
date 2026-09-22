import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STRUCT_ROOT = Path("/path/to/user/m4oe/StructV4.0")
PROJECT_ROOT = Path("/path/to/user/workspaces/m4oe")
CODE_ROOT = PROJECT_ROOT / "server-code"
PYTHON_BIN = "/path/to/user/.conda/envs/m4oe/bin/python"
DATASET_ROOT = "/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408"
DATE_TAG = "20260411"
ROUTER_LR = 2e-4
MSOE = 8
SEEDS = [23, 42]
T_SWEEP = [0.15, 0.20]
MTOE_SWEEP = [12, 8]
SBATCH_GPU_PARTITION = "A800-N"
SBATCH_CPU_PARTITION = "Intel-8358"
SBATCH_QOS = "normal"
SOURCE_JSON = STRUCT_ROOT / "unified_eval_source.json"
EXPORT_SCRIPT = STRUCT_ROOT / "build_unified_eval_table.py"
DASHBOARD_SCRIPT = STRUCT_ROOT / "unified_eval_dashboard.py"
TABLE_JSON = STRUCT_ROOT / "unified_eval_table.json"
DASHBOARD_HTML = STRUCT_ROOT / "unified_eval_dashboard.html"
PIPELINE_LOG = STRUCT_ROOT / f"pipeline_t_sweep_mtoe_sweep_{DATE_TAG}.log"
PIPELINE_SUMMARY = STRUCT_ROOT / f"pipeline_t_sweep_mtoe_sweep_{DATE_TAG}.json"
TEMPERATURE_COMPARISON_JSON = STRUCT_ROOT / f"temperature_comparison_{DATE_TAG}.json"
ROBETRA_PATH = "/path/to/user/models/AI-ModelScope_roberta-base"
VIT_PATH = "/path/to/user/models/vit-base-patch16-224-in21k"
HUBERT_PATH = "/path/to/user/models/hubert-base-ls960"
VISION_BACKBONE_PATH = "/path/to/user/models/resnet-18"
TRAIN_RE = re.compile(r"^Epoch (\d+) Train Loss: ([0-9.]+) Polarity Acc: ([0-9.]+) MacroF1: ([0-9.]+)")
VAL_RE = re.compile(
    r"^Epoch (\d+) Val Loss: ([0-9.]+) Polarity Acc: ([0-9.]+) MacroF1: ([0-9.]+) RMSE: ([0-9.]+) MAE: ([0-9.]+)"
)
FS_RE = re.compile(r"^Epoch (\d+) FinalScore Acc2: ([0-9.]+) Acc7: ([0-9.]+) MacroF1_7: ([0-9.]+)")


def now_iso():
    return datetime.now().replace(microsecond=0).isoformat()


def log(message: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    PIPELINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_cmd(args, cwd=None, check=True, capture_output=True):
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        text=True,
        capture_output=capture_output,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(args)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_payload():
    if not SOURCE_JSON.exists():
        return {"generated_at": now_iso(), "root_dir": str(STRUCT_ROOT), "rows": []}
    payload = read_json(SOURCE_JSON)
    if isinstance(payload, list):
        return {"generated_at": now_iso(), "root_dir": str(STRUCT_ROOT), "rows": payload}
    payload.setdefault("generated_at", now_iso())
    payload.setdefault("root_dir", str(STRUCT_ROOT))
    payload.setdefault("rows", [])
    return payload


def find_existing_train_row(run_name: str):
    payload = load_source_payload()
    for row in payload.get("rows", []):
        if row.get("stage") == "train" and row.get("row_id") == run_name:
            return row
    return None


def save_source_rows(rows):
    payload = load_source_payload()
    payload["generated_at"] = now_iso()
    payload["root_dir"] = str(STRUCT_ROOT)
    payload["rows"] = rows
    write_json(SOURCE_JSON, payload)


def upsert_rows(new_rows):
    payload = load_source_payload()
    rows = payload["rows"]
    by_id = {row["row_id"]: row for row in rows}
    for row in new_rows:
        by_id[row["row_id"]] = row
    merged = list(by_id.values())
    merged.sort(key=lambda row: (str(row.get("stage", "")), str(row.get("row_id", ""))))
    save_source_rows(merged)


def export_tables():
    run_cmd([PYTHON_BIN, str(EXPORT_SCRIPT), "--mode", "export", "--root-dir", str(STRUCT_ROOT)], check=True)
    run_cmd(
        [
            PYTHON_BIN,
            str(DASHBOARD_SCRIPT),
            "--input-json",
            str(TABLE_JSON),
            "--output-html",
            str(DASHBOARD_HTML),
        ],
        check=True,
    )


def current_git_commit():
    return run_cmd(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True).stdout.strip()


def temp_tag(value: float) -> str:
    return f"T{int(round(value * 100)):03d}"


def format_router_lr_tag(value: float) -> str:
    return "routerlr2e4" if abs(value - 2e-4) < 1e-12 else f"routerlr{value:g}"


def make_run_name(router_temperature: float, seed: int, mtoe: int) -> str:
    return (
        f"mosei_vit_msoe{MSOE}_mtoe{mtoe}_{format_router_lr_tag(ROUTER_LR)}_"
        f"{temp_tag(router_temperature)}_seed{seed}_vctx00_{DATE_TAG}"
    )


def build_hparams(router_temperature: float, seed: int, mtoe: int, run_name: str, exp_dir: Path):
    return {
        "dataset": "cmumosei",
        "dataset_root": DATASET_ROOT,
        "vision_backbone_type": "vit",
        "vit_backbone_path": VIT_PATH,
        "bert_backbone_path": ROBETRA_PATH,
        "tokenizer_path": ROBETRA_PATH,
        "hubert_model_path": HUBERT_PATH,
        "vision_backbone_path": VISION_BACKBONE_PATH,
        "num_experts_msoe": MSOE,
        "num_experts_mtoe": mtoe,
        "router_lr": ROUTER_LR,
        "router_temperature": router_temperature,
        "alpha": 0.03,
        "lambda_polarity": 1.0,
        "batch_size": 64,
        "num_frames": 4,
        "dropout": 0.3,
        "attention_dropout": 0.0,
        "epochs": 50,
        "lr": 3e-5,
        "backbone_lr_ratio": 0.05,
        "vit_context_ratio": 0.0,
        "seed": seed,
        "save_dir": str(exp_dir / "checkpoints"),
        "log_dir": str(exp_dir / "runs"),
        "run_name": run_name,
        "freeze_backbone_epochs": 100,
        "frame_policy": "middle",
        "local_files_only": True,
    }


def write_train_script(exp_dir: Path, hparams: dict):
    script_path = exp_dir / f"submit_train_{hparams['run_name']}.sh"
    checkpoints_dir = exp_dir / "checkpoints"
    content = f"""#!/bin/bash
#SBATCH --job-name={hparams['run_name'][:80]}
#SBATCH --partition={SBATCH_GPU_PARTITION}
#SBATCH --qos={SBATCH_QOS}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output={exp_dir}/slurm-%j.out

set -euo pipefail

PROJECT_ROOT={PROJECT_ROOT}
CODE_ROOT="$PROJECT_ROOT/server-code"
PYTHON_BIN={PYTHON_BIN}
SAVE_DIR={checkpoints_dir}

mkdir -p "$SAVE_DIR"
mkdir -p "{exp_dir / 'runs'}"
cd "$CODE_ROOT"

"$PYTHON_BIN" -u "$CODE_ROOT/train_emotion.py" \\
  --dataset cmumosei \\
  --dataset_root {DATASET_ROOT} \\
  --vision_backbone_type vit \\
  --vit_backbone_path {VIT_PATH} \\
  --bert_backbone_path {ROBETRA_PATH} \\
  --tokenizer_path {ROBETRA_PATH} \\
  --hubert_model_path {HUBERT_PATH} \\
  --vision_backbone_path {VISION_BACKBONE_PATH} \\
  --num_experts_msoe {hparams['num_experts_msoe']} \\
  --num_experts_mtoe {hparams['num_experts_mtoe']} \\
  --router_lr {hparams['router_lr']} \\
  --router_temperature {hparams['router_temperature']} \\
  --batch_size {hparams['batch_size']} \\
  --epochs {hparams['epochs']} \\
  --lr {hparams['lr']} \\
  --alpha {hparams['alpha']} \\
  --lambda_polarity {hparams['lambda_polarity']} \\
  --dropout {hparams['dropout']} \\
  --attention_dropout {hparams['attention_dropout']} \\
  --backbone_lr_ratio {hparams['backbone_lr_ratio']} \\
  --freeze_backbone_epochs {hparams['freeze_backbone_epochs']} \\
  --num_frames {hparams['num_frames']} \\
  --frame_policy {hparams['frame_policy']} \\
  --vit_context_ratio {hparams['vit_context_ratio']} \\
  --save_dir "$SAVE_DIR" \\
  --log_dir {exp_dir / 'runs'} \\
  --run_name {hparams['run_name']} \\
  --seed {hparams['seed']}
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o700)
    return script_path


def create_experiment(router_temperature: float, seed: int, mtoe: int):
    run_name = make_run_name(router_temperature, seed, mtoe)
    exp_dir = STRUCT_ROOT / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    hparams = build_hparams(router_temperature, seed, mtoe, run_name, exp_dir)
    write_json(exp_dir / "params.json", hparams)
    (exp_dir / "git_commit.txt").write_text(current_git_commit() + "\n", encoding="utf-8")
    notes = [
        "Current code variant: context add-back removed; context keeps concat-only path.",
        "IntensityGate text routing removed; task2 text-conditioned fusion router retained.",
        "Training checkpoint policy: keep val_acc7 top5, average val_mae top3 within the pool, keep only best_model.pth.",
    ]
    (exp_dir / "experiment_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")
    script_path = write_train_script(exp_dir, hparams)
    return {"run_name": run_name, "exp_dir": exp_dir, "hparams": hparams, "script_path": script_path}


def make_train_row(spec: dict, status: str, job_id=None):
    exp_dir = spec["exp_dir"]
    hparams = spec["hparams"]
    row = {
        "row_id": spec["run_name"],
        "results_root": str(STRUCT_ROOT),
        "output_dir": str(exp_dir),
        "model_path": str(exp_dir / "checkpoints" / "best_model.pth"),
        "relative_model_path": f"{spec['run_name']}/checkpoints/best_model.pth",
        "model_name": "best_model.pth",
        "model_stem": "best_model",
        "status": status,
        "stage": "train",
        "run_name": spec["run_name"],
        "combo_id": spec["run_name"],
        "hparams": hparams,
        "train_script": str(spec["script_path"]),
        "train_program": str(CODE_ROOT / "train_emotion.py"),
        "train_project_root": str(PROJECT_ROOT),
        "train_git_commit": current_git_commit(),
        "submitted_at": now_iso(),
        "structure_variant": "no_context",
    }
    if job_id is not None:
        row["train_job_id"] = str(job_id)
    return row


def submit_training(spec: dict):
    final_ckpt = spec["exp_dir"] / "checkpoints" / "best_model.pth"
    if final_ckpt.exists():
        log(f"skip_submit existing_final_ckpt run={spec['run_name']}")
        upsert_rows([make_train_row(spec, status="ok")])
        export_tables()
        return None
    existing_row = find_existing_train_row(spec["run_name"])
    if existing_row and existing_row.get("train_job_id"):
        existing_job_id = str(existing_row["train_job_id"])
        state = read_job_state(existing_job_id).upper()
        if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            log(f"resume_submit run={spec['run_name']} existing_job_id={existing_job_id} state={state}")
            return existing_job_id
    result = run_cmd(["sbatch", "--parsable", str(spec["script_path"])], check=True)
    job_id = result.stdout.strip().split(";")[0]
    (spec["exp_dir"] / "submitted_sbatch.sh").write_text(spec["script_path"].read_text(encoding="utf-8"), encoding="utf-8")
    log(f"submitted run={spec['run_name']} job_id={job_id}")
    upsert_rows([make_train_row(spec, status="submitted", job_id=job_id)])
    export_tables()
    return job_id


def slurm_log_for_job(exp_dir: Path, job_id: str) -> Path:
    return exp_dir / f"slurm-{job_id}.out"


def read_job_state(job_id: str):
    state = run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"], check=False).stdout.strip()
    if state:
        return state
    acct = run_cmd(["sacct", "-j", str(job_id), "--format=State", "--noheader"], check=False).stdout.strip().splitlines()
    acct = [line.strip().split()[0] for line in acct if line.strip()]
    return acct[0] if acct else ""


def wait_for_epoch1(spec: dict, job_id: str, timeout_sec: int = 7200):
    log_path = slurm_log_for_job(spec["exp_dir"], job_id)
    start = time.time()
    while time.time() - start < timeout_sec:
        log_path = slurm_log_for_job(spec["exp_dir"], job_id)
        if log_path and log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if "Epoch 1 Train Loss:" in text:
                log(f"epoch1_started run={spec['run_name']} job_id={job_id} log={log_path}")
                return True
            if "Traceback" in text or "RuntimeError" in text:
                raise RuntimeError(f"Training failed before epoch 1 for {spec['run_name']}. See {log_path}")
        state = read_job_state(job_id)
        if state and state.upper() in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise RuntimeError(f"Training job failed before epoch 1 for {spec['run_name']}: state={state}")
        time.sleep(20)
    raise TimeoutError(f"Timed out waiting for epoch 1 start for {spec['run_name']}")


def latest_slurm_log(exp_dir: Path):
    logs = sorted(exp_dir.glob("slurm-*.out"))
    return logs[-1] if logs else None


def wait_for_training_completion(spec: dict, job_id: str):
    final_ckpt = spec["exp_dir"] / "checkpoints" / "best_model.pth"
    while True:
        state = read_job_state(job_id)
        if state:
            upper = state.upper()
            if upper in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                raise RuntimeError(f"Training job failed for {spec['run_name']}: state={state}")
            if upper in {"COMPLETED"} and final_ckpt.exists():
                break
        if final_ckpt.exists():
            break
        time.sleep(60)
    row = make_train_row(spec, status="ok", job_id=job_id)
    summary_path = spec["exp_dir"] / "checkpoints" / "checkpoint_selection_summary.json"
    if summary_path.exists():
        row["checkpoint_selection_summary"] = str(summary_path)
        summary = read_json(summary_path)
        row["selected_epochs"] = [item["epoch"] for item in summary.get("selected_for_averaging", [])]
        row["top_acc7_candidate_epochs"] = [item["epoch"] for item in summary.get("top_acc7_candidates", [])]
    row["completed_at"] = now_iso()
    upsert_rows([row])
    export_tables()
    log(f"training_complete run={spec['run_name']} job_id={job_id} final_ckpt={final_ckpt}")


def parse_best_val_metrics(slurm_path: Path):
    epochs = {}
    last_section = None
    for line in slurm_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        m = TRAIN_RE.match(line)
        if m:
            ep = int(m.group(1))
            epochs.setdefault(ep, {})
            last_section = ("train", ep)
            continue
        m = VAL_RE.match(line)
        if m:
            ep = int(m.group(1))
            row = epochs.setdefault(ep, {})
            row["val_loss"] = float(m.group(2))
            row["val_binary_acc"] = float(m.group(3))
            row["val_binary_f1"] = float(m.group(4))
            row["val_rmse"] = float(m.group(5))
            row["val_mae"] = float(m.group(6))
            last_section = ("val", ep)
            continue
        m = FS_RE.match(line)
        if m and last_section and last_section[1] == int(m.group(1)):
            ep = int(m.group(1))
            row = epochs.setdefault(ep, {})
            prefix = "val_" if last_section[0] == "val" else "train_"
            row[prefix + "acc2"] = float(m.group(2))
            row[prefix + "acc7"] = float(m.group(3))
            row[prefix + "macro_f1_7"] = float(m.group(4))
    val_epochs = [ep for ep, row in epochs.items() if "val_acc7" in row and "val_mae" in row]
    if not val_epochs:
        raise RuntimeError(f"No val metrics found in {slurm_path}")
    best_acc7_ep = max(val_epochs, key=lambda ep: epochs[ep]["val_acc7"])
    best_mae_ep = min(val_epochs, key=lambda ep: epochs[ep]["val_mae"])
    return {
        "best_acc7_epoch": best_acc7_ep,
        "best_acc7": epochs[best_acc7_ep]["val_acc7"],
        "best_mae_epoch": best_mae_ep,
        "best_mae": epochs[best_mae_ep]["val_mae"],
        "best_acc2_at_best_acc7": epochs[best_acc7_ep].get("val_acc2"),
    }


def compute_non0_metrics(details_csv: Path):
    rows = list(csv.DictReader(open(details_csv, "r", encoding="utf-8")))
    subset = []
    for row in rows:
        true_value = float(row["true_value"])
        pred_value = float(row["pred_value"])
        if true_value != 0.0:
            subset.append((true_value, pred_value))
    if not subset:
        return {"test_acc2_non0": None, "test_f1_non0": None, "test_non0_num_samples": 0}
    gt = [1 if t >= 0.0 else 0 for t, _ in subset]
    pred = [1 if p >= 0.0 else 0 for _, p in subset]
    correct = sum(1 for g, p in zip(gt, pred) if g == p)
    acc2 = correct / len(subset)
    conf = [[0, 0], [0, 0]]
    for g, p in zip(gt, pred):
        conf[g][p] += 1
    f1s = []
    for i in range(2):
        tp = conf[i][i]
        fn = conf[i][1 - i]
        fp = conf[1 - i][i]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return {
        "test_acc2_non0": acc2,
        "test_f1_non0": sum(f1s) / 2.0,
        "test_non0_num_samples": len(subset),
    }


def prepare_eval_targets(tag: str, specs: list):
    target_root = STRUCT_ROOT / f"eval_targets_{tag}_{DATE_TAG}"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        dst_dir = target_root / spec["run_name"]
        dst_dir.mkdir(parents=True, exist_ok=True)
        model_src = spec["exp_dir"] / "checkpoints" / "best_model.pth"
        model_dst = dst_dir / "best_model.pth"
        os.symlink(model_src, model_dst)
        write_json(dst_dir / "best_model.json", spec["hparams"])
    return target_root


def run_eval(tag: str, specs: list):
    target_root = prepare_eval_targets(tag, specs)
    results_root = STRUCT_ROOT / f"eval_{tag}_bestmodel_valtest_{DATE_TAG}"
    results_root.mkdir(parents=True, exist_ok=True)
    log(f"eval_start tag={tag} target_root={target_root} results_root={results_root}")
    run_cmd(
        [
            PYTHON_BIN,
            str(CODE_ROOT / "eval_all_mosei_maefixed.py"),
            "--checkpoints_root",
            str(target_root),
            "--results_root",
            str(results_root),
            "--default_dataset",
            "cmumosei",
            "--default_dataset_root",
            DATASET_ROOT,
            "--default_tokenizer_path",
            ROBETRA_PATH,
            "--default_bert_path",
            ROBETRA_PATH,
            "--default_vit_path",
            VIT_PATH,
            "--default_hubert_path",
            HUBERT_PATH,
        ],
        check=True,
    )
    return results_root


def build_eval_row(spec: dict, results_root: Path):
    out_dir = results_root / spec["run_name"] / "best_model"
    val_result = read_json(out_dir / "val_results.json")["metrics"]
    test_result = read_json(out_dir / "test_results.json")["metrics"]
    non0 = compute_non0_metrics(out_dir / "test_details.csv")
    row = {
        "row_id": f"{spec['run_name']}/best_model",
        "results_root": str(STRUCT_ROOT),
        "eval_results_root": str(results_root),
        "output_dir": str(out_dir),
        "model_path": str(spec["exp_dir"] / "checkpoints" / "best_model.pth"),
        "relative_model_path": f"{spec['run_name']}/checkpoints/best_model.pth",
        "model_name": "best_model.pth",
        "model_stem": "best_model",
        "status": "ok",
        "stage": "test_eval",
        "run_name": spec["run_name"],
        "combo_id": spec["run_name"],
        "hparams": spec["hparams"],
        "val_mae": val_result["mae"],
        "test_mae": test_result["mae"],
        "val_acc7": val_result["acc7"],
        "test_acc7": test_result["acc7"],
        "val_acc2": val_result["acc2"],
        "test_acc2": test_result["acc2"],
        "val_binary_f1": val_result["binary_f1"],
        "test_binary_f1": test_result["binary_f1"],
        "val_rmse": val_result["rmse"],
        "test_rmse": test_result["rmse"],
        "val_pearson": val_result["pearson"],
        "test_pearson": test_result["pearson"],
        "val_spearman": val_result["spearman"],
        "test_spearman": test_result["spearman"],
        "inference_program": str(CODE_ROOT / "eval_all_mosei_maefixed.py"),
        "inference_project_root": str(PROJECT_ROOT),
        "inference_git_commit": current_git_commit(),
        "structure_variant": "no_context",
        "evaluated_split": "val_test",
        "updated_at": now_iso(),
        "val_metric_source": str(out_dir / "val_results.json"),
        "test_metric_source": str(out_dir / "test_results.json"),
        "test_acc2_non0": non0["test_acc2_non0"],
        "test_f1_non0": non0["test_f1_non0"],
        "test_non0_num_samples": non0["test_non0_num_samples"],
        "test_non0_metric_source": str(out_dir / "test_details.csv"),
    }
    return row


def compare_temperatures(stage1_specs: list):
    baseline_runs = [
        {
            "temperature": 0.10,
            "seed": 23,
            "run_name": "mosei_vit_msoe8_mtoe16_routerlr2e4_seed23_vctx00_20260409",
            "exp_dir": STRUCT_ROOT / "mosei_vit_msoe8_mtoe16_routerlr2e4_seed23_vctx00_20260409",
        },
        {
            "temperature": 0.10,
            "seed": 42,
            "run_name": "mosei_vit_msoe8_mtoe16_routerlr2e4_seed42_vctx00_20260409",
            "exp_dir": STRUCT_ROOT / "mosei_vit_msoe8_mtoe16_routerlr2e4_seed42_vctx00_20260409",
        },
    ]
    aggregates = {}
    for item in baseline_runs + stage1_specs:
        log_path = latest_slurm_log(item["exp_dir"])
        if log_path is None:
            raise RuntimeError(f"Missing slurm log for temperature comparison: {item['run_name']}")
        parsed = parse_best_val_metrics(log_path)
        bucket = aggregates.setdefault(item["temperature"], [])
        bucket.append(
            {
                "run_name": item["run_name"],
                "seed": item["seed"],
                "best_val_acc7": parsed["best_acc7"],
                "best_val_mae": parsed["best_mae"],
                "best_val_acc7_epoch": parsed["best_acc7_epoch"],
                "best_val_mae_epoch": parsed["best_mae_epoch"],
            }
        )
    comparison_rows = []
    for temp, rows in sorted(aggregates.items()):
        mean_acc7 = sum(row["best_val_acc7"] for row in rows) / len(rows)
        mean_mae = sum(row["best_val_mae"] for row in rows) / len(rows)
        comparison_rows.append(
            {
                "router_temperature": temp,
                "num_runs": len(rows),
                "mean_best_val_acc7": mean_acc7,
                "mean_best_val_mae": mean_mae,
                "runs": rows,
            }
        )
    comparison_rows.sort(key=lambda row: (-row["mean_best_val_acc7"], row["mean_best_val_mae"], row["router_temperature"]))
    payload = {
        "generated_at": now_iso(),
        "comparison_metric": "mean_best_val_acc7_then_mean_best_val_mae",
        "rows": comparison_rows,
        "selected_router_temperature": comparison_rows[0]["router_temperature"],
    }
    write_json(TEMPERATURE_COMPARISON_JSON, payload)
    log(f"selected_temperature={payload['selected_router_temperature']} comparison={TEMPERATURE_COMPARISON_JSON}")
    return payload["selected_router_temperature"], payload


def monitor_first_epoch(stage_name: str, submissions: list):
    for spec, job_id in submissions:
        wait_for_epoch1(spec, job_id)
    log(f"{stage_name}_all_started")


def stage_submit_and_wait(stage_name: str, specs: list):
    submissions = []
    for spec in specs:
        job_id = submit_training(spec)
        if job_id:
            submissions.append((spec, job_id))
    if submissions:
        monitor_first_epoch(stage_name, submissions)
        for spec, job_id in submissions:
            wait_for_training_completion(spec, job_id)
    else:
        log(f"{stage_name}_all_skipped_existing")


def build_stage2_specs(best_temperature: float):
    specs = []
    for mtoe in MTOE_SWEEP:
        for seed in SEEDS:
            spec = create_experiment(best_temperature, seed, mtoe)
            spec["temperature"] = best_temperature
            spec["seed"] = seed
            spec["mtoe"] = mtoe
            specs.append(spec)
    return specs


def run_stage_eval(tag: str, specs: list):
    results_root = run_eval(tag, specs)
    rows = [build_eval_row(spec, results_root) for spec in specs]
    upsert_rows(rows)
    export_tables()
    log(f"eval_complete tag={tag} results_root={results_root}")
    return results_root, rows


def main():
    log("pipeline_start")
    pipeline_summary = {
        "generated_at": now_iso(),
        "git_commit": current_git_commit(),
        "orchestrator_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "assumptions": [
            "Stage 2 MToE sweep uses both seed=23 and seed=42 for MToE=12 and MToE=8.",
            "Temperature selection uses mean best validation Acc7 across seeds, with mean best validation MAE as tie-breaker.",
        ],
    }

    stage1_specs = []
    for temp in T_SWEEP:
        for seed in SEEDS:
            spec = create_experiment(temp, seed, 16)
            spec["temperature"] = temp
            spec["seed"] = seed
            spec["mtoe"] = 16
            stage1_specs.append(spec)

    stage_submit_and_wait("stage1_t_sweep", stage1_specs)
    stage1_results_root, stage1_rows = run_stage_eval("tsweep", stage1_specs)
    best_temperature, comparison_payload = compare_temperatures(stage1_specs)

    stage2_specs = build_stage2_specs(best_temperature)
    stage_submit_and_wait("stage2_mtoe_sweep", stage2_specs)
    stage2_results_root, stage2_rows = run_stage_eval("mtoesweep", stage2_specs)

    pipeline_summary.update(
        {
            "stage1_runs": [spec["run_name"] for spec in stage1_specs],
            "stage1_eval_results_root": str(stage1_results_root),
            "selected_router_temperature": best_temperature,
            "temperature_comparison": comparison_payload,
            "stage2_runs": [spec["run_name"] for spec in stage2_specs],
            "stage2_eval_results_root": str(stage2_results_root),
            "completed_at": now_iso(),
        }
    )
    write_json(PIPELINE_SUMMARY, pipeline_summary)
    log(f"pipeline_complete summary={PIPELINE_SUMMARY}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"pipeline_failed error={exc}")
        raise
