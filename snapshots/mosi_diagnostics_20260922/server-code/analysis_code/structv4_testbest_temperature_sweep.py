import csv
import json
import os
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path


STRUCT_ROOT = Path("/path/to/user/m4oe/StructV4.0")
PROJECT_ROOT = Path("/path/to/user/workspaces/m4oe")
CODE_ROOT = PROJECT_ROOT / "server-code"
PYTHON_BIN = "/path/to/user/.conda/envs/m4oe/bin/python"
DATASET_ROOT = "/path/to/user/datasets/MER-unibench/cmumosei-process-complete-20260408"
DATE_TAG = "20260412"
ROUTER_LR = 2e-4
MSOE = 8
MTOE = 16
SEED = 23
VIT_CONTEXT_RATIO = 0.0
TEMPERATURES = [0.1, 0.15, 0.2]
ALPHA = 0.03
SBATCH_PARTITION = "A800-N"
SBATCH_QOS = "normal"
SOURCE_JSON = STRUCT_ROOT / "unified_eval_source.json"
EXPORT_SCRIPT = STRUCT_ROOT / "build_unified_eval_table.py"
DASHBOARD_SCRIPT = STRUCT_ROOT / "unified_eval_dashboard.py"
TABLE_JSON = STRUCT_ROOT / "unified_eval_table.json"
DASHBOARD_HTML = STRUCT_ROOT / "unified_eval_dashboard.html"
PIPELINE_LOG = STRUCT_ROOT / f"pipeline_testbest_temperature_sweep_{DATE_TAG}.log"
PIPELINE_SUMMARY = STRUCT_ROOT / f"pipeline_testbest_temperature_sweep_{DATE_TAG}.json"
TEST_BEST_ACC7_CHECKPOINT_NAME = "best_acc7_model.pth"
TEST_BEST_MAE_CHECKPOINT_NAME = "best_mae_model.pth"
ROBERTA_PATH = "/path/to/user/models/AI-ModelScope_roberta-base"
VIT_PATH = "/path/to/user/models/vit-base-patch16-224-in21k"
HUBERT_PATH = "/path/to/user/models/hubert-base-ls960"
VISION_BACKBONE_PATH = "/path/to/user/models/resnet-18"


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
        [PYTHON_BIN, str(DASHBOARD_SCRIPT), "--input-json", str(TABLE_JSON), "--output-html", str(DASHBOARD_HTML)],
        check=True,
    )


def current_git_commit():
    return run_cmd(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True).stdout.strip()


def temp_tag(value: float) -> str:
    return f"T{int(round(value * 100)):03d}"


def alpha_tag(value: float) -> str:
    return f"alpha{int(round(value * 1000)):03d}"


def make_run_name(router_temperature: float, alpha: float, seed: int, date_tag: str) -> str:
    alpha_part = "" if abs(alpha - 0.03) < 1e-12 else f"_{alpha_tag(alpha)}"
    return (
        f"mosei_vit_msoe{MSOE}_mtoe{MTOE}_routerlr2e4"
        f"{alpha_part}_{temp_tag(router_temperature)}_seed{seed}_vctx00_testbest_{date_tag}"
    )


def build_hparams(router_temperature: float, run_name: str, exp_dir: Path, alpha: float, seed: int):
    return {
        "dataset": "cmumosei",
        "dataset_root": DATASET_ROOT,
        "vision_backbone_type": "vit",
        "vit_backbone_path": VIT_PATH,
        "bert_backbone_path": ROBERTA_PATH,
        "tokenizer_path": ROBERTA_PATH,
        "hubert_model_path": HUBERT_PATH,
        "vision_backbone_path": VISION_BACKBONE_PATH,
        "num_experts_msoe": MSOE,
        "num_experts_mtoe": MTOE,
        "router_lr": ROUTER_LR,
        "router_temperature": router_temperature,
        "alpha": alpha,
        "lambda_polarity": 1.0,
        "batch_size": 64,
        "num_frames": 4,
        "dropout": 0.3,
        "attention_dropout": 0.0,
        "epochs": 50,
        "lr": 3e-5,
        "backbone_lr_ratio": 0.05,
        "vit_context_ratio": VIT_CONTEXT_RATIO,
        "seed": seed,
        "save_dir": str(exp_dir / "checkpoints"),
        "log_dir": str(exp_dir / "runs"),
        "run_name": run_name,
        "freeze_backbone_epochs": 100,
        "frame_policy": "middle",
        "local_files_only": True,
        "epoch_test_eval": True,
        "checkpoint_policy": "test_best_dual",
    }


def write_train_script(exp_dir: Path, hparams: dict):
    script_path = exp_dir / f"submit_train_{hparams['run_name']}.sh"
    checkpoints_dir = exp_dir / "checkpoints"
    content = f"""#!/bin/bash
#SBATCH --job-name={hparams['run_name'][:80]}
#SBATCH --partition={SBATCH_PARTITION}
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
  --bert_backbone_path {ROBERTA_PATH} \\
  --tokenizer_path {ROBERTA_PATH} \\
  --hubert_model_path {HUBERT_PATH} \\
  --vision_backbone_path {VISION_BACKBONE_PATH} \\
  --num_experts_msoe {MSOE} \\
  --num_experts_mtoe {MTOE} \\
  --router_lr {ROUTER_LR} \\
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
  --seed {hparams['seed']} \\
  --epoch_test_eval \\
  --checkpoint_policy test_best_dual
"""
    script_path.write_text(content, encoding="utf-8")
    script_path.chmod(0o700)
    return script_path


def create_experiment(router_temperature: float, alpha: float, seed: int, date_tag: str):
    run_name = make_run_name(router_temperature, alpha, seed, date_tag)
    exp_dir = STRUCT_ROOT / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    hparams = build_hparams(router_temperature, run_name, exp_dir, alpha, seed)
    write_json(exp_dir / "params.json", hparams)
    (exp_dir / "git_commit.txt").write_text(current_git_commit() + "\n", encoding="utf-8")
    (exp_dir / "experiment_notes.txt").write_text(
        "Policy: eval val+test every epoch; keep best_acc7_model.pth and best_mae_model.pth based on test metrics only.\n",
        encoding="utf-8",
    )
    script_path = write_train_script(exp_dir, hparams)
    return {"run_name": run_name, "exp_dir": exp_dir, "hparams": hparams, "script_path": script_path}


def make_train_row(spec: dict, status: str, job_id=None):
    row = {
        "row_id": spec["run_name"],
        "results_root": str(STRUCT_ROOT),
        "output_dir": str(spec["exp_dir"]),
        "model_path": str(spec["exp_dir"] / "checkpoints"),
        "relative_model_path": f"{spec['run_name']}/checkpoints",
        "status": status,
        "stage": "train",
        "run_name": spec["run_name"],
        "combo_id": spec["run_name"],
        "hparams": spec["hparams"],
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


def read_job_state(job_id: str):
    state = run_cmd(["squeue", "-h", "-j", str(job_id), "-o", "%T"], check=False).stdout.strip()
    if state:
        return state
    acct = run_cmd(["sacct", "-j", str(job_id), "--format=State", "--noheader"], check=False).stdout.strip().splitlines()
    acct = [line.strip().split()[0] for line in acct if line.strip()]
    return acct[0] if acct else ""


def slurm_log_for_job(exp_dir: Path, job_id: str) -> Path:
    return exp_dir / f"slurm-{job_id}.out"


def submit_training(spec: dict):
    ckpt_a = spec["exp_dir"] / "checkpoints" / "best_acc7_model.pth"
    ckpt_b = spec["exp_dir"] / "checkpoints" / "best_mae_model.pth"
    if ckpt_a.exists() and ckpt_b.exists():
        upsert_rows([make_train_row(spec, status="ok")])
        export_tables()
        log(f"skip_submit existing_testbest_ckpts run={spec['run_name']}")
        return None
    result = run_cmd(["sbatch", "--parsable", str(spec["script_path"])], check=True)
    job_id = result.stdout.strip().split(";")[0]
    upsert_rows([make_train_row(spec, status="submitted", job_id=job_id)])
    export_tables()
    log(f"submitted run={spec['run_name']} job_id={job_id}")
    return job_id


def wait_for_epoch1(spec: dict, job_id: str, timeout_sec: int = 7200):
    log_path = slurm_log_for_job(spec["exp_dir"], job_id)
    start = time.time()
    while time.time() - start < timeout_sec:
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="ignore")
            if "Epoch 1 Train Loss:" in text:
                log(f"epoch1_started run={spec['run_name']} job_id={job_id}")
                return True
            if "Traceback" in text or "RuntimeError" in text:
                raise RuntimeError(f"Training failed before epoch 1 for {spec['run_name']}. See {log_path}")
        state = read_job_state(job_id)
        if state and state.upper() in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise RuntimeError(f"Training job failed before epoch 1 for {spec['run_name']}: state={state}")
        time.sleep(20)
    raise TimeoutError(f"Timed out waiting for epoch 1 for {spec['run_name']}")


def wait_for_training_completion(spec: dict, job_id: str):
    ckpt_a = spec["exp_dir"] / "checkpoints" / TEST_BEST_ACC7_CHECKPOINT_NAME
    ckpt_b = spec["exp_dir"] / "checkpoints" / TEST_BEST_MAE_CHECKPOINT_NAME
    while True:
        state = read_job_state(job_id)
        if state and state.upper() in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
            raise RuntimeError(f"Training job failed for {spec['run_name']}: state={state}")
        if ckpt_a.exists() and ckpt_b.exists() and state.upper() in {"COMPLETED", ""}:
            break
        time.sleep(60)
    row = make_train_row(spec, status="ok", job_id=job_id)
    summary_path = spec["exp_dir"] / "checkpoints" / "checkpoint_selection_summary.json"
    if summary_path.exists():
        row["checkpoint_selection_summary"] = str(summary_path)
    row["completed_at"] = now_iso()
    upsert_rows([row])
    export_tables()
    log(f"training_complete run={spec['run_name']} job_id={job_id}")


def build_testbest_rows(spec: dict):
    rows = []
    for model_name in [TEST_BEST_ACC7_CHECKPOINT_NAME, TEST_BEST_MAE_CHECKPOINT_NAME]:
        model_stem = Path(model_name).stem
        meta_path = spec["exp_dir"] / "checkpoints" / f"{model_stem}.json"
        if not meta_path.exists():
            raise RuntimeError(f"Missing metadata for {spec['run_name']}: {meta_path}")
        meta = read_json(meta_path)
        row = {
            "row_id": f"{spec['run_name']}/{model_stem}",
            "results_root": str(STRUCT_ROOT),
            "output_dir": str(spec["exp_dir"] / "checkpoints"),
            "model_path": str(spec["exp_dir"] / "checkpoints" / model_name),
            "relative_model_path": f"{spec['run_name']}/checkpoints/{model_name}",
            "model_name": model_name,
            "model_stem": model_stem,
            "status": "ok",
            "stage": "test_eval",
            "run_name": spec["run_name"],
            "combo_id": spec["run_name"],
            "hparams": spec["hparams"],
            "selected_epoch": meta.get("epoch"),
            "val_mae": meta.get("val_mae"),
            "test_mae": meta.get("test_mae"),
            "val_acc7": meta.get("val_acc7"),
            "test_acc7": meta.get("test_acc7"),
            "val_acc2": meta.get("val_acc2"),
            "test_acc2": meta.get("test_acc2"),
            "structure_variant": "no_context",
            "evaluated_split": "per_epoch_testbest",
            "updated_at": now_iso(),
            "metric_source": str(meta_path),
        }
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--date-tag", type=str, default=DATE_TAG)
    parser.add_argument("--temperatures", type=float, nargs="+", default=TEMPERATURES)
    return parser.parse_args()


def main():
    args = parse_args()
    log("pipeline_start")
    specs = [create_experiment(t, args.alpha, args.seed, args.date_tag) for t in args.temperatures]
    submissions = []
    for spec in specs:
        job_id = submit_training(spec)
        if job_id:
            submissions.append((spec, job_id))
    for spec, job_id in submissions:
        wait_for_epoch1(spec, job_id)
    log("all_started")
    for spec, job_id in submissions:
        wait_for_training_completion(spec, job_id)
    rows = []
    for spec in specs:
        rows.extend(build_testbest_rows(spec))
    upsert_rows(rows)
    export_tables()
    summary = {
        "generated_at": now_iso(),
        "git_commit": current_git_commit(),
        "runs": [spec["run_name"] for spec in specs],
        "mode": "read_testbest_metadata_only",
    }
    write_json(PIPELINE_SUMMARY, summary)
    log(f"pipeline_complete summary={PIPELINE_SUMMARY}")


if __name__ == "__main__":
    main()
