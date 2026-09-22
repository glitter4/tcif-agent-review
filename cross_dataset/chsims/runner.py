"""C1 fixed center-L1 seed experiment, with complete eta audit."""
import argparse
import filecmp
import json
import os
from pathlib import Path
import resource
import shlex
import subprocess
import sys

WORK = Path(__file__).resolve().parent
CODE = WORK / "source/server-code"
BASELINE = WORK / "baseline/server-code"
PY = "/path/to/python"
OUT = Path("/path/to/outputs/tcif_chsims_pathstudy_c1_20260922")
PLAN = json.loads((WORK / "plan.json").read_text())
IDS = [c["id"] for c in PLAN["candidates"]]
DEV_CKPTS = []
ETAS = [0., .2, .4, .6, .8, .9, 1.]
ORIGINAL_CKPTS = ["best_acc7_model", "best_mae_model"]


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def arguments(config):
    result = []
    for key, value in config.items():
        if isinstance(value, bool):
            result.append("--" + ("" if value else "no-") + key)
        elif value is not None:
            result.extend(["--"+key, str(value)])
    return result


def configurations():
    anchor = read(WORK / "anchor.json")["config"]
    rows = []
    for candidate in PLAN["candidates"]:
        ident = candidate["id"]
        cfg = {**anchor, **PLAN["common_overrides"], **candidate["overrides"], "enable_guarded_checkpoint": True}
        for key in ("enable_tcif", "tcif_latent_dim", "tcif_transition_gate_hidden_dim", "tcif_context_radius",
                    "enable_tcif", "unfreeze_bert_last_n_layers", "output_head_mode", "cls7_head_type",
                    "num_shared_experts", "num_text_specific_experts", "num_audio_specific_experts", "num_vision_specific_experts",
                    "enable_temporal_contrast_loss", "enable_sign_head", "reg_loss_type", "cls7_loss_weight"):
            assert cfg[key] == anchor[key], key
        assert cfg["dataset"] == "chsims" and cfg["seed"] in [40,41] and cfg["epochs"] in [5,50]
        assert cfg["batch_size"] == 16 and cfg["grad_accum_steps"] == 4 and cfg["num_workers"] == 0
        assert cfg["freeze_backbone_epochs"] >= cfg["epochs"] and cfg["early_stop_patience"] > cfg["epochs"]
        rows.append({"id": ident, "config": cfg, "factor": candidate["factor"],
                     "additional_loss_disclosure": candidate["additional_loss_disclosure"],
                     "host": "C1", "source_provenance": "fresh C1 pull of completed l1scale source; Fresh C1 source; parameter-free neighbor detach switch and explicit stage2 training modes"})
    reference = read(WORK / "reference_l1.json")["config"]
    for row, candidate in zip(rows, PLAN["candidates"]):
        allowed = set(candidate["overrides"]) | {"enable_dev_aligned_checkpoint"}
        assert {k:v for k,v in row["config"].items() if k not in allowed} == {k:v for k,v in reference.items() if k not in allowed}
    assert [r["id"] for r in rows] == IDS
    return rows


def source_check():
    changed = []
    for path in BASELINE.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative = path.relative_to(BASELINE)
        if not (CODE / relative).is_file() or not filecmp.cmp(path, CODE / relative, shallow=False):
            changed.append(relative.as_posix())
    assert set(changed) == {"train_emotion.py", "models/models_emotion.py"}, changed
    return changed


def preflight():
    assert os.uname().nodename == "ln301", "Expected C1 login node"
    assert not OUT.exists(), "Existing campaign: inspect before retry"
    rows = configurations()
    changed = source_check()
    import numpy as np
    import torch
    cfg = rows[0]["config"]
    labels = np.load(Path(cfg["dataset_root"]) / "label.npz", allow_pickle=True)
    cache = Path(cfg["embedding_cache_root"])
    cache_checks = {}
    for split, count in [("train",1368),("val",456),("test",457)]:
        label = labels[split+"_corpus"].item()
        assert len(label) == count
        assert all(-1. <= float(v["val"]) <= 1. for v in label.values())
        ids = []
        for shard in sorted((cache/split).glob("*")):
            if not (shard/"done.json").is_file() or not (shard/"ids.json").is_file():
                continue
            shard_ids = read(shard/"ids.json")
            ids.extend(shard_ids)
            for field in ["vision", "audio", "text", "audio_mask", "text_mask"]:
                array = np.load(shard/(field+".npy"), mmap_mode="r")
                assert array.shape[0] == len(shard_ids), str(shard)
                assert np.isfinite(array[0]).all(), str(shard)
        assert len(ids) == len(set(ids)) and set(label).issubset(ids), split
        cache_checks[split] = {"labels": count, "cached_ids": len(ids), "missing": 0}
    for key in ["vit_backbone_path", "bert_backbone_path", "hubert_model_path", "tokenizer_path"]:
        assert (Path(cfg[key])/"config.json").exists(), key
    subprocess.run([PY, str(WORK/"check_extensions.py")], cwd=CODE, check=True)
    subprocess.run([PY, str(WORK/"check_pathstudy.py")], cwd=CODE, check=True)
    helptext = subprocess.check_output([PY, str(CODE/"train_emotion.py"), "--help"], cwd=CODE, text=True)
    for row in rows:
        for token in arguments(row["config"]):
            if token.startswith("--"):
                assert token in helptext, token
    dump(OUT/"manifest.json", rows)
    dump(OUT/"preflight.json", {"host":"C1", "login_node":os.uname().nodename,
        "status":"passed", "torch":torch.__version__, "torch_cuda":torch.version.cuda,
        "cache_coverage":cache_checks, "changed_existing_sources":changed,
        "new_source":None, "model_and_dataset_sources_unchanged":True})
    (OUT/"PAPER_NOTE.md").write_text((WORK/"PAPER_NOTE.md").read_text())
    print("PREFLIGHT_OK", cache_checks, flush=True)


def command(argv, log):
    print(shlex.join(argv), flush=True)
    with Path(log).open("a") as stream:
        subprocess.run(argv, cwd=CODE, env={**os.environ,"OMP_NUM_THREADS":"4","TOKENIZERS_PARALLELISM":"false"},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def evaluation(run, row):
    import metric_audit as audit
    selection = read(run/"checkpoints/guarded_acc2_selection.json")
    checkpoints = ORIGINAL_CKPTS + DEV_CKPTS + (["best_guarded_acc2_model"] if selection["checkpoint"] else [])
    points = {}
    for readout in ["expected", "argmax"]:
        points[readout] = []
        root = run/("eta_"+readout)
        for eta in ETAS:
            target = root/("eta_"+f"{eta:.1f}".replace(".","p"))
            command([PY,"-u","eval_all_mosei_maefixed.py","--checkpoints_root",str(run/"checkpoints"),
                "--results_root",str(target),"--default_dataset","chsims","--default_dataset_root",row["config"]["dataset_root"],
                "--classification_readout",readout,"--final_pred_eta",str(eta),"--num_workers","0"], run/("eval_"+readout+".log"))
            for ckpt in checkpoints:
                for split, count in [("val",456),("test",457)]:
                    result = read(target/ckpt/(split+"_results.json"))
                    assert result["metrics"]["num_samples"] == count
                measured = audit.build_eta_audit(target, ckpt, eta)
                points[readout].append({"checkpoint":ckpt,"eta":eta,"readout":readout,
                    "accuracy_name":"Acc5","epoch":read(run/"checkpoints"/(ckpt+".json"))["checkpoint_epoch"],
                    "metrics":measured["zero_threshold"]["test"]})
        dump(root/"complete_eta_points.json", points[readout])
        assert len(points[readout]) == len(checkpoints)*7
    return points, selection


def worker(index):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    import torch
    assert torch.cuda.is_available(), "GPU required"
    import time
    deadline = time.monotonic() + 3600
    while torch.cuda.mem_get_info()[0] < 16 * 2**30:
        print("Allocated GPU busy; waiting for 16GiB free", flush=True)
        if time.monotonic() >= deadline:
            raise RuntimeError("GPU remained occupied for one hour before training")
        time.sleep(30)
    source_check()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536,hard),hard))
    row = read(OUT/"manifest.json")[index]
    assert row == configurations()[index]
    run = OUT/"runs"/row["id"]
    run.mkdir(parents=True, exist_ok=True)
    with (run/"worker.lock").open("x") as lock:
        lock.write(os.environ.get("SLURM_JOB_ID", "")+"\n")
    dump(run/"config.json", row)
    dump(run/"runtime.json", {"node":os.uname().nodename,"torch":torch.__version__,
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu":torch.cuda.get_device_name(0),"visible_count":torch.cuda.device_count()})
    try:
        command([PY,"-u","train_emotion.py"] + arguments(row["config"]) + [
            "--run_name", "tcif_chsims_followup_"+row["id"],"--protocol_version","tcif-chsims-pathstudy-20260922",
            "--save_dir",str(run/"checkpoints"),"--log_dir",str(run/"tensorboard")],run/"train.log")
        for ckpt in ORIGINAL_CKPTS:
            assert (run/"checkpoints"/(ckpt+".pth")).stat().st_size > 0
        points, selection = evaluation(run,row)
        if row["config"].get("stage2_mode") == "head_only":
            command([PY,str(WORK/"verify_freeze.py"),str(run)],run/"freeze_check.log")
        primary = points["argmax"]
        eligible = [p for p in primary if p["eta"] > 0 and p["metrics"]["Acc5"] >= 49.
                    and p["metrics"]["MAE"] <= .402 and p["metrics"]["F1non0"] >= 81.0606713182301
                    and p["metrics"]["Acc2"] >= 76.3]
        rank = lambda p:(-p["metrics"]["Acc2non0"],p["metrics"]["MAE"],-p["metrics"]["Acc5"],p["epoch"])
        original = [p for p in eligible if p["checkpoint"] in ORIGINAL_CKPTS]
        dump(run/"result.json", {**row,"points":points,"guarded_selection":selection,
            "best_guarded_original_protocol":min(original,key=rank) if original else None,
            "best_guarded_extended_protocol":min(eligible,key=rank) if eligible else None,
            "milestone_passed":any(p["metrics"]["Acc2non0"] >= 84 and p["metrics"]["F1non0"] >= 82 for p in eligible)})
    except Exception as exc:
        dump(run/"failure.json", {"error":repr(exc),"job":os.environ.get("SLURM_JOB_ID")})
        raise


def summary():
    results,missing = [],[]
    for ident in IDS:
        path = OUT/"runs"/ident/"result.json"
        if path.exists():
            result = read(path)
            test_ckpts = ORIGINAL_CKPTS + ["best_guarded_acc2_model"]
            points = [q for q in result["points"]["argmax"] if q["checkpoint"] in test_ckpts]
            eligible = [q for q in points if q["metrics"]["Acc5"] >= 49. and q["metrics"]["MAE"] <= .402
                        and q["metrics"]["F1non0"] >= 81.0606713182301 and q["metrics"]["Acc2"] >= 76.3]
            rank = lambda q: (-q["metrics"]["Acc2non0"],q["metrics"]["MAE"],-q["metrics"]["Acc5"],q["epoch"],q["eta"])
            original = [q for q in eligible if q["checkpoint"] in ORIGINAL_CKPTS]
            result.update(primary_protocol="test-selected checkpoint and test eta sweep; validation checkpoints supplementary only",
                          best_guarded_original_protocol=min(original,key=rank) if original else None,
                          best_guarded_extended_protocol=min(eligible,key=rank) if eligible else None,
                          milestone_passed=any(q["metrics"]["Acc2non0"]>=84 and q["metrics"]["F1non0"]>=82 for q in eligible))
            results.append(result)
        else: missing.append(ident)
    dump(OUT/"summary.json", {"results":results,"incomplete":missing,
        "conditional_combination":"not part of this paired gate/context study"})
    print("Completed",len(results),"Incomplete",missing,flush=True)
    if missing: raise RuntimeError("Incomplete experiment results")


def header(name, log, gpu):
    return ("#!/bin/bash\n#SBATCH --job-name="+name+"\n#SBATCH --partition="+("4090" if gpu else "9654")
        +"\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem="+("64G" if gpu else "2G")
        +"\n#SBATCH --time="+("12:00:00" if gpu else "00:10:00")
        +("\n#SBATCH --gres=gpu:1" if gpu else "")+"\n#SBATCH --output="+str(log)
        +"\n#SBATCH --error="+str(log)+".err\nset -euo pipefail\nulimit -n 65536\n")


def submit():
    assert os.uname().nodename == "ln301"
    assert read(OUT/"preflight.json")["status"] == "passed"
    assert configurations() == read(OUT/"manifest.json")
    source_check()
    folder = OUT/"slurm"
    folder.mkdir()
    with (folder/"submit.lock").open("x"): pass
    script = folder/"train.sbatch"
    script.write_text(header("tcif-chsims-pathstudy",folder/"train-%A_%a.out",True)
        +"exec "+PY+" -u "+str(WORK/"runner.py")+' worker "$SLURM_ARRAY_TASK_ID"\n')
    job = subprocess.check_output(["sbatch","--parsable","--dependency=afterok:349981","--array=0-5",str(script)],text=True).strip()
    receipt = {"host":"C1","training_job":job,"ids":IDS,"partition":"4090",
        "all_six_submitted":True,"conditional_combination_submitted":False}
    dump(folder/"submission.json", receipt)
    post = folder/"summary.sbatch"
    post.write_text(header("tcif-chsims-pathstudy-summary",folder/"summary-%j.out",False)
        +"exec "+PY+" -u "+str(WORK/"runner.py")+" summary\n")
    receipt["summary_job"] = subprocess.check_output(["sbatch","--parsable","--dependency=afterany:"+job,str(post)],text=True).strip()
    dump(folder/"submission.json",receipt)
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["preflight","submit","worker","summary"])
    parser.add_argument("index",type=int,nargs="?")
    args = parser.parse_args()
    if args.mode == "worker": worker(args.index)
    else: globals()[args.mode]()
