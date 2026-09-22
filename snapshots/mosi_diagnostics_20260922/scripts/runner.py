"""Bounded MOSI optimization-budget experiment; unchanged TCIF source code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

WORK = Path(__file__).resolve().parent
CODE = WORK.parent / 'server-code'
OUT = Path('/path/to/user/m4oe/tcif_mosi_budget_20260916')
PY = '/path/to/user/envs/m4oe-lab5090/bin/python'
ETAS = [0., .2, .4, .6, .8, .9, 1.]
CKPTS = ['best_acc7_model', 'best_mae_model']
NEAR = {'Acc7': 44.2, 'MAE': .790, 'Acc2non0': 83.3, 'F1non0': 83.0}
MID = {'Acc7': 45.5, 'MAE': .770, 'Acc2non0': 84.5, 'F1non0': 84.0}
# Exact same-point C1 reference; NOT a paired same-environment control.
REFERENCE = {'Acc7': 42.12827988338192, 'MAE': .79990910374459,
             'Acc2non0': 80.9451219512195, 'F1non0': 80.51918741254423}


def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def base():
    return dict(
        dataset='cmumosi', dataset_root='/path/to/user/datasets/MER-unibench/cmumosi-process',
        embedding_cache_root='/path/to/user/m4oe/embedding_cache/cmumosi_faceonly_nf4_vctx02_fp16_20260518',
        signed_class_count=7, label_min=-3., label_max=3., epochs=100,
        batch_size=16, grad_accum_steps=4, lr=7.5e-6, router_lr=2e-4,
        router_temperature=.1, weight_decay=.01, scheduler='cosine',
        lr_warmup_epochs=1, lr_warmup_start_factor=0., lr_min_factor=.03,
        clip_grad_norm=1., output_head_mode='signed_reg_cls7', cls7_head_type='flat',
        reg_loss_type='smooth_l1', cls7_loss_type='soft_ce', cls7_loss_weight=.75,
        cls7_soft_tau=.3, cls7_class_weight_mode='none', num_shared_experts=1,
        num_text_specific_experts=2, num_audio_specific_experts=2,
        num_vision_specific_experts=2, expert_mlp_ratio=4., enable_text_shared_experts=True,
        num_temporal_contrast_experts=0, enable_temporal_contrast_experts=False,
        enable_temporal_contrast_loss=False, temporal_contrast_weight=0.,
        temporal_contrast_temperature=.07, temporal_decay_tau=1., temporal_positive_radius=1.,
        temporal_weak_positive_radius=4., temporal_min_positive_weight=.2, temporal_kernel='legacy_exp',
        alpha=0., dropout=.5, num_frames=4, vit_context_ratio=.2,
        vit_context_lambda_mode='attention', vit_context_lambda_init='0.8,0.1,0.1',
        vit_context_attention_dim=64, freeze_backbone_epochs=100,
        unfreeze_bert_last_n_layers=12, bert_last_layer_lr_ratio=.5, use_text_cache=False,
        final_pred_eta=.85, enable_sign_head=False, final_pred_sign_beta=0.,
        checkpoint_selection_split='test', checkpoint_metrics='dual',
        evaluate_test_each_epoch=True, early_stop_metric='val_mae', early_stop_patience=101,
        local_files_only=True, num_workers=0, vision_backbone_type='vit',
        vit_backbone_path='/path/to/user/models/vit-base-patch16-224-in21k',
        bert_backbone_path='/path/to/user/models/AI-ModelScope_roberta-base',
        tokenizer_path='/path/to/user/models/AI-ModelScope_roberta-base',
        hubert_model_path='/path/to/user/models/hubert-base-ls960', seed=123,
        enable_tcif=True, tcif_latent_dim=96, tcif_context_radius=1,
        tcif_context_mode='neighbors', tcif_output_mode='posterior', tcif_context_aux_weight=0.,
        tcif_enable_transition_gate=True, tcif_transition_gate_hidden_dim=64,
        tcif_transition_gate_init_bias=2., tcif_transition_gate_loss_weight=.05,
        tcif_transition_gate_tau=.75, tcif_transition_gate_conflict_target=0.,
        tcif_transition_gate_lr=6e-5,
    )


def config(epochs=100, accum=4, **changes):
    cfg = dict(base(), epochs=epochs, grad_accum_steps=accum,
               freeze_backbone_epochs=epochs, early_stop_patience=epochs + 1)
    cfg.update(changes)
    validate(cfg)
    return cfg


def validate(cfg):
    allowed = {'epochs', 'grad_accum_steps', 'freeze_backbone_epochs',
               'early_stop_patience', 'lr_warmup_epochs', 'router_lr'}
    assert set(cfg) == set(base()), 'Unexpected config keys'
    for key, value in base().items():
        if key not in allowed:
            assert cfg[key] == value, (key, cfg[key], value)
    assert cfg['epochs'] in (100, 200)
    assert cfg['grad_accum_steps'] in (2, 4)
    assert cfg['freeze_backbone_epochs'] >= cfg['epochs']
    assert cfg['early_stop_patience'] > cfg['epochs']
    assert cfg['lr_warmup_epochs'] in (1, 5)
    assert cfg['router_lr'] in (2e-4, 5e-5)


def argv(cfg):
    out = []
    for key, value in cfg.items():
        if isinstance(value, bool):
            out.append('--' + ('' if value else 'no-') + key)
        else:
            out += ['--' + key, str(value)]
    return out


def deficit(metrics, target=NEAR):
    assert all(math.isfinite(metrics[k]) for k in target)
    gaps = [max(0., target['Acc7'] - metrics['Acc7']) / .5,
            max(0., metrics['MAE'] - target['MAE']) / .01,
            max(0., target['Acc2non0'] - metrics['Acc2non0']) / .5,
            max(0., target['F1non0'] - metrics['F1non0']) / .5]
    return max(gaps), sum(gaps)


def improved(metrics):
    d, total = deficit(metrics)
    ref_d, ref_total = deficit(REFERENCE)
    return d == 0 or (d <= ref_d - .5 and total < ref_total) or (d <= ref_d and total <= ref_total - 1.)


def command(args, logfile, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    with Path(logfile).open('a') as log:
        log.write(shlex.join(args) + '\n')
        log.flush()
        subprocess.run(args, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate(run, cfg, gpu):
    eroot = run / 'eta_expected'
    for eta in ETAS:
        target = eroot / ('eta_' + f'{eta:.1f}'.replace('.', 'p'))
        command([PY, '-u', 'eval_all_mosei_maefixed.py', '--checkpoints_root', str(run / 'checkpoints'),
                 '--results_root', str(target), '--default_dataset', 'cmumosi',
                 '--default_dataset_root', cfg['dataset_root'], '--classification_readout', 'expected',
                 '--final_pred_eta', str(eta), '--num_workers', '0'], run / 'eval.log', gpu)
        for ckpt in CKPTS:
            for split, count in [('val', 229), ('test', 686)]:
                assert read(target / ckpt / (split + '_results.json'))['metrics']['num_samples'] == count
    summary = eroot / 'eta_sweep_summary.json'
    command([PY, str(WORK / 'sv754_cross_dataset_eta_audit.py'), '--eval_root', str(eroot),
             '--output', str(summary), '--readout_mode', 'expected'], run / 'audit.log', gpu)
    data = read(summary)
    points = []
    for ckpt in CKPTS:
        info = data['per_checkpoint'][ckpt]
        assert info['eta_sweep_complete'] and info['eta_available'] == ETAS
        for p in info['per_eta'].values():
            points.append(dict(checkpoint=ckpt, eta=p['eta'], readout='expected', T=1,
                               metrics=p['zero_threshold']['test']))
    return points


def run_one(name, cfg, gpu):
    validate(cfg)
    run = OUT / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    # Exclusive marker prevents silent reruns or overwriting partial experiments.
    with (run / 'started.json').open('x') as handle:
        json.dump({'pid': os.getpid(), 'gpu': gpu, 'config': cfg}, handle, indent=2)
    dump(run / 'config.json', cfg)
    print('START', name, 'GPU', gpu, flush=True)
    try:
        command([PY, '-u', 'train_emotion.py'] + argv(cfg) + [
            '--run_name', 'tcif_mosi_budget_' + name, '--protocol_version', 'tcif-mosi-budget-20260916',
            '--save_dir', str(run / 'checkpoints'), '--log_dir', str(run / 'tensorboard')], run / 'train.log', gpu)
        for ckpt in CKPTS:
            assert (run / 'checkpoints' / (ckpt + '.pth')).stat().st_size > 0
        points = evaluate(run, cfg, gpu)
        best = min((p for p in points if p['eta'] > 0),
                   key=lambda p: (deficit(p['metrics']), p['metrics']['MAE'], -p['metrics']['Acc7']))
        result = dict(id=name, config=cfg, points=points, best_joint=best,
                      near_pass=deficit(best['metrics']) == (0., 0.),
                      mid_pass=any(deficit(p['metrics'], MID) == (0., 0.) for p in points if p['eta'] > 0),
                      improves_cross_environment_reference=improved(best['metrics']))
        dump(run / 'result.json', result)
        print('DONE', name, best, flush=True)
        return result
    except Exception as exc:
        dump(run / 'failure.json', {'error': repr(exc), 'gpu': gpu})
        raise


def preflight():
    import numpy as np
    import torch
    assert socket.gethostname() == 'lab-host'
    assert str(CODE) == '/path/to/user/workspaces/m4oe-tcif-mosi-budget-20260916/server-code'
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    assert not (OUT / 'controller.lock').exists(), 'Already launched'
    cfg = base()
    for key in ['dataset_root', 'embedding_cache_root', 'vit_backbone_path', 'bert_backbone_path', 'hubert_model_path']:
        assert Path(cfg[key]).is_dir(), key
    labels = np.load(Path(cfg['dataset_root']) / 'label.npz', allow_pickle=True)
    counts = {}
    for split, expected in [('train', 1284), ('val', 229), ('test', 686)]:
        rows = labels[split + '_corpus'].item()
        counts[split] = len(rows)
        assert len(rows) == expected, (split, len(rows))
        ids = set()
        for path in (Path(cfg['embedding_cache_root']) / split).glob('*/ids.json'):
            if (path.parent / 'done.json').exists():
                ids.update(str(x) for x in read(path))
        assert set(map(str, rows)).issubset(ids), 'Missing cached samples: ' + split
    helptext = subprocess.check_output([PY, str(CODE / 'train_emotion.py'), '--help'], text=True)
    for token in argv(cfg):
        if token.startswith('--'):
            assert token in helptext, token
    assert 'TemporalContextInnovationFilter' in (CODE / 'models' / 'models_emotion.py').read_text()
    assert 'class TemporalContextInnovationFilter' in (CODE / 'models' / 'tcif.py').read_text()
    report = dict(host=socket.gethostname(), code=str(CODE), torch=torch.__version__, counts=counts,
                  gpu_names=[torch.cuda.get_device_name(i) for i in range(2)], status='passed',
                  no_same_machine_control='Explicitly requested by user', anchor='C1 loss_loss_0_1',
                  reference=REFERENCE, reference_comparison='Cross-environment, not causal paired evidence',
                  max_runs=5, initial={'A_e200_b64': config(200, 4), 'B_e100_b32': config(100, 2)})
    dump(OUT / 'preflight.json', report)
    print(json.dumps(report, indent=2))


def launch():
    import resource
    assert read(OUT / 'preflight.json')['status'] == 'passed'
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    with (OUT / 'controller.lock').open('x') as handle:
        handle.write(str(os.getpid()) + '\n')
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(run_one, 'A_e200_b64', config(200, 4), 0)
            b = pool.submit(run_one, 'B_e100_b32', config(100, 2), 1)
            first = [a.result(), b.result()]
        dump(OUT / 'results_AB.json', first)
        if not any(r['improves_cross_environment_reference'] for r in first):
            dump(OUT / 'complete.json', {'status': 'stopped_without_budget_gain', 'after': 'AB'})
            return
        if any(r['mid_pass'] for r in first):
            dump(OUT / 'complete.json', {'status': 'single_seed_mid_goal_reached', 'after': 'AB'})
            return
        c = run_one('C_e200_b32', config(200, 2), 0)
        pool_results = first + [c]
        if c['mid_pass']:
            dump(OUT / 'complete.json', {'status': 'single_seed_mid_goal_reached', 'after': 'C'})
            return
        winner = min(pool_results, key=lambda r: deficit(r['best_joint']['metrics']))
        chosen = winner['config']
        dump(OUT / 'followup_selection.json', {'winner': winner['id'], 'reason': 'minimum joint deficit',
                                              'same_machine_control': False})
        w = dict(chosen, lr_warmup_epochs=5)
        r = dict(chosen, router_lr=5e-5)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fw = pool.submit(run_one, 'W_warmup5', w, 0)
            fr = pool.submit(run_one, 'R_router5e5', r, 1)
            pool_results += [fw.result(), fr.result()]
        dump(OUT / 'all_results.json', pool_results)
        dump(OUT / 'complete.json', {'status': 'bounded_search_complete', 'count': len(pool_results),
                                    'near_pass': any(x['near_pass'] for x in pool_results),
                                    'mid_pass': any(x['mid_pass'] for x in pool_results)})
    except Exception as exc:
        dump(OUT / 'attention.json', {'status': 'halted_on_failure', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['preflight', 'launch'])
    args = parser.parse_args()
    {'preflight': preflight, 'launch': launch}[args.mode]()
