"""Validate this review snapshot using only the Python standard library."""
import ast
import csv
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ETAS = [0, .2, .4, .6, .8, .9, 1]
CHECKPOINTS = {'best_acc7_model', 'best_mae_model'}


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def check_points(points, samples, nonzero):
    assert len(points) == 14
    assert {p['checkpoint'] for p in points} == CHECKPOINTS
    for ck in CHECKPOINTS:
        assert sorted(p['eta'] for p in points if p['checkpoint'] == ck) == ETAS
    for point in points:
        metrics = point['metrics']
        assert metrics['num_samples_all'] == samples
        assert metrics['num_samples_non0'] == nonzero
        assert metrics['F1non0'] == metrics['F1_macro_non0']
        assert all(math.isfinite(metrics[k]) for k in ('MAE', 'Acc2non0', 'F1non0'))


def main():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and
                   '__pycache__' not in p.parts and '.git' not in p.parts)
    python_files = [p for p in files if p.suffix == '.py']
    # Parsing avoids executing training code or importing optional ML dependencies.
    for path in python_files:
        ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for path in files:
        if path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8'))
        if path.suffix == '.md':
            for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
                if '://' in target or target.startswith('#'):
                    continue
                target = target.split('#', 1)[0]
                assert (path.parent / target).exists(), (path, target)
    code = ROOT / 'server-code'
    local_modules = {'head_utils', 'losses', 'output_layout', 'tcif_ablation_config',
                     'train_emotion', 'models', 'datasets', 'test_expert_load_output',
                     'test_signed_reg_cls7_head', 'test_tcif'}
    for path in python_files:
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                if name.split('.')[0] in local_modules:
                    module_path = code.joinpath(*name.split('.'))
                    assert module_path.with_suffix('.py').is_file() or (module_path/'__init__.py').is_file(), name
    base = load('configs/tcif_paper_single_model.json')
    assert base['dataset'] == 'cmumosei'
    assert base['checkpoint_selection_split'] == 'test'
    assert base['checkpoint_metrics'] == 'dual'
    assert base['unfreeze_bert_last_n_layers'] == 2 and base['use_text_cache'] is False
    rows = load('results/mosei_ablation_full_sweeps.json')
    assert len(rows) == 10
    variants = {'full', 'standard_context', 'equal_weight', 'no_gate', 'shared_filter'}
    assert {(r['variant'], r['checkpoint']) for r in rows} == {(v,c) for v in variants for c in CHECKPOINTS}
    for row in rows:
        assert sorted(p['eta'] for p in row['sweep']) == ETAS
        assert row['selected_eta'] in row['sweep'] and row['swept_best'] in row['sweep']
        assert all(p['samples'] == 4659 and p['nonzero_samples'] == 3634 for p in row['sweep'])
    with (ROOT/'results/mosei_ablation_full_sweeps.csv').open(encoding='utf-8', newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 70
    for row in rows:
        for point in row['sweep']:
            match = [r for r in csv_rows if r['run'] == row['variant'] and
                     r['checkpoint'] == row['checkpoint'] and float(r['eta']) == point['eta']]
            assert len(match) == 1
            for key in ('acc7', 'mae', 'acc2non0', 'f1non0'):
                assert float(match[0][key]) == point[key]
    mosi = load('results/mosi_recent_full_sweeps.json')
    assert len(mosi) == 3
    for run in mosi:
        check_points(run['points'], 686, 656)
        assert run['evaluation_complete'] is True
        assert all(p['readout'] == 'expected' and p['T'] == 1 for p in run['points'])
    chsims = load('results/chsims_seed_sweeps.json')['results']
    assert len(chsims) == 4
    for run in chsims:
        assert set(run['points']) == {'expected', 'argmax'}
        for mode, points in run['points'].items():
            check_points(points, 457, 388)
            assert all(p['readout'] == mode for p in points)
    print(f'PASS: {len(python_files)} Python files parsed; local imports and Markdown links resolve.')
    print('PASS: MOSEI 70 + MOSI 42 + CH-SIMS 112 = 224 complete historical sweep points; CSV matches JSON.')
    print('NOTE: four MOSEI sensitivity rows are fixed-eta summaries, not complete sweeps.')


if __name__ == '__main__':
    main()
