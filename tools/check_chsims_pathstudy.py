"""Validate completed metrics independently from supplementary export availability."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/chsims_pathstudy_20260922'
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    summary=load(ROOT/'summary.json');status=load(ROOT/'artifact_status.json')
    assert not summary['incomplete'] and len(summary['results'])==6
    total=0
    for run in summary['results']:
        expected={'best_acc7_model','best_mae_model'}
        if run['guarded_selection']['checkpoint']:expected.add('best_guarded_acc2_model')
        assert set(run['points'])=={'expected','argmax'}
        for mode,points in run['points'].items():
            assert {p['checkpoint'] for p in points}==expected
            for ck in expected:assert sorted(p['eta'] for p in points if p['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
            for p in points:
                m=p['metrics'];assert p['readout']==mode
                assert m['num_samples_all']==457 and m['num_samples_non0']==388
                assert all(math.isfinite(m[k]) for k in ['Acc5','MAE','Acc2non0','F1non0'])
            total+=len(points)
    assert total==182==status['complete_eta_points']
    samples=0;files=0
    for path in (ROOT/'predictions').glob('*/*.jsonl'):
        rows=[json.loads(x) for x in path.read_text().splitlines()]
        assert len(rows)==(456 if '_val.' in path.name else 457)
        assert len({r['sample_id'] for r in rows})==len(rows)
        assert all(len(r['logits'])==5 for r in rows)
        samples+=len(rows);files+=1
    assert files==10 and samples==4565
    assert len(status['raw_output_export']['pending_runs'])==4
    for seed in [40,41]:
        checks=load(ROOT/f'freeze_verification_s{seed}.json')
        assert len(checks)==4
        assert all(r['max_regression_delta']==0 and set(r['changed_parameters'])=={'head_cls7.weight','head_cls7.bias'} for r in checks)
    checks=load(ROOT/'gpu_cpu_recovery_comparison.json')
    assert len(checks)==18 and all(r['max_abs_delta']<=2e-4 and r['binary_changes']==0 for r in checks)
    print('PASS: six runs / 182 points; 10 raw exports / 4565 records; freeze and CPU/GPU checks; four supplementary exports pending.')
if __name__=='__main__':main()
