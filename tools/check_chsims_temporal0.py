"""Verify the paired CH-SIMS temporal-zero result after the campaign completes."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'results/chsims_temporal0_20260923'
ETAS=[0.,.2,.4,.6,.8,.9,1.]
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    summary=load(ROOT/'summary.json');manifest=load(ROOT/'manifest.json')
    assert summary['incomplete']==[] and len(summary['results'])==len(manifest)==4
    names={f'{name}_S{seed}' for seed in (40,41) for name in ('B0','T0')}
    assert {r['id'] for r in summary['results']}=={r['id'] for r in manifest}==names
    by_id={r['id']:r['config'] for r in manifest}
    for seed in (40,41):
        baseline=by_id[f'B0_S{seed}'];treatment=by_id[f'T0_S{seed}']
        assert baseline['temporal_contrast_weight']==.0065 and treatment['temporal_contrast_weight']==0.
        assert baseline['enable_temporal_contrast_loss'] is treatment['enable_temporal_contrast_loss'] is True
        assert {k:v for k,v in baseline.items() if k!='temporal_contrast_weight'}=={k:v for k,v in treatment.items() if k!='temporal_contrast_weight'}
        assert baseline['epochs']==treatment['epochs']==50
    total=0
    for result in summary['results']:
        name=result['id'];points=result['points']
        assert set(points)=={'argmax','expected'}
        expected={'best_acc7_model','best_mae_model'}
        if result['guarded_selection']['checkpoint']:expected.add('best_guarded_acc2_model')
        for mode,rows in points.items():
            assert {r['checkpoint'] for r in rows}==expected
            for ck in expected:assert sorted(r['eta'] for r in rows if r['checkpoint']==ck)==ETAS
            assert len(rows)==7*len(expected)
            for row in rows:
                m=row['metrics'];assert row['readout']==mode and row['accuracy_name']=='Acc5'
                assert m['num_samples_all']==457 and m['num_samples_non0']==388
                assert all(math.isfinite(m[k]) for k in ('Acc5','MAE','Acc2non0','F1non0'))
            total+=len(rows)
        run=ROOT/'runs'/name
        assert len((run/'argmax_epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines())==50
        runtime=load(run/'runtime.json')
        assert runtime['visible_count']==1 and 'RTX 5090' in runtime['gpu']
        if name.startswith('T0_'):assert not (run/'temporal_followup_history.jsonl').exists()
    assert total>=112 and total%14==0
    print(f'PASS: 4 Lab5090 CH-SIMS runs, {total} complete eta points, paired config differs only temporal weight.')
if __name__=='__main__':main()
