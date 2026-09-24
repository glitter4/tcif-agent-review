"""Validate G/R/S sweeps, metric provenance, and per-update mechanism logs."""
import collections
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/mosi_grs_20260924'
ETAS=[0,.2,.4,.6,.8,.9,1]
CK=['best_acc7_model','best_mae_model']


def main():
    rows=json.loads((DATA/'results.json').read_text(encoding='utf-8'))
    assert {r['id'] for r in rows}=={'G','R','S'}
    summary=json.loads((DATA/'audit_summary.json').read_text(encoding='utf-8'))
    ref=json.loads((DATA/'c1_D1_reference.json').read_text(encoding='utf-8'))
    for r in rows:
        name=r['id'];assert r['training_complete'] and r['config']==dict(ref['config'],grs_mode=name)
        points=r['points'];assert len(points)==14
        metadata=json.loads((DATA/'mechanism'/name/'checkpoint_metadata.json').read_text())
        for ck in CK:
            assert sorted(p['eta'] for p in points if p['checkpoint']==ck)==ETAS
            assert metadata[ck]['grs_mode']==name
            assert metadata[ck]['checkpoint_epoch']==r['checkpoint_epochs'][ck]
        for p in points:
            m=p['metrics'];assert p['readout']=='expected' and p['T']==1
            assert m['num_samples_all']==686 and m['num_samples_non0']==656
            assert m['F1non0']==m['F1_macro_non0']
            assert p['A_star']==max(m['Acc2'],m['Acc2non0'])==m[p['A_source']]
            assert p['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[p['F_source']]
            assert p['F_source'] in ('F1_macro_all','F1_macro_non0')
            assert all(math.isfinite(m[k]) for k in ('Acc7','MAE','Acc2','Acc2non0','F1_macro_all','F1_macro_non0'))
        stage=any(p['eta']>0 and p['metrics']['Acc7']>=46.1 and p['A_star']>=85 and p['F_star']>=85 and p['metrics']['MAE']<=.730 for p in points)
        final=any(p['eta']>0 and p['metrics']['Acc7']>48.5 and p['A_star']>86.95 and p['F_star']>86.94 and p['metrics']['MAE']<.697 for p in points)
        assert r['stage_pass']==stage and r['final_pass']==final
        steps=[json.loads(l) for l in (DATA/'mechanism'/name/'grs_steps.jsonl').read_text().splitlines()]
        assert len(steps)==8200
        assert collections.Counter(s['epoch'] for s in steps)=={i:41 for i in range(1,201)}
        for epoch in range(1,201):
            assert [s['micro_step'] for s in steps if s['epoch']==epoch]==list(range(2,81,2))+[81]
        for s in steps:
            assert len(s['microbatches'])==(1 if s['micro_step']==81 else 2)
            if name=='G':
                expected=s['reg_cls_dot']<0 and s['cls_norm_squared']>1e-12
                assert s['projected']==expected
                value=-s['reg_cls_dot']/(s['cls_norm_squared']+1e-12) if expected else 0
                assert math.isclose(value,s['correction_coefficient'],rel_tol=1e-9,abs_tol=1e-12)
            else:
                for m in s['microbatches']:
                    assert m['extra_raw']>=0 and math.isfinite(m['extra_raw'])
                    assert math.isclose(m['extra_raw']*(.2 if name=='R' else .05),m['extra_weighted'],rel_tol=2e-6,abs_tol=1e-8)
                    if name=='S' and m['weak_positive']+m['weak_negative']==0:assert m['extra_raw']==0
        if name=='G':assert sum(s['projected'] for s in steps)==summary[name]['mechanism']['projected_steps']==4779
        for p in summary[name]['picks'].values():assert p in points and p['eta']>0
    print('PASS: 3x200 epochs,42 eta points,24600 optimizer-update records, A*/macro-F* provenance, G accumulation projection and R/S activation.')


if __name__=='__main__':main()
