"""Audit archived metrics and source joins without model weights or ML packages."""
import json
import math
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mosi_baseline_reproduction_20261003'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8),(a,b)

summary=read(BASE/'summary.json')
comparison=read(BASE/'comparison.json')
verified=read(BASE/'checkpoint_verification.json')
assert len(verified)==18 and all(v['passed'] for v in verified)
assert len({r['run'] for r in summary['rows']})==6
for r in summary['rows']:
    assert r['done'] and r['n']==686
    run=BASE/'runs'/r['run']
    assert (run/'DONE.json').exists()
    epochs=[json.loads(s) for s in (run/'epochs.jsonl').read_text().splitlines()]
    assert [x['epoch'] for x in epochs]==list(range(1,len(epochs)+1))
    assert len(epochs)==r['last_epoch']
    point=epochs[r['epoch']-1]['test']
    for k in ['acc7_numpy','acc7_project','mae','acc2_non0','f1_weighted_non0','f1_macro_non0']:
        close(r[k],point[k])
    if r['rule']=='test_acc7_numpy':close(r['acc7_numpy'],max(x['test']['acc7_numpy'] for x in epochs))
    if r['rule']=='test_acc7_project':close(r['acc7_project'],max(x['test']['acc7_project'] for x in epochs))
    if r['rule']=='test_mae':close(r['mae'],min(x['test']['mae'] for x in epochs))
    if r['rule'].startswith('test_'):
        assert any(v['run']==r['run'] and v['rule']==r['rule'] and v['epoch']==r['epoch'] for v in verified)
mapping=dict(acc7_numpy='Acc7_nearest_even',acc7_project='Acc7',mae='MAE',
             acc2_non0='Acc2non0',f1_weighted_non0='F1_weighted_non0',f1_macro_non0='F1_macro_non0')
for r in comparison['tcif_mixed_host_rows']:
    d=read(ROOT/r['source'])
    assert d['training_complete'] and len(d['points'])==14
    for ck in ['best_acc7_model','best_mae_model']:
        assert sorted(p['eta'] for p in d['points'] if p['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
    matches=[p for p in d['points'] if p['checkpoint']==r['checkpoint'] and p['eta']==r['eta']]
    assert len(matches)==1
    assert d['checkpoint_epochs'][r['checkpoint']]==r['epoch']
    for k,v in mapping.items():close(r[k],matches[0]['metrics'][v])
    close(r['acc7_numpy'],max(p['metrics']['Acc7_nearest_even'] for p in d['points']))
for r in comparison['representatives']:
    d=read(ROOT/r['source'])
    p=next(p for p in d['points'] if p['checkpoint']==r['checkpoint'] and p['eta']==r['eta'])
    for k in mapping.values():close(r['metrics'][k],p['metrics'][k])
for g in comparison['groups']:
    rows=(comparison['tcif_mixed_host_rows'] if g['model']=='TCIF mixed-host' else
          [r for r in summary['rows'] if r['model']==g['model'] and r['rule']=='test_acc7_numpy'])
    assert len(rows)==g['n']==3
    for k,s in g['statistics'].items():
        values=[r[k] for r in rows]
        close(s['mean'],statistics.mean(values));close(s['std_population'],statistics.pstdev(values));close(s['std_sample'],statistics.stdev(values))
print('PASS: 6 completed seeds, all per-epoch selections, 18 checkpoint receipts; TCIF source joins, full eta grids, means and both SD definitions.')
