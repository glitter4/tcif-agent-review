"""Rebuild MOSI eta metrics from published numeric outputs, standard library only."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/mosi_diagnostics_20260922'
ETAS=[0,.2,.4,.6,.8,.9,1]


def cls(y):
    y=max(-3,min(3,y))
    return math.floor(y+.5) if y>=0 else -math.floor(-y+.5)


def score(row,eta):
    z=row['cls7_logits'];top=max(z);q=[math.exp(x-top) for x in z]
    expected=sum((i-3)*v for i,v in enumerate(q))/sum(q)
    return (1-eta)*row['y_reg_clipped']+eta*expected


def metrics(rows,pred):
    pairs=[(r['raw_label'],p) for r,p in zip(rows,pred)]
    nz=[(y,p) for y,p in pairs if abs(y)>1e-12]
    f1=[]
    for c in [False,True]:
        tp=sum((y>=0)==c and (p>=0)==c for y,p in nz)
        fp=sum((y>=0)!=c and (p>=0)==c for y,p in nz)
        fn=sum((y>=0)==c and (p>=0)!=c for y,p in nz)
        f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
    return dict(Acc7=100*sum(cls(y)==cls(p) for y,p in pairs)/len(pairs),
        MAE=sum(abs(y-p) for y,p in pairs)/len(pairs),
        Acc2non0=100*sum((y>=0)==(p>=0) for y,p in nz)/len(nz),F1non0=50*sum(f1))


def main():
    reference={r['id']:r for r in json.loads((ROOT/'results/mosi_recent_full_sweeps.json').read_text(encoding='utf-8'))}
    decisions=json.loads((DATA/'calibration_results.json').read_text(encoding='utf-8'))
    count=0
    for run in ['D1','R1']:
        for ck in ['best_acc7_model','best_mae_model']:
            files=DATA/'samples'/run/ck
            rows={s:[json.loads(l) for l in (files/(s+'.jsonl')).read_text().splitlines()] for s in ['val','test']}
            for split,n in [('val',229),('test',686)]:
                ids={r['sample_id'] for r in rows[split]}
                assert len(rows[split])==len(ids)==n
                for r in rows[split]:
                    assert r['split']==split and len(r['cls7_logits'])==7
                    assert set(r['neighbor_ids'])<=ids
                    assert r['sample_id'].startswith(split+'_')
                    assert all(math.isfinite(x) for x in r['cls7_logits'])
                count+=n
            expected=reference['D1' if run=='D1' else 'R1_router_temp015']['points']
            for eta in ETAS:
                current=metrics(rows['test'],[score(r,eta) for r in rows['test']])
                point=next(x for x in expected if x['checkpoint']==ck and x['eta']==eta)['metrics']
                assert all(abs(current[k]-point[k])<1e-5 for k in current),(run,ck,eta,current)
            selected=next(x for x in decisions if x['run']==run and x['checkpoint']==ck)
            eta=min(ETAS,key=lambda e:(metrics(rows['val'],[score(r,e) for r in rows['val']])['MAE'],
                -metrics(rows['val'],[score(r,e) for r in rows['val']])['Acc7'],e))
            assert eta==selected['eta']
            candidates=selected['validation_candidates']
            baseline=next(x for x in candidates if x['scale']==1)['metrics']['MAE']
            best=min((x for x in candidates if x['metrics']['MAE']<=baseline+1e-12),
                key=lambda x:(-x['metrics']['Acc7'],x['metrics']['MAE'],abs(x['scale']-1)))
            assert best['scale']==selected['scale']==1
            for r in rows['test']:
                before=score(r,eta);after=max(-3,min(3,before*best['scale']))
                assert (before>=0)==(after>=0)
    assert count==3660
    print('PASS: 3660 numeric rows, stable ID joins, 28 test eta points reconstructed, validation eta choices and sign invariance verified.')


if __name__=='__main__':main()
