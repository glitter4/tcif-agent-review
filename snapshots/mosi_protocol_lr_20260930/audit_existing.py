"""No inference or reselection: fixed saved checkpoint/eta, local full-precision branch exports."""
import json
from pathlib import Path
from metric_protocol import audit,reconstruct

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    base=ROOT/'analysis/tcif_mosi_diagnostics_20260922'
    sources={n:base/'exports'/n for n in ['D1','R1']}
    sources.update({n:base/'final_studies/runs'/n/'exports' for n in ['DETACH','HEAD10','FULL10']})
    recent=read(ROOT.parent/'m4oe-github-rdrop-20260929/results/mosi_recent_full_sweeps.json')
    studies=read(ROOT.parent/'m4oe-github-rdrop-20260929/results/mosi_diagnostics_20260922/study_results.json')
    def collect(x):
        if isinstance(x,list):return [r for a in x for r in collect(a)]
        if isinstance(x,dict):
            if 'id' in x and 'points' in x:return [x]
            return [r for a in x.values() for r in collect(a)]
        return []
    refs={r['id']:r for r in collect(recent)+collect(studies)}
    rows=[]
    for name,folder in sources.items():
        for ck in ['best_acc7_model','best_mae_model']:
            for split,n in [('val',229),('test',686)]:
                records=[json.loads(s) for s in (folder/ck/(split+'.jsonl')).read_text().splitlines()]
                assert len(records)==n and len({r['sample_id'] for r in records})==n
                for eta in ETAS:
                    result=audit([r['raw_label'] for r in records],reconstruct(records,eta))
                    if split=='test':
                        ref=next(p for p in refs['R1_router_temp015' if name=='R1' else name]['points'] if p['checkpoint']==ck and p['eta']==eta)['metrics']
                        assert abs(result['Acc7_legacy']-ref['Acc7'])<1e-9,(name,ck,eta)
                        assert abs(result['MAE']-ref['MAE'])<1e-5,(name,ck,eta)
                        for a,b in [('macro_F1','F1_macro_non0'),('weighted_F1','F1_weighted_non0'),('Acc2','Acc2non0')]:
                            assert abs(result['binary']['nonzero'][a]-ref[b])<1e-8
                    write(HERE/'results/changes'/name/ck/(split+'_eta'+str(eta)+'.json'),result['changed_positions'])
                    result.pop('changed_positions')
                    rows.append(dict(run=name,checkpoint=ck,split=split,eta=eta,**result))
    write(HERE/'results/local_metric_audit.json',dict(status='partial_remote_blocked',selection='existing test-selected checkpoints; no reselection',
        prediction_precision='float32 reconstruction from full precision saved regression/logits; historical test Acc7 and binary metrics exactly reproduced; MAE within1e-5; direct remote prediction CSV audit still required',
        pending=['all transferred checkpoints including C0 and R-Drop','MOSEI','other MOSI sweeps without local branch exports'],rows=rows))
    print('PASS: 5 runs x2 checkpoints x7 eta x2 splits =',len(rows),'fixed points; original test metrics reproduced')
    for r in rows:
        if r['run']=='D1' and r['checkpoint']=='best_acc7_model' and r['split']=='test' and r['eta'] in [.6,.8,1]:print(r)

if __name__=='__main__':main()
