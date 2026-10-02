"""Post-training diagnosis against saved Lab V1 outputs; explicitly cross-environment."""
import json,sys
from pathlib import Path
CODE=Path('/path/to/user/workspaces/m4oe-tcif-weak-reweight-20261002')
sys.path.insert(0,str(CODE/'.codex-jobs/twohot'))
from branch_analysis import read,predictions,canonical,detailed
from metric_protocol import audit
ROOT=Path('/path/to/user/m4oe/c1_runs/tcif_weak_reweight_20261002')
run=ROOT/'runs/REWEIGHT1P5';ref=json.loads((ROOT/'reference/V1_result.json').read_text())
records=[];reconstruction=[]
for ck in ['best_acc7_model','best_mae_model']:
    for split in ['val','test']:
        parent=read(ROOT/'reference/V1_exports'/ck/(split+'.jsonl'))
        current=read(ROOT/'exports/REWEIGHT1P5'/ck/(split+'.jsonl'))
        for eta in [0,.2,.4,.6,.8,.9,1]:
            values=predictions(parent,eta);y,_,_,p=values
            rebuilt=audit(y,p);old=next(p for p in ref['points' if split=='test' else 'validation_points'] if p['checkpoint']==ck and p['eta']==eta)['metrics']
            assert abs(rebuilt['Acc7_legacy']-old['Acc7'])<1e-8
            assert abs(rebuilt['MAE']-old['MAE'])<1e-5
            assert abs(rebuilt['binary']['nonzero']['Acc2']-old['Acc2non0'])<1e-8
            reconstruction.append(dict(checkpoint=ck,split=split,eta=eta,MAE_difference=rebuilt['MAE']-old['MAE'],discrete_metrics_equal=True))
            records.append(dict(checkpoint=ck,split=split,eta=eta,**detailed(current,parent,eta,canonical(run,ck,split,eta,current),values)))
(ROOT/'reweight_analysis.json').write_text(json.dumps(dict(rows=records,reference_reconstruction=reconstruction,
    limitation='C1 new trial vs Lab5090 parent; host/backend and weighting differ. Parent predictions reconstructed from full-precision branch exports, discrete metrics equal and MAE within1e-5 verified.'),indent=2,allow_nan=False)+'\n')
print('REWEIGHT_ANALYSIS_COMPLETE',len(records))
