"""Local-to-posterior and same-point C0 repairs from position-only branch exports."""
import json,sys
from pathlib import Path
import numpy as np
from metric_protocol import classes,binary_metrics
ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def score(rows,eta,local=False):
    r=np.asarray([x['local_y_reg' if local else 'y_reg_clipped'] for x in rows],dtype=np.float32).clip(-3,3)
    z=np.asarray([x['local_cls7_logits' if local else 'cls7_logits'] for x in rows],dtype=np.float32)
    p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    e=(p*np.arange(-3,4,dtype=np.float32)).sum(-1)
    return np.float32(1-eta)*r+np.float32(eta)*e
def stats(y,p,base,mask):
    y=y[mask];p=p[mask];base=base[mask];n=len(y)
    if not n:return dict(n=0)
    a=classes(base,'legacy_away')==classes(y,'legacy_away');b=classes(p,'legacy_away')==classes(y,'legacy_away')
    sa=(base>=0)==(y>=0);sb=(p>=0)==(y>=0)
    return dict(n=n,MAE=float(np.abs(p-y).mean()),Acc7=float(b.mean()*100),binary=binary_metrics(y,p),
        class_repairs=int((~a&b).sum()),class_new_errors=int((a&~b).sum()),sign_repairs=int((~sa&sb).sum()),sign_new_errors=int((sa&~sb).sum()),
        sign_repairs_still_wrong_class=int((~sa&sb&~b).sum()),same_sign_wrong_class=int((sb&~b).sum()))
def main(root):
    root=Path(root);report=[]
    for ck in ['best_acc7_model','best_mae_model']:
        for split in ['val','test']:
            new=read(root/'exports/NEIGHBOR_DROP'/ck/(split+'.jsonl'));ref=read(root/'exports/C0'/ck/(split+'.jsonl'))
            new=sorted(new,key=lambda r:r['position']);ref=sorted(ref,key=lambda r:r['position'])
            assert [r['position'] for r in new]==[r['position'] for r in ref]==list(range(len(new)))
            y=np.asarray([r['raw_label'] for r in new],dtype=np.float32);assert np.array_equal(y,np.asarray([r['raw_label'] for r in ref],dtype=np.float32))
            has=np.array([any(r['neighbor_valid']) for r in new]);conflict=np.array([any(v and label*r['raw_label']<0 for v,label in zip(r['neighbor_valid'],r['neighbor_labels'])) for r in new])
            same=np.array([r['raw_label']!=0 and any(v for v in r['neighbor_valid']) and all((not v) or label*r['raw_label']>0 for v,label in zip(r['neighbor_valid'],r['neighbor_labels'])) for r in new])
            groups=dict(all=np.ones(len(y),dtype=bool),no_neighbor=~has,opposite_nonzero=conflict,same_polarity_all_valid_nonzero=same,
                valid_without_opposite_including_neutral=has&~conflict,weak_nonzero=(np.abs(y)>0)&(np.abs(y)<.5),strong=np.abs(y)>=2)
            for eta in ETAS:
                pred=score(new,eta);local=score(new,eta,True);baseline=score(ref,eta);ref_local=score(ref,eta,True)
                report.append(dict(checkpoint=ck,split=split,eta=eta,groups={g:dict(local_to_posterior=stats(y,pred,local,mask),C0_to_drop=stats(y,pred,baseline,mask),
                    C0_local_to_posterior=stats(y,baseline,ref_local,mask)) for g,mask in groups.items()}))
    (root/'neighbor_group_analysis.json').write_text(json.dumps(dict(protocol='same checkpoint type/eta, legacy bins; val diagnostics not selection; overlapping groups explicitly named',rows=report),indent=2))
    print('GROUP_ANALYSIS_COMPLETE',len(report))
if __name__=='__main__':main(sys.argv[1])
