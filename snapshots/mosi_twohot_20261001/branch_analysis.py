"""All decisions below use val exports only, at preregistered best-MAE/eta.4."""
import csv,json,sys
from pathlib import Path
import numpy as np
from metric_protocol import audit
ROOT=Path('/path/to/user/m4oe/tcif_mosi_twohot_20261001')
V1=Path('/path/to/user/m4oe/tcif_mosi_regularization_20261001/exports/NEIGHBOR_DROP')
V1_RUN=Path('/path/to/user/m4oe/tcif_mosi_regularization_20261001/runs/NEIGHBOR_DROP')
def read(p):return sorted([json.loads(s) for s in p.read_text().splitlines()],key=lambda r:r['position'])
def predictions(rows,eta):
    y=np.array([r['raw_label'] for r in rows],dtype=np.float32);r=np.array([r['y_reg_clipped'] for r in rows],dtype=np.float32)
    z=np.array([r['cls7_logits'] for r in rows],dtype=np.float32);p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
    cls=(p*np.arange(-3,4,dtype=np.float32)).sum(-1)
    return y,r,cls,np.float32(1-eta)*r+np.float32(eta)*cls
def stat(y,p):
    m=audit(y,p);m.pop('changed_positions');return m
def canonical(run,ck,split,eta,rows):
    def load(e):
        f=run/'eta_expected'/('eta_'+f'{e:.1f}'.replace('.','p'))/ck/(split+'_details.csv')
        data=list(csv.DictReader(f.open()))
        y=np.asarray([float(x['true_value']) for x in data],dtype=np.float32);p=np.asarray([float(x['pred_value']) for x in data],dtype=np.float32)
        assert np.array_equal(y,np.asarray([r['raw_label'] for r in rows],dtype=np.float32))
        return y,p
    y,p=load(eta);_,r=load(0.);_,c=load(1.)
    return y,r,c,p
def detailed(rows,reference,eta,new_outputs=None,reference_outputs=None):
    y,r,c,p=predictions(rows,eta) if new_outputs is None else new_outputs
    yr,rr,cr,pr=predictions(reference,eta) if reference_outputs is None else reference_outputs
    assert np.array_equal(y,yr)
    groups={}
    for n,mask in dict(weak_positive=(y>0)&(y<.5),weak_negative=(y<0)&(y>-.5),other_nonzero=np.abs(y)>=.5,zero=y==0,strong=np.abs(y)>=2).items():
        if not mask.any():groups[n]={'n':0};continue
        a=(pr[mask]>=0)==(y[mask]>=0);b=(p[mask]>=0)==(y[mask]>=0)
        ca=(cr[mask]>=0)==(y[mask]>=0);cb=(c[mask]>=0)==(y[mask]>=0)
        groups[n]=dict(n=int(mask.sum()),final_sign_errors=int((~b).sum()),reference_final_sign_errors=int((~a).sum()),
            repairs=int((~a&b).sum()),new_errors=int((a&~b).sum()),class_expectation_sign_errors=int((~cb).sum()),reference_class_expectation_sign_errors=int((~ca).sum()),
            final=stat(y[mask],p[mask]),reference_final=stat(y[mask],pr[mask]))
    return dict(regression=stat(y,r),classification_expectation=stat(y,c),final=stat(y,p),reference_classification_expectation=stat(y,cr),reference_final=stat(y,pr),groups=groups)
def main():
    ck='best_mae_model';eta=.4
    new=read(ROOT/'exports/V3_TCIF_TWOHOT'/ck/'val.jsonl');ref=read(V1/ck/'val.jsonl')
    d=detailed(new,ref,eta,canonical(ROOT/'runs/V3_TCIF_TWOHOT',ck,'val',eta,new),canonical(V1_RUN,ck,'val',eta,ref))
    weak=['weak_positive','weak_negative']
    conditions=dict(classification_MAE_improves=d['classification_expectation']['MAE']<d['reference_classification_expectation']['MAE'],
        final_MAE_improves=d['final']['MAE']<d['reference_final']['MAE'],Acc7_guard=d['final']['Acc7_legacy']>=d['reference_final']['Acc7_legacy']-.3,
        each_weak_sign_error_not_worse=all(d['groups'][g]['final_sign_errors']<=d['groups'][g]['reference_final_sign_errors'] for g in weak),
        combined_weak_sign_error_improves=sum(d['groups'][g]['final_sign_errors'] for g in weak)<sum(d['groups'][g]['reference_final_sign_errors'] for g in weak))
    decision=dict(split='val',checkpoint=ck,eta=eta,conditions=conditions,twohot_expansion=all(conditions.values()),details=d,
        note='decision saved before test branch analysis; checkpoint itself remains test-selected')
    (ROOT/'twohot_expansion_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    rows=[]
    for name in ['V2_STANDARD_DROP','V3_TCIF_TWOHOT']:
        for ck in ['best_acc7_model','best_mae_model']:
            for split in ['val','test']:
                new=read(ROOT/'exports'/name/ck/(split+'.jsonl'));ref=read(V1/ck/(split+'.jsonl'))
                for eta in [0,.2,.4,.6,.8,.9,1]:rows.append(dict(run=name,checkpoint=ck,split=split,eta=eta,
                    **detailed(new,ref,eta,canonical(ROOT/'runs'/name,ck,split,eta,new),canonical(V1_RUN,ck,split,eta,ref))))
    (ROOT/'branch_analysis.json').write_text(json.dumps(dict(rows=rows,reference='V1 NEIGHBOR_DROP'),indent=2)+'\n')
    print('TWOHOT_VALIDATION_GATE',conditions,all(conditions.values()))
if __name__=='__main__':main()
