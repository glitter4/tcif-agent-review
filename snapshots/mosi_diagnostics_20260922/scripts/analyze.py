"""Offline reconstruction, development-only calibration and error decomposition."""
import csv
import json
from pathlib import Path
import re
import sys
import torch
from collect import ROOT,OUT,EVALS,SOURCES,CKPTS
sys.path.insert(0,str(ROOT/'server-code'))
from head_utils import compute_final_prediction,continuous_to_cls7_hard
from sv754_cross_dataset_eta_audit import binary_metrics
ETAS=[0,.2,.4,.6,.8,.9,1]
SCALES=[.9,.95,1,1.05,1.1]


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metrics(y,p):
    return dict(Acc7=float((continuous_to_cls7_hard(y)==continuous_to_cls7_hard(p)).double().mean()*100),
                MAE=float((p.double()-y.double()).abs().mean()),**binary_metrics(y.tolist(),p.tolist(),0))


def scores(rows,eta):
    reg=torch.tensor([r['y_reg_clipped'] for r in rows],dtype=torch.float32)
    logits=torch.tensor([r['cls7_logits'] for r in rows],dtype=torch.float32)
    return compute_final_prediction(reg,logits,eta=eta)


def group_stats(rows,y,p):
    truth_class=continuous_to_cls7_hard(y); pred_class=continuous_to_cls7_hard(p)
    masks={'zero':y==0,'weak_nonzero':(y!=0)&(y.abs()<.5),'strong':y.abs()>=2,
           'all':torch.ones(len(y),dtype=torch.bool)}
    conflict=[];valid=[]
    for r in rows:
        valid.append(any(r['neighbor_valid']))
        conflict.append(any(v and float(label)*r['raw_label']<0 for v,label in zip(r['neighbor_valid'],r['neighbor_labels'])))
    masks.update(conflict_neighbor=torch.tensor(conflict),nonconflict_neighbor=torch.tensor(valid)&~torch.tensor(conflict),
                 no_neighbor=~torch.tensor(valid))
    output={}
    for name,mask in masks.items():
        n=int(mask.sum())
        output[name]=dict(n=n)
        if n:
            output[name].update(MAE=float((p[mask]-y[mask]).abs().double().mean()),
                correct_class=int((truth_class[mask]==pred_class[mask]).sum()),
                cross_sign_errors=int(((p[mask]>=0)!=(y[mask]>=0)).sum()),
                same_sign_wrong_class=int((((p[mask]>=0)==(y[mask]>=0))&(truth_class[mask]!=pred_class[mask])).sum()),
                far_class_errors=int(((truth_class[mask]-pred_class[mask]).abs()>=2).sum()))
    return output


def main():
    assert (OUT/'export_complete.json').exists()
    audits=[];calibrations=[];decompositions=[]
    for name in SOURCES:
        for ck in CKPTS:
            datasets={s:[json.loads(l) for l in (OUT/'exports'/name/ck/(s+'.jsonl')).read_text().splitlines()] for s in ['val','test']}
            rebuilt={}
            for split,rows in datasets.items():
                assert len(rows)==(229 if split=='val' else 686)
                y=torch.tensor([r['raw_label'] for r in rows])
                rebuilt[split]={}
                for eta in ETAS:
                    p=scores(rows,eta)
                    oldpath=EVALS[name]/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/(split+'_details.csv')
                    old={r['id']:r for r in csv.DictReader(oldpath.open())}
                    assert set(old)=={r['sample_id'] for r in rows}
                    delta=max(abs(float(p[i])-float(old[r['sample_id']]['pred_value'])) for i,r in enumerate(rows))
                    label_delta=max(abs(r['raw_label']-float(old[r['sample_id']]['true_value'])) for r in rows)
                    assert delta<=1e-5 and label_delta<=1e-6,(name,ck,split,eta,delta,label_delta)
                    audits.append(dict(run=name,checkpoint=ck,split=split,eta=eta,max_abs_difference=delta,label_difference=label_delta))
                    rebuilt[split][eta]=metrics(y,p)
            # Choose eta before scale, using only validation data, including eta0 as diagnostic.
            eta=min(ETAS,key=lambda e:(rebuilt['val'][e]['MAE'],-rebuilt['val'][e]['Acc7'],e))
            val=datasets['val'];vy=torch.tensor([r['raw_label'] for r in val]);vp=scores(val,eta)
            candidates=[]
            for a in SCALES:
                pred=(vp*a).clamp(-3,3)
                assert torch.equal(vp>=0,pred>=0)
                candidates.append(dict(scale=a,metrics=metrics(vy,pred)))
            baseline=next(c for c in candidates if c['scale']==1)['metrics']
            chosen=min((c for c in candidates if c['metrics']['MAE']<=baseline['MAE']+1e-12),
                key=lambda c:(-c['metrics']['Acc7'],c['metrics']['MAE'],abs(c['scale']-1)))
            # Frozen val selection is stored before accessing test performance for calibration.
            decision=dict(run=name,checkpoint=ck,eta=eta,scale=chosen['scale'],validation_candidates=candidates)
            dump(OUT/'exports'/name/ck/'calibration_selection.json',decision)
            rows=datasets['test'];y=torch.tensor([r['raw_label'] for r in rows]);p=scores(rows,eta)
            scaled=(p*chosen['scale']).clamp(-3,3)
            assert torch.equal(p>=0,scaled>=0)
            before=metrics(y,p);after=metrics(y,scaled)
            for k in ['Acc2','Acc2non0','F1non0']:assert before[k]==after[k]
            calibrations.append(dict(decision,uncalibrated_test=before,calibrated_test=after,
                caveat='Parent checkpoint test-selected; not an independent holdout validation'))
            reg=scores(rows,0);cls=scores(rows,1);fused=scores(rows,.6)
            local_reg=torch.tensor([r['local_y_reg'] for r in rows]).clamp(-3,3)
            local_logits=torch.tensor([r['local_cls7_logits'] for r in rows])
            local=compute_final_prediction(local_reg,local_logits,eta=.6)
            er=reg-y;ec=cls-y
            decompositions.append(dict(run=name,checkpoint=ck,regression=metrics(y,reg),classification_expected=metrics(y,cls),
                fixed_eta06=metrics(y,fused),local_fixed_eta06=metrics(y,local),
                branch_opposite_error_sign_count=int((er*ec<0).sum()),
                regression_only_class_correct=int(((continuous_to_cls7_hard(reg)==continuous_to_cls7_hard(y))&(continuous_to_cls7_hard(cls)!=continuous_to_cls7_hard(y))).sum()),
                classification_only_class_correct=int(((continuous_to_cls7_hard(cls)==continuous_to_cls7_hard(y))&(continuous_to_cls7_hard(reg)!=continuous_to_cls7_hard(y))).sum()),
                groups=group_stats(rows,y,fused),
                tcif_means={branch:{k:sum(r[branch][k] for r in rows)/len(rows) for k in rows[0][branch]} for branch in ['reg_tcif','cls_tcif']},
                video_groups={g:group_stats([r for r in rows if r['group_id']==g],y[torch.tensor([r['group_id']==g for r in rows])],fused[torch.tensor([r['group_id']==g for r in rows])])['all'] for g in sorted({r['group_id'] for r in rows})}))
    dump(OUT/'reconstruction_audit.json',dict(status='passed',points=audits))
    dump(OUT/'calibration_results.json',calibrations)
    dump(OUT/'error_decomposition.json',decompositions)
    for name,source in SOURCES.items():
        text=(source/'train.log').read_text(errors='replace')
        selected=[line for line in text.splitlines() if line.startswith('Epoch ') and any(k in line for k in ['Parts:', 'SignedMetrics','FinalScore','LR','Loss:'])]
        (OUT/(name+'_epoch_records.txt')).write_text('\n'.join(selected)+'\n')
        dump(OUT/(name+'_log_coverage.json'),dict(record_lines=len(selected),
            lr_lines=sum('LR' in s for s in selected),gradient_norms_historical_available=False,
            component_gradient_angles_historical_available=False,notes='Missing fields are not reconstructed or inferred'))
    print(json.dumps([dict(run=r['run'],checkpoint=r['checkpoint'],eta=r['eta'],scale=r['scale'],test=r['calibrated_test']) for r in calibrations],indent=2))


if __name__=='__main__':main()
