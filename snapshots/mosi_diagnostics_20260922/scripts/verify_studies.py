"""Verify exported study branches against sweeps and the fixed D1 parent."""
import csv
import json
from pathlib import Path
import sys
import torch
from collect import OUT,CKPTS
from analyze import scores,metrics,group_stats,ETAS,dump


def main():
    for name in sys.argv[1:]:
        source=OUT/'training/runs'/name
        assert all((source/'exports'/ck/'complete.json').exists() for ck in CKPTS)
        audits=[];summary=[]
        for ck in CKPTS:
            for split in ['val','test']:
                rows=[json.loads(l) for l in (source/'exports'/ck/(split+'.jsonl')).read_text().splitlines()]
                truth=torch.tensor([r['raw_label'] for r in rows])
                parent={r['sample_id']:r for r in [json.loads(l) for l in (OUT/'exports/D1/best_acc7_model'/(split+'.jsonl')).read_text().splitlines()]}
                assert {r['sample_id'] for r in rows}==set(parent)
                max_reg=max(abs(r['y_reg_raw']-parent[r['sample_id']]['y_reg_raw']) for r in rows)
                if name=='HEAD10':assert max_reg<=1e-6,max_reg
                for eta in ETAS:
                    pred=scores(rows,eta)
                    path=source/'eta_expected'/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/(split+'_details.csv')
                    old={r['id']:float(r['pred_value']) for r in csv.DictReader(path.open())}
                    assert set(old)==set(parent)
                    delta=max(abs(float(pred[i])-old[r['sample_id']]) for i,r in enumerate(rows))
                    assert delta<=1e-5,(name,ck,split,eta,delta)
                    audits.append(dict(checkpoint=ck,split=split,eta=eta,max_prediction_difference=delta))
                fused=scores(rows,.6)
                summary.append(dict(checkpoint=ck,split=split,max_regression_difference_from_parent=max_reg,
                    fixed_eta06_metrics=metrics(truth,fused),groups=group_stats(rows,truth,fused),
                    tcif_means={branch:{key:sum(r[branch][key] for r in rows)/len(rows) for key in rows[0][branch]} for branch in ['reg_tcif','cls_tcif']}))
        gradients=[json.loads(l)['preclip_norm'] for l in (source/'checkpoints/gradient_norms.jsonl').read_text().splitlines()]
        expected=json.loads((source/'config.json').read_text())['epochs']*41
        assert len(gradients)==expected,(name,len(gradients),expected)
        report=dict(status='passed',name=name,eta_points=audits,branch_summary=summary,
            gradient_norm_summary=dict(optimizer_steps=len(gradients),mean=sum(gradients)/len(gradients),
                max=max(gradients),fraction_over_clip1=sum(x>1 for x in gradients)/len(gradients)))
        dump(source/'output_verification.json',report)
        print('VERIFIED',name,'head regression invariant' if name=='HEAD10' else '',flush=True)


if __name__=='__main__':main()
