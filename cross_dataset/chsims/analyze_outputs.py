"""Reconstruct every existing eta point, then finite sign-preserving calibration."""
import csv
import json
from pathlib import Path
import sys

WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'source/server-code'))
import torch
from head_utils import get_cls7_centers,compute_final_prediction,continuous_to_cls7_hard
from chsims_followup_support import binary_metrics

ROOT=Path('/path/to/outputs/tcif_chsims_gatectx_c1_20260922')
OUT=Path('/path/to/outputs/tcif_chsims_diagnostics_c1_20260922')
ETAS=[0.,.2,.4,.6,.8,.9,1.]
SCALES=[.9,.95,1.,1.05,1.1]
centers=get_cls7_centers(num_classes=5,label_min=-1,label_max=1)

def metrics(p,y):
    nz=y!=0
    acc,f1=binary_metrics(p,y);accnz,f1nz=binary_metrics(p[nz],y[nz])
    return {'Acc5':100.*float((continuous_to_cls7_hard(p,centers)==continuous_to_cls7_hard(y,centers)).double().mean()),
            'MAE':float((p.double()-y.double()).abs().mean()),'Acc2':acc,'F1':f1,'Acc2non0':accnz,'F1non0':f1nz}

def buckets(p,y,rows):
    pc=continuous_to_cls7_hard(p,centers);yc=continuous_to_cls7_hard(y,centers)
    nz=y!=0
    masks={'zero':~nz,'weak_nonzero':nz & (y.abs()<=.2),'nonzero':nz,
           'same_sign_wrong_bin':((p>=0)==(y>=0))&(pc!=yc),'cross_sign':(p>=0)!=(y>=0),
           'distant_wrong_bin':(pc-yc).abs()>=2}
    for category in ['conflicting_neighbors','same_sign_neighbors','no_neighbors']:
        flags=[]
        for row in rows:
            neighbors=[v for v in row['neighbors'] if v['valid']]
            conflict=any(row['raw_label']*v['raw_label']<0 for v in neighbors)
            flags.append((bool(neighbors) and conflict) if category=='conflicting_neighbors' else
                         (bool(neighbors) and not conflict) if category=='same_sign_neighbors' else not neighbors)
        masks[category]=torch.tensor(flags)
    return {k:{'n':int(m.sum()),'MAE':float((p[m].double()-y[m].double()).abs().mean()) if m.any() else None,
               'Acc5':100.*float((pc[m]==yc[m]).double().mean()) if m.any() else None,
               'binary_correct':int(((p>=0)==(y>=0))[m].sum())} for k,m in masks.items()}

def main():
    torch.set_num_threads(2)
    checks=[];grid=[];chosen=[];diagnostics=[]
    for run in ['B0_S40','C003_S40','B0_S41','C003_S41']:
        for ck in ['best_acc7_model','best_mae_model']:
            data={split:[json.loads(line) for line in (OUT/'exports'/run/(ck+'_'+split+'.jsonl')).read_text().splitlines()] for split in ['val','test']}
            for split,n in [('val',456),('test',457)]:
                assert len(data[split])==n
            for mode in ['expected','argmax']:
                for eta in ETAS:
                    predictions={};labels={};measured={}
                    for split,rows in data.items():
                        y=torch.tensor([r['raw_label'] for r in rows],dtype=torch.float32)
                        reg=torch.tensor([r['y_reg_clipped'] for r in rows],dtype=torch.float32)
                        logits=torch.tensor([r['logits'] for r in rows],dtype=torch.float32)
                        p=compute_final_prediction(reg,logits,eta=eta,centers=centers,classification_readout=mode)
                        saved=ROOT/'runs'/run/('eta_'+mode)/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/(split+'_details.csv')
                        with saved.open() as f:old={r['id']:r for r in csv.DictReader(f)}
                        assert set(old)=={r['sample_id'] for r in rows}
                        oldp=torch.tensor([float(old[r['sample_id']]['pred_value']) for r in rows],dtype=torch.float32)
                        oldy=torch.tensor([float(old[r['sample_id']]['true_value']) for r in rows],dtype=torch.float32)
                        assert torch.equal(y,oldy)
                        delta=float((oldp-p).abs().max())
                        assert delta<=2e-5,(run,ck,split,mode,eta,delta)
                        binary_changed=int(((oldp>=0)!=(p>=0)).sum())
                        bins_changed=int((continuous_to_cls7_hard(oldp,centers)!=continuous_to_cls7_hard(p,centers)).sum())
                        assert binary_changed==bins_changed==0,(run,ck,split,mode,eta,'boundary mismatch')
                        checks.append({'run':run,'checkpoint':ck,'split':split,'readout':mode,'eta':eta,'max_abs_prediction_delta':delta,'binary_changed':binary_changed,'bins_changed':bins_changed})
                        predictions[split]=p;labels[split]=y;measured[split]={}
                        for scale in SCALES:
                            scaled=(p*scale).clamp(-1,1)
                            assert torch.equal(scaled>=0,p>=0)
                            met=metrics(scaled,y)
                            base=metrics(p,y)
                            for key in ['Acc2','Acc2non0','F1','F1non0']:assert met[key]==base[key]
                            measured[split][scale]=met
                            grid.append({'run':run,'checkpoint':ck,'split':split,'readout':mode,'eta':eta,'scale':scale,'metrics':met})
                    eligible=[a for a in SCALES if measured['val'][a]['MAE']<=measured['val'][1.]['MAE']+1e-12]
                    a=min(eligible,key=lambda a:(-measured['val'][a]['Acc5'],measured['val'][a]['MAE'],abs(a-1)))
                    chosen.append({'run':run,'checkpoint':ck,'readout':mode,'eta':eta,'scale_selected_on_val':a,
                                   'val':measured['val'][a],'test':measured['test'][a],'test_unscaled':measured['test'][1.]})
                    if mode=='argmax' and eta==.8:
                        rows=data['test'];p=predictions['test'];y=labels['test']
                        reg=torch.tensor([r['y_reg_clipped'] for r in rows]);logits=torch.tensor([r['logits'] for r in rows])
                        cp=compute_final_prediction(reg,logits,eta=1.,centers=centers,classification_readout='expected')
                        re=(reg-y).double();ce=(cp-y).double()
                        diagnostics.append({'run':run,'checkpoint':ck,'buckets':buckets(p,y,rows),'regression':metrics(reg,y),'classification_expected':metrics(cp,y),
                            'error_product_mean':float((re*ce).mean()),'opposite_residual_count':int((re*ce<0).sum()),
                            'both_wrong_sign':int((((reg>=0)!=(y>=0))&((cp>=0)!=(y>=0))).sum()),
                            'regression_relative_residual_mean':sum(r['tcif']['regression_relative_residual'] for r in rows)/len(rows),
                            'ordinal_relative_residual_mean':sum(r['tcif']['ordinal_relative_residual'] for r in rows)/len(rows)})
    report={'reconstruction_checks':checks,'calibration_grid':grid,'val_selected_scale_test_eta_candidates':chosen,'diagnostics':diagnostics,
            'protocol':'Scale selected on val per checkpoint/readout/eta with no val-MAE worsening; final eta remains test-selected. Not an independent test estimate.',
            'training_not_repeated':True}
    (OUT/'diagnostics.json').write_text(json.dumps(report,indent=2))
    print('RECONSTRUCTION_PASSED',len(checks),'CALIBRATION_POINTS',len(grid),flush=True)
    for row in chosen:
        if row['readout']=='argmax' and row['eta'] in [.8,.9] and row['run']=='C003_S41':print(row,flush=True)

if __name__=='__main__':main()
