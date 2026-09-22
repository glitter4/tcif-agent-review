"""Read-only forward hooks around the exact execution evaluator; no text export."""
import json
import os
from pathlib import Path
import sys
import torch
CODE=Path(os.environ['STUDY_CODE'])
sys.path.insert(0,str(CODE))
import eval_all_mosei_maefixed as ev

original=ev.evaluate_split


def evaluate(model,loader,device,**kwargs):
    split='val' if len(loader.dataset)==229 else 'test'
    assert len(loader.dataset) in (229,686)
    dataset=loader.dataset
    id_index={str(s):i for i,s in enumerate(dataset.ids)}
    batch_box={}; tcif_box={}; rows=[]
    class Loader:
        def __iter__(self):
            for batch in loader:
                batch_box['batch']=batch
                yield batch
    def tcif_hook(name):
        def hook(module,args,output):
            local=output['local_feature']
            post=output['posterior_feature']
            tcif_box[name]={
                'feature_relative_change':((post-local).norm(dim=-1)/(local.norm(dim=-1)+1e-8)).detach().cpu(),
                'local_variance_mean':output['local_variance'].mean(dim=-1).detach().cpu(),
                'prior_variance_mean':output['prior_variance'].mean(dim=-1).detach().cpu(),
                'gate':output['continuation_gate'].detach().cpu(),
                'innovation_norm':output['innovation'].norm(dim=-1).detach().cpu(),
            }
        return hook
    def final_hook(module,args,output):
        batch=batch_box['batch']; payload=output['extras']['tcif']
        y=output['y_reg'].detach().cpu()
        logits=output['cls7_logits'].detach().cpu()
        for i,sid in enumerate(batch['id']):
            sid=str(sid); ix=id_index[sid]
            contexts=dataset.temporal_context_index[ix]
            row=dict(sample_id=sid,split=split,group_id=str(batch['base_id'][i]),
                raw_label=float(batch['raw_valence'][i]),y_reg_raw=float(y[i]),
                y_reg_clipped=float(y[i].clamp(-3,3)),cls7_logits=logits[i].tolist(),
                local_y_reg=float(payload['local_y_reg'][i].detach().cpu()),
                local_cls7_logits=payload['local_cls7_logits'][i].detach().cpu().tolist(),
                prior_y_reg=float(payload['context_y_reg'][i].detach().cpu()),
                prior_cls7_logits=payload['context_cls7_logits'][i].detach().cpu().tolist(),
                neighbor_ids=[str(dataset.ids[x['index']]) for x in contexts],
                neighbor_valid=batch['tcif_context_valid_mask'][i].tolist(),
                neighbor_relative_pos=batch['tcif_context_relative_pos'][i].tolist(),
                neighbor_labels=batch['tcif_context_raw_valence'][i].tolist(),
                token_count=int(batch['attention_mask'][i].sum()))
            for name,values in tcif_box.items():
                row[name]={key:float(value[i]) for key,value in values.items()}
            rows.append(row)
    handles=[model.tcif_regression.register_forward_hook(tcif_hook('reg_tcif')),
             model.tcif_ordinal.register_forward_hook(tcif_hook('cls_tcif')),
             model.register_forward_hook(final_hook)]
    try: result=original(model,Loader(),device,**kwargs)
    finally:
        for h in handles:h.remove()
    assert len(rows)==len(dataset) and len({r['sample_id'] for r in rows})==len(rows)
    out=Path(os.environ['STUDY_EXPORT']);out.mkdir(parents=True,exist_ok=True)
    with (out/(split+'.jsonl')).open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    ev.evaluate_split=evaluate
    ev.main()
