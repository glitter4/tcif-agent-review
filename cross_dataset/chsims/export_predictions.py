"""Read-only export using the exact existing evaluator; no extra forward passes."""
import json
import os
from pathlib import Path
import sys
import time

WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'source/server-code'))
import torch
import eval_all_mosei_maefixed as ev

ROOT=Path(os.environ.get('CHSIMS_EXPORT_SOURCE','/path/to/outputs/tcif_chsims_gatectx_c1_20260922'))
OUT=Path(os.environ.get('CHSIMS_EXPORT_OUTPUT','/path/to/outputs/tcif_chsims_diagnostics_c1_20260922'))
RUNS=os.environ.get('CHSIMS_EXPORT_RUNS','B0_S40,C003_S40,B0_S41,C003_S41').split(',')
run=RUNS[int(sys.argv[1])]
dest=OUT/'exports'/run
dest.mkdir(parents=True,exist_ok=True)
original_discover=ev.discover_models
models=[]
def discover(root):
    global models
    models=[x for x in original_discover(root) if x['model_stem'] in ['best_acc7_model','best_mae_model']]
    return models
ev.discover_models=discover
original_evaluate=ev.evaluate_split
model_index=-1

def evaluate(model,loader,device,**kwargs):
    global model_index
    ds=loader.dataset
    if ds.split=='val': model_index+=1
    stem=models[model_index]['model_stem']
    rows=[]
    state={}
    lookup={str(v):i for i,v in enumerate(ds.ids)}
    class Loader:
        dataset=ds
        def __iter__(self):
            for batch in loader:
                state['batch']=batch
                yield batch
    def tcif_hook(name):
        def hook(module,inputs,out):
            local=out['local_feature']
            state[name]={
                'relative_residual':((out['posterior_feature']-local).norm(dim=-1)/(local.norm(dim=-1)+1e-8)).detach().cpu().tolist(),
                'local_feature_norm':local.norm(dim=-1).detach().cpu().tolist(),
                'local_variance':out['local_variance'].mean(dim=-1).detach().cpu().tolist()}
        return hook
    def capture(module,inputs,out):
        if not isinstance(out,dict) or 'tcif' not in out.get('extras',{}): return
        b=state['batch']; payload=out['extras']['tcif']
        raw=out['y_reg'].detach().cpu().tolist()
        logits=out['cls7_logits'].detach().cpu().tolist()
        payload={k:v.detach().cpu().tolist() for k,v in payload.items() if torch.is_tensor(v)}
        for j,sample_id in enumerate(b['id']):
            sample_id=str(sample_id); idx=lookup[sample_id]
            neighbors=ds.temporal_context_index[idx]
            meta=ds.temporal_metadata_by_id[sample_id]
            row={'sample_id':sample_id,'split':ds.split,'group_id':meta['temporal_group_id'],'source_group':meta['base_id'],'temporal_pos':meta['temporal_pos'],
                 'raw_label':float(b['raw_valence'][j]),'y_reg_raw':raw[j],'y_reg_clipped':max(-1.,min(1.,raw[j])),
                 'logits':logits[j],'centers':model.cls7_centers.detach().cpu().tolist(),
                 'neighbors':[{'sample_id':ds.ids[n['index']],'valid':n['valid'],'relative_pos':n['relative_pos'],
                               'raw_label':float(ds.labels[ds.ids[n['index']]]['val'])} for n in neighbors],
                 'tcif':{k:v[j] for k,v in payload.items()}}
            for name in ('regression','ordinal'):
                row['tcif'][name+'_relative_residual']=state[name]['relative_residual'][j]
                row['tcif'][name+'_local_feature_norm']=state[name]['local_feature_norm'][j]
                row['tcif'][name+'_local_variance']=state[name]['local_variance'][j]
            rows.append(row)
    handles=[model.tcif_regression.register_forward_hook(tcif_hook('regression')),
             model.tcif_ordinal.register_forward_hook(tcif_hook('ordinal')),model.register_forward_hook(capture)]
    try: result=original_evaluate(model,Loader(),device,**kwargs)
    finally:
        for h in handles:h.remove()
    assert len(rows)==len(ds) and len({r['sample_id'] for r in rows})==len(rows)
    with (dest/(stem+'_'+ds.split+'.jsonl')).open('w') as stream:
        for row in rows:stream.write(json.dumps(row)+'\n')
    texts=[ds.text_by_id.get(x,'') for x in ds.ids]
    lengths=[len(x) for x in ds.tokenizer(texts,truncation=False,padding=False)['input_ids']]
    audit={'split':ds.split,'samples':len(ds),'max_length':ds.max_length,'truncated_count':sum(n>ds.max_length for n in lengths),
           'empty_texts':sum(not t for t in texts),'tokenizer_class':type(ds.tokenizer).__name__,
           'examples':[{'sample_id':ds.ids[i],'text':texts[i],'tokens':ds.tokenizer.convert_ids_to_tokens(ds.encoded_text[i][0].tolist())} for i in range(2)]}
    (dest/('input_audit_'+ds.split+'.json')).write_text(json.dumps(audit,indent=2))
    return result
ev.evaluate_split=evaluate
if __name__=='__main__':
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    deadline=time.monotonic()+3600
    while torch.cuda.mem_get_info()[0]<16*2**30:
        if time.monotonic()>deadline:raise RuntimeError('GPU occupied')
        time.sleep(30)
    sys.argv=['eval','--checkpoints_root',str(ROOT/'runs'/run/'checkpoints'),'--results_root',str(dest/'verification'),
              '--default_dataset','chsims','--classification_readout','argmax','--final_pred_eta','.8','--num_workers','0']
    ev.main()
    assert len(list(dest.glob('best_*_*.jsonl')))==4
    print('EXPORT_COMPLETE',run,flush=True)
