"""Isolated interventions; disabled by default, unchanged parameter shapes."""
import json
import os
from pathlib import Path
import torch
_validation_records={}


def detach_context(module,args):
    return (args[0],args[1].detach(),*args[2:])


def configure(model,groups,args):
    mode=os.environ.get('STUDY_MODE','baseline')
    if mode=='detach':
        model.tcif_regression.register_forward_pre_hook(detach_context)
        model.tcif_ordinal.register_forward_pre_hook(detach_context)
    elif mode in ('head_only','full_continue'):
        state=torch.load(os.environ['STUDY_INIT'],map_location='cpu',weights_only=False)
        model.load_state_dict(state.get('model_state_dict',state.get('state_dict',state)),strict=True)
        ids={id(p) for p in model.head_cls7.parameters()}
        if mode=='head_only':
            for p in model.parameters():p.requires_grad_(id(p) in ids)
            groups=[]
        else:
            groups=[dict(g,params=[p for p in g['params'] if id(p) not in ids]) for g in groups]
            groups=[g for g in groups if g['params']]
        groups.append(dict(name='classifier_only',params=list(model.head_cls7.parameters()),lr=2.25e-5))
    record=dict(mode=mode,initialization=os.environ.get('STUDY_INIT'),
        fresh_optimizer=True,groups=[dict(name=g['name'],lr=g['lr'],parameters=sum(p.numel() for p in g['params'])) for g in groups])
    Path(args.save_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.save_dir)/'study_protocol.json').write_text(json.dumps(record,indent=2))
    print('STUDY_PROTOCOL',record,flush=True)
    return groups


def enforce_epoch(model):
    if os.environ.get('STUDY_MODE')=='head_only':
        model.eval()
        model.head_cls7.train()
        ids={id(p) for p in model.head_cls7.parameters()}
        for p in model.parameters():p.requires_grad_(id(p) in ids)
        assert {id(p) for p in model.parameters() if p.requires_grad}==ids


def log_gradient(epoch,step,norm,args):
    with (Path(args.save_dir)/'gradient_norms.jsonl').open('a') as f:
        f.write(json.dumps(dict(epoch=epoch+1,step=step+1,preclip_norm=float(norm)))+'\n')


def component_probe(model,losses,args):
    named=[(n,p) for n,p in model.named_parameters() if n.startswith('shared_specific_layers.') and n.endswith('.phi') and p.requires_grad]
    if not named:
        record=dict(status='not_applicable',reason='shared router parameters frozen')
    else:
        vectors={}
        for name,loss in losses.items():
            grads=torch.autograd.grad(loss,[p for _,p in named],retain_graph=True,allow_unused=True)
            vectors[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten() for (_,p),g in zip(named,grads)])
        norms={n:float(v.norm()) for n,v in vectors.items()}
        cos={a+'__'+b:float(torch.nn.functional.cosine_similarity(vectors[a],vectors[b],dim=0)) for a in vectors for b in vectors if a<b}
        record=dict(status='measured',parameter_names=[n for n,_ in named],scope='first seeded training batch; router-phi subspace only',norms=norms,cosines=cos)
    (Path(args.save_dir)/'component_gradient_probe.json').write_text(json.dumps(record,indent=2))


def save_validation(model,args,epoch,val_metrics,test_metrics,updater,summary_writer):
    import copy
    selected=copy.copy(args)
    selected.checkpoint_selection_split='val'
    path=Path(args.save_dir).parent/'validation_checkpoints'
    path.mkdir(exist_ok=True)
    selected.save_dir=str(path)
    updater(model,str(path),epoch,val_metrics,test_metrics,_validation_records,selected)
    summary_writer(str(path),_validation_records,selected)
