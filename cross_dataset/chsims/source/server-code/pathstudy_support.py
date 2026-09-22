import json
from pathlib import Path
import torch

def stage_mode(model,args):
    if args.stage2_mode=='head_only':
        for name,p in model.named_parameters():p.requires_grad_(name.startswith('head_cls7.'))
        model.eval()
        model.head_cls7.train()

def configure_stage(model,args):
    model.tcif_detach_context_features=bool(args.tcif_detach_context_features)
    if args.init_checkpoint:
        state=torch.load(args.init_checkpoint,map_location='cpu')
        model.load_state_dict(state,strict=True)
        print('Stage2 initialized from',args.init_checkpoint,flush=True)
    if args.stage2_mode!='none':assert args.init_checkpoint and not args.tcif_detach_context_features
    stage_mode(model,args)
    if args.stage2_mode=='head_only':
        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
        assert trainable==['head_cls7.weight','head_cls7.bias'],trainable
    else:trainable=[n for n,p in model.named_parameters() if p.requires_grad]
    Path(args.save_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.save_dir).parent/'stage2_trainable.json').write_text(json.dumps({'stage':args.stage2_mode,'init_checkpoint':args.init_checkpoint,'trainable_names':trainable},indent=2))

def stage_groups(model,groups,args):
    if args.stage2_mode=='none':return groups
    head=list(model.head_cls7.parameters())
    if args.stage2_mode=='head_only':return [{'name':'head_cls7_only','params':head,'lr':3*args.lr}]
    new=[]
    for group in groups:
        params=[p for p in group['params'] if not any(p is h for h in head)]
        if params:new.append({**group,'params':params})
    return new+[{'name':'head_cls7','params':head,'lr':3*args.lr}]
