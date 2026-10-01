"""Read-only optimization diagnostics: every clip event, first actual update per epoch."""
import json
from pathlib import Path
import torch

class UpdateAudit:
    def __init__(self,model,optimizer,path):
        self.optimizer=optimizer;self.root=Path(path);self.root.mkdir(parents=True,exist_ok=True)
        names={id(p):n for n,p in model.named_parameters()};self.count=0;self.last_epoch=None;self.snapshot=None
        groups=[]
        for g in optimizer.param_groups:
            groups.append(dict(name=g['name'],initial_lr=g['lr'],parameter_count=sum(p.numel() for p in g['params']),names=[names[id(p)] for p in g['params']]))
        (self.root/'optimizer_groups.json').write_text(json.dumps(groups,indent=2))
    def before(self,epoch):
        self.sampled=self.last_epoch!=epoch
        if not self.sampled:return
        self.snapshot={};self.pre={}
        for g in self.optimizer.param_groups:
            pairs=[(p,p.detach().clone()) for p in g['params'] if p.grad is not None]
            self.snapshot[g['name']]=pairs
            self.pre[g['name']]=dict(weight_l2=float(torch.stack([v.float().square().sum() for _,v in pairs]).sum().sqrt()) if pairs else 0.,
                gradient_l2=float(torch.stack([p.grad.detach().float().square().sum() for p,_ in pairs]).sum().sqrt()) if pairs else 0.,
                active_elements=sum(p.numel() for p,_ in pairs))
    def after(self,epoch,grad_norm,clip):
        self.count+=1;row=dict(epoch=epoch,optimizer_step=self.count,preclip_global_norm=float(grad_norm),clipped=float(grad_norm)>clip,
            lr={g['name']:g['lr'] for g in self.optimizer.param_groups})
        if self.sampled:
            groups={}
            for name,pairs in self.snapshot.items():
                delta=float(torch.stack([(p.detach()-old).float().square().sum() for p,old in pairs]).sum().sqrt()) if pairs else 0.
                groups[name]=dict(**self.pre[name],update_l2=delta,relative_update_l2=delta/max(self.pre[name]['weight_l2'],1e-30))
            row['first_update_of_epoch']=groups
        with (self.root/'optimizer_updates.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        self.last_epoch=epoch;self.snapshot=None;self.pre=None
