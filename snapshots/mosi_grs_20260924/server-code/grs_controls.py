"""Default-off MOSI G/R/S interventions; no new model parameters."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def readout_scores(y_reg,logits,centers):
    expected=(logits.softmax(-1)*centers).sum(-1)
    fused=.2*y_reg.clamp(-3,3)+.8*expected
    return expected,fused


def extra_loss(mode,y_reg,logits,target,centers):
    expected,fused=readout_scores(y_reg,logits,centers)
    stats=dict(weak_positive=int(((target>0)&(target<.5)).sum()),
               weak_negative=int(((target<0)&(target>-.5)).sum()))
    if mode=='R':
        raw=.5*(F.l1_loss(expected,target)+F.l1_loss(fused,target))
        weighted=.2*raw
    elif mode=='S':
        margin=target.abs().clamp(max=.1)
        hinge=.5*(F.relu(margin-target.sign()*expected)+F.relu(margin-target.sign()*fused))
        groups=[]
        for mask in [(target>0)&(target<.5),(target<0)&(target>-.5)]:
            if mask.any():groups.append(hinge[mask].mean())
        raw=torch.stack(groups).mean() if groups else logits.sum()*0
        weighted=.05*raw
    else:
        raw=logits.sum()*0
        weighted=raw
    stats.update(extra_raw=float(raw.detach()),extra_weighted=float(weighted.detach()))
    return weighted,stats


def projection_delta(reg,cls,eps=1e-12):
    dot=sum((a.double()*b.double()).sum() for a,b in zip(reg,cls))
    norm=sum(b.double().square().sum() for b in cls)
    active=bool(norm>eps and dot<0)
    # Both phi tensors form one parameter subspace, not independent per-layer projections.
    coefficient=float(-dot/(norm+eps)) if active else 0.
    return coefficient,dict(projected=active,reg_cls_dot=float(dot),cls_norm_squared=float(norm),
        skip_near_zero=bool(norm<=eps),correction_coefficient=coefficient)


class Controller:
    def __init__(self,model,args):
        self.mode=args.grs_mode
        self.named=[(n,p) for n,p in model.named_parameters() if n.startswith('shared_specific_layers.') and n.endswith('.phi')]
        assert len(self.named)==2, 'Expected two SharedSpecificMoE phi tensors'
        self.reg=None;self.cls=None;self.records=[]
        self.path=Path(args.save_dir).parent/'grs_steps.jsonl'
        self.micro=[]
        print('GRS_PROTOCOL',dict(mode=self.mode,phi_names=[n for n,_ in self.named],
            projection='effective-batch accumulated then before global clipping',readout_weight=.2,
            weak_weight=.05,weak_mask='0<abs(y)<.5',weak_reduction='mean within each present sign group, then mean over present groups'),flush=True)

    def observe(self,reg_loss,cls_loss,divisor):
        if self.mode!='G':return
        params=[p for _,p in self.named]
        r=torch.autograd.grad(reg_loss/divisor,params,retain_graph=True,allow_unused=False)
        c=torch.autograd.grad(cls_loss/divisor,params,retain_graph=True,allow_unused=False)
        if self.reg is None:
            self.reg=[g.detach().clone() for g in r];self.cls=[g.detach().clone() for g in c]
        else:
            for a,g in zip(self.reg,r):a.add_(g.detach())
            for a,g in zip(self.cls,c):a.add_(g.detach())

    def before_clip(self,epoch,step):
        record=dict(epoch=epoch+1,micro_step=step+1,microbatches=self.micro)
        if self.mode=='G':
            assert self.reg is not None and self.cls is not None
            coefficient,info=projection_delta(self.reg,self.cls)
            if coefficient:
                for (_,p),gc in zip(self.named,self.cls):
                    assert p.grad is not None
                    p.grad.add_(gc,alpha=coefficient)
            record.update(info)
        self.reg=None;self.cls=None;self.micro=[]
        with self.path.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
