"""Mechanism tests, including accumulated projection and unchanged other gradients."""
import sys
from pathlib import Path
from types import SimpleNamespace
import tempfile
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'server-code'))
from grs_controls import Controller,projection_delta,extra_loss,readout_scores


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.phi=torch.nn.Parameter(torch.zeros(2));self.scale=torch.nn.Parameter(torch.ones(1))
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.shared_specific_layers=torch.nn.ModuleList([Layer(),Layer()])
        self.other=torch.nn.Parameter(torch.ones(1))

def probe(regs,classes):
    m=Model();p=[l.phi for l in m.shared_specific_layers]
    with tempfile.TemporaryDirectory() as tmp:
        ctr=Controller(m,SimpleNamespace(grs_mode='G',save_dir=str(Path(tmp)/'checkpoints')))
        for rv,cv in zip(regs,classes):
            lr=sum((x*torch.tensor(v,dtype=x.dtype)).sum() for x,v in zip(p,rv))
            lc=sum((x*torch.tensor(v,dtype=x.dtype)).sum() for x,v in zip(p,cv))
            aux=sum(.3*x.sum() for x in p)+m.other.sum()+sum(l.scale.sum() for l in m.shared_specific_layers)
            ctr.observe(lr,lc,2)
            ((lr+lc+aux)/2).backward()
        original=[x.grad.clone() for x in p]
        other=m.other.grad.clone();scales=[l.scale.grad.clone() for l in m.shared_specific_layers]
        rg=[torch.tensor([rv[i] for rv in regs],dtype=torch.float32).sum(0)/2 for i in range(2)]
        cg=[torch.tensor([cv[i] for cv in classes],dtype=torch.float32).sum(0)/2 for i in range(2)]
        coeff,info=projection_delta(rg,cg)
        ctr.before_clip(0,1)
        for old,x,g in zip(original,p,cg):assert torch.allclose(x.grad,old+coeff*g,atol=1e-7)
        assert torch.equal(m.other.grad,other)
        assert all(torch.equal(l.scale.grad,g) for l,g in zip(m.shared_specific_layers,scales))
        assert ctr.reg is None and ctr.cls is None
        return info

assert probe([[[-2,0],[0,0]],[[-2,0],[0,0]]],[[[1,0],[0,0]],[[1,0],[0,0]]])['projected']
# Conflicting first microbatch but positive effective dot: must not project.
assert not probe([[[-3,0],[0,0]],[[5,0],[0,0]]],[[[1,0],[0,0]],[[1,0],[0,0]]])['projected']
assert not projection_delta([torch.tensor([2.])],[torch.tensor([0.])])[0]
centers=torch.arange(-3,4,dtype=torch.float32)
logits=torch.zeros(4,7,requires_grad=True);reg=torch.tensor([4.,-.2,.3,0.],requires_grad=True)
target=torch.tensor([1.,-.2,.3,0.])
extra,stats=extra_loss('R',reg,logits,target,centers)
e,s=readout_scores(reg,logits,centers)
assert torch.allclose(extra,.1*((e-target).abs().mean()+(s-target).abs().mean()))
extra.backward();assert logits.grad.abs().sum()>0 and reg.grad[0]==0
# Raw L1 still supervises saturated regression outside clamp.
torch.nn.functional.l1_loss(reg,target).backward();assert reg.grad[0]>0
logits=torch.zeros(4,7,requires_grad=True);reg=torch.zeros(4,requires_grad=True)
target=torch.tensor([.2,-.2,0.,1.])
extra,stats=extra_loss('S',reg,logits,target,centers)
assert abs(float(extra)-.005)<1e-7
extra.backward();assert reg.grad[0]<0 and reg.grad[1]>0 and reg.grad[2]==reg.grad[3]==0
assert logits.grad[2:].abs().sum()==0
z=torch.zeros(2,7,requires_grad=True)
extra,_=extra_loss('S',torch.zeros(2),z,torch.tensor([0.,.5]),centers)
extra.backward();assert float(extra)==0 and z.grad.abs().sum()==0
z=torch.tensor([[0.,0.,0.,0.,0.,0.,20.]],requires_grad=True)
extra,_=extra_loss('S',torch.tensor([1.]),z,torch.tensor([.2]),centers)
extra.backward();assert float(extra)==0 and z.grad.abs().sum()==0
print('PASS G effective-batch-only projection, auxiliaries/non-phi untouched, no-conflict/near-zero; R clamps+gradients; S masks/balance/finite margin')
