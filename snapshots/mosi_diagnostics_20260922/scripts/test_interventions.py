import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
sys.path.insert(0,os.environ['STUDY_CODE'])
from models.tcif import TemporalContextInnovationFilter
import study_controls as s

torch.manual_seed(123)
m=TemporalContextInnovationFilter(8,4,enable_transition_gate=True,transition_gate_hidden_dim=4)
with torch.no_grad():m.output_projection.weight.normal_(std=.2)
x=torch.randn(3,8,requires_grad=True);ctx=torch.randn(3,2,8,requires_grad=True)
mask=torch.ones(3,2,dtype=torch.bool);pos=torch.tensor([[-1.,1.]]*3)
y=m(x,ctx,mask,pos)['feature']; original=y.detach().clone()
y.square().sum().backward()
assert ctx.grad.abs().sum()>0
m.zero_grad();x.grad=None;ctx.grad=None
h=m.register_forward_pre_hook(s.detach_context)
y=m(x,ctx,mask,pos)['feature']
assert torch.equal(original,y.detach())
y.square().sum().backward()
assert ctx.grad is None and x.grad.abs().sum()>0
assert m.output_projection.weight.grad.abs().sum()>0
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.transition_gate.parameters())
h.remove()

class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.encoder=torch.nn.Sequential(torch.nn.Linear(4,6),torch.nn.Dropout(.5))
        self.head_cls7=torch.nn.Linear(6,7);self.head_signed_reg=torch.nn.Linear(6,1)
    def forward(self,x):
        z=self.encoder(x);return self.head_cls7(z),self.head_signed_reg(z)

with tempfile.TemporaryDirectory() as tmp:
    m=Tiny();state=Path(tmp)/'initial.pth';torch.save(m.state_dict(),state)
    os.environ['STUDY_INIT']=str(state);os.environ['STUDY_MODE']='head_only'
    groups=s.configure(m,[dict(name='head',params=list(m.parameters()),lr=7.5e-6)],SimpleNamespace(save_dir=tmp))
    # Simulate original per-epoch unfreezing, then enforce the intervention.
    m.train()
    for p in m.parameters():p.requires_grad_(True)
    s.enforce_epoch(m)
    assert not m.encoder.training and not m.head_signed_reg.training
    assert {n for n,p in m.named_parameters() if p.requires_grad}=={'head_cls7.weight','head_cls7.bias'}
    x=torch.randn(5,4);before=m(x)[1].detach().clone()
    opt=torch.optim.AdamW(groups);m(x)[0].square().sum().backward();opt.step()
    assert torch.equal(before,m(x)[1].detach())
    os.environ['STUDY_MODE']='full_continue'
    groups=s.configure(m,[dict(name='head',params=list(m.parameters()),lr=7.5e-6)],SimpleNamespace(save_dir=tmp))
    ids=[id(p) for g in groups for p in g['params']]
    assert len(ids)==len(set(ids))==len(list(m.parameters()))
    assert groups[-1]['lr']==2.25e-5
print('PASS: detach forward equality, blocked neighbor gradient, live center/gate gradients; exact head-only freeze, fixed regression and matched optimizer coverage')
