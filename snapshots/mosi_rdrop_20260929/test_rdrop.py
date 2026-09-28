"""Objective, graph-boundary, stochastic-view and reference-order tests."""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parent/'server-code'))
from rdrop_support import symmetric_kl,combine,versions,assert_unchanged,Controller

torch.manual_seed(123)
a=torch.randn(5,7,requires_grad=True);b=torch.randn(5,7,requires_grad=True)
kl=symmetric_kl(a,b)
explicit=.5*(F.kl_div(F.log_softmax(a,-1),F.softmax(b,-1),reduction='batchmean')+
             F.kl_div(F.log_softmax(b,-1),F.softmax(a,-1),reduction='batchmean'))
assert torch.allclose(kl,explicit,atol=1e-7) and torch.allclose(kl,symmetric_kl(b,a))
ga,gb=torch.autograd.grad(kl,[a,b]);assert ga.abs().sum()>0 and gb.abs().sum()>0
x=torch.randn(4,7,requires_grad=True);same=symmetric_kl(x,x)
assert abs(float(same))<1e-7 and torch.autograd.grad(same,x)[0].abs().max()<1e-6
assert torch.isfinite(symmetric_kl(torch.tensor([[1000.,-1000.]]),torch.tensor([[-1000.,1000.]])))
for weight in [0.,.1]:
    a=torch.randn(5,7,requires_grad=True);b=torch.randn(5,7,requires_grad=True)
    reg1=torch.randn(5,requires_grad=True);reg2=torch.randn(5,requires_grad=True)
    y=torch.randn(5)
    l1=F.l1_loss(reg1,y)+.75*F.cross_entropy(a,torch.zeros(5,dtype=torch.long))+.05*a.square().mean()
    l2=F.l1_loss(reg2,y)+.75*F.cross_entropy(b,torch.zeros(5,dtype=torch.long))+.05*b.square().mean()
    got,k=combine(l1,l2,a,b,weight)
    want=.5*(l1+l2)+weight*symmetric_kl(a,b)
    assert torch.allclose(got,want)
    gs=torch.autograd.grad(got,[a,b,reg1,reg2],retain_graph=True)
    ws=torch.autograd.grad(want,[a,b,reg1,reg2])
    assert all(torch.allclose(g,w,atol=1e-7) for g,w in zip(gs,ws))
    assert torch.allclose(gs[2],.5*(reg1-y).sign()/5)

m=torch.nn.Sequential(torch.nn.Linear(6,30),torch.nn.Dropout(.5),torch.nn.Linear(30,7));m.train()
data=torch.randn(4,6);snap=versions((data,),{})
one=m(data);two=m(data);assert_unchanged(snap)
assert not torch.equal(one,two)
m.eval();assert torch.equal(m(data),m(data))
with tempfile.TemporaryDirectory() as tmp:
    attrs=dict(rdrop_views=2,rdrop_kl_weight=.1,output_head_mode='signed_reg_cls7',cls7_head_type='flat',dataset='cmumosi',seed=123,num_workers=0,
        temporal_batch_mode='shuffle',target_sampler='none',reg_loss_type='l1',cls7_loss_type='soft_ce',cls7_loss_weight=.75,cls7_soft_tau=.3,
        tcif_transition_gate_loss_weight=.05,save_dir=str(Path(tmp)/'checkpoints'),batch_size=16)
    for k in ['alpha','sign_aux_weight','hier_sign_loss_weight','hier_mag_loss_weight','sign_marginal_loss_weight','signed_neutral_band_weight',
        'zero_sign_margin_weight','reg_cls_mag_consistency_weight','cumulative_loss_weight','score_sign_aux_weight','lambda_oacr','temporal_contrast_weight','tcif_context_aux_weight']:attrs[k]=0
    ids=[str(i) for i in range(1284)]
    ctl=Controller(SimpleNamespace(**attrs),ids)
    independent=DataLoader(ids,batch_size=16,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(123))
    for epoch in range(2):
        ctl.begin_epoch()
        for batch in independent:assert list(next(ctl.iterator))==list(batch)
print('PASS symmetric KL value/gradients, zero-KL, C1 exact loss average, no regression KL, same inputs/different dropout, independent data-order RNG')
