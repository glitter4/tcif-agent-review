import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'server-code'))
from soft_targets import two_hot
from head_utils import continuous_to_cls7_soft,continuous_to_cls7_hard,soft_cross_entropy
from train_emotion import _compute_cls7_loss
from context_variant import StandardContextFusion
for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
    for dtype in [torch.float32,torch.float64]:
        y=torch.cat([torch.linspace(-3,3,10001,device=device,dtype=dtype),torch.arange(-3,4,device=device,dtype=dtype)])
        c=torch.arange(-3,4,device=device,dtype=dtype);q=two_hot(y,c)
        assert q.device==y.device and q.dtype==y.dtype and (q>=0).all()
        assert torch.allclose(q.sum(-1),torch.ones_like(y),atol=1e-7,rtol=0)
        assert torch.allclose(q@c,y,atol=5e-7 if dtype==torch.float32 else 1e-14,rtol=0)
        assert (q>0).sum(-1).max()<=2 and torch.equal(q[-7:],torch.eye(7,device=device,dtype=dtype))
        z=torch.randn(len(y),7,device=device,dtype=dtype,requires_grad=True)
        loss,hard=_compute_cls7_loss(z,y,'soft_ce',.3,soft_target='two_hot');loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and torch.equal(hard,continuous_to_cls7_hard(y))
        old,_=_compute_cls7_loss(z,y,'soft_ce',.3,soft_target='distance')
        assert torch.equal(old,soft_cross_entropy(z,continuous_to_cls7_soft(y,tau=.3)))
for y in [torch.tensor([3.01]),torch.tensor([-3.01]),torch.tensor([float('nan')])]:
    try:two_hot(y)
    except ValueError:pass
    else:raise AssertionError('Invalid labels must be rejected')
try:two_hot(torch.tensor([.2]),torch.tensor([-.9,-.4,0,.4,.9]))
except ValueError:pass
else:raise AssertionError('Do not apply MOSI target to CH-SIMS')
# Standard fusion must obey the same training neighbor mask, including all-invalid.
m=StandardContextFusion(8,4,1000)
with torch.no_grad():m.fusion[-1].weight.normal_()
x=torch.randn(2,8);ctx=torch.randn(2,2,8,requires_grad=True);mask=torch.tensor([[True,False],[False,False]]);pos=torch.tensor([[-1.,1.]]*2)
a=m(x,ctx,mask,pos);other=ctx.detach().clone();other[~mask]=123
b=m(x,other,mask,pos)
assert torch.equal(a['feature'],b['feature']) and torch.equal(a['prior_mean'],b['prior_mean'])
assert torch.equal(a['feature'][1],x[1]) and torch.isfinite(a['feature']).all()
a['feature'].sum().backward();assert ctx.grad[~mask].abs().max()==0
print('PASS twohot CPU/GPU float32/64, mass/mean/integer/endpoint/range/gradient; legacy loss bitwise unchanged; standard masked fusion')
