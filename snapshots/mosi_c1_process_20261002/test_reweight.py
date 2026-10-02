import sys,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).resolve().parent/'server-code'))
from weak_main_loss import normalization_from_training_labels,sample_weights,weighted_main_losses
from head_utils import continuous_to_cls7_soft
data=np.load('/path/to/user/datasets/MER-unibench/cmumosi-process/label.npz',allow_pickle=True)['train_corpus'].item()
y=torch.tensor([float(data[k]['val']) for k in sorted(data)],dtype=torch.float32)
mean=normalization_from_training_labels(y,split='train');w=sample_weights(y,mean)
assert len(y)==1284 and int(((y.abs()>0)&(y.abs()<.5)).sum())==172
assert abs(float(w.double().mean())-1)<1e-7 and (w>0).all()
assert torch.allclose(sample_weights(torch.tensor([0.,.5,-.5,2.,-.2,.2]),mean),torch.tensor([1,1,1,1,1.5,1.5])/mean)
try:normalization_from_training_labels(y,split='val')
except ValueError:pass
else:raise AssertionError('val must not set weights')
# Split any batch arbitrarily: weighted sums agree because normalization is frozen across batches.
r=torch.randn(len(y),requires_grad=True);z=torch.randn(len(y),7,requires_grad=True);q=continuous_to_cls7_soft(y,tau=.3)
a,b=weighted_main_losses(r,z,y,q,mean)
ar,ac=weighted_main_losses(r[:17],z[:17],y[:17],q[:17],mean);br,bc=weighted_main_losses(r[17:],z[17:],y[17:],q[17:],mean)
assert torch.allclose(a,(ar*17+br*(len(y)-17))/len(y)) and torch.allclose(b,(ac*17+bc*(len(y)-17))/len(y))
gate=torch.randn(len(y),requires_grad=True);gate_loss=F.binary_cross_entropy_with_logits(gate,torch.ones_like(gate))
old=(r-y).abs().mean()+.75*(-q*F.log_softmax(z,-1)).sum(-1).mean()+.05*gate_loss
new=a+.75*b+.05*gate_loss
assert torch.equal(torch.autograd.grad(old,gate,retain_graph=True)[0],torch.autograd.grad(new,gate,retain_graph=True)[0])
gr,gz=torch.autograd.grad(new,[r,z]);assert torch.isfinite(gr).all() and torch.isfinite(gz).all()
assert torch.allclose(gr,torch.sign(r-y)*w/len(y),atol=1e-8)
assert torch.allclose(gz,.75*(F.softmax(z,-1)-q)*w[:,None]/len(y),atol=1e-8)
print('PASS train-only normalization; zero/strong retained; per-batch invariance; exact reg/class gradients; gate auxiliary gradient unchanged',mean)
