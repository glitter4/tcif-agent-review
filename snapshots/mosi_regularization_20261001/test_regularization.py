import sys,tempfile,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'server-code'))
from regularization_support import EMA,Controller,drop_neighbors
from models.tcif import TemporalContextInnovationFilter
from train_emotion import _compute_tcif_transition_gate_loss,_tcif_context_kwargs,_build_lr_scheduler,_step_lr_scheduler

# Exact EMA recurrence including frozen weights and non-floating buffers; restores raw/RNG.
m=nn.Sequential(nn.Linear(3,2),nn.Dropout(.5));m.register_buffer('counter',torch.tensor(3));m[0].bias.requires_grad=False
ema=EMA();ema.initialize(m);previous={k:v.clone() for k,v in m.state_dict().items()}
with torch.no_grad():m[0].weight.add_(.2);m.counter.add_(1)
ema.update(m)
assert torch.allclose(ema.shadow['0.weight'],previous['0.weight'].lerp(m[0].weight,.01))
assert torch.equal(ema.shadow['0.bias'],m[0].bias) and ema.shadow['counter']==4
raw={k:v.clone() for k,v in m.state_dict().items()};state=torch.get_rng_state().clone();mode=m.training
with ema.applied(m):
    m.eval();torch.rand(20);random.random();np.random.rand()
assert torch.equal(state,torch.get_rng_state()) and m.training==mode
assert all(torch.equal(v,m.state_dict()[k]) for k,v in raw.items())

# Entire paired optimizer trajectory is bitwise equal with/without EMA observation.
torch.manual_seed(3);a=nn.Sequential(nn.Linear(3,2),nn.Dropout(.5));b=nn.Sequential(nn.Linear(3,2),nn.Dropout(.5));b.load_state_dict(a.state_dict())
oa=torch.optim.AdamW(a.parameters(),lr=.01);ob=torch.optim.AdamW(b.parameters(),lr=.01);em=EMA();em.initialize(a)
for _ in range(4):
    state=torch.get_rng_state();x=torch.ones(2,3)
    a(x).sum().backward();oa.step();oa.zero_grad();em.update(a)
    post=torch.get_rng_state()
    with em.applied(a):a.eval();a(x);torch.rand(12)
    assert torch.equal(post,torch.get_rng_state())
    torch.set_rng_state(state);b(x).sum().backward();ob.step();ob.zero_grad()
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))

# Mask invariants and finite all-masked fallback. Alter masked context cannot affect priors/gates.
gen=torch.Generator().manual_seed(123);mask=torch.tensor([[True,False],[True,True],[False,False]])
batch={'tcif_context_valid_mask':mask,'input_ids':torch.tensor([[1],[2],[3]]),'tcif_context_raw_valence':torch.tensor([[1.,-1.],[1.,-1.],[0.,0.]])}
changed,counts=drop_neighbors(batch,1.,gen);assert not changed['tcif_context_valid_mask'].any() and torch.equal(batch['tcif_context_valid_mask'],mask)
assert changed['input_ids'] is batch['input_ids'];assert counts['dropped']==3
changed0,_=drop_neighbors(batch,0.,gen);assert torch.equal(changed0['tcif_context_valid_mask'],mask)
big={'tcif_context_valid_mask':torch.ones(20000,2,dtype=torch.bool)};sampled,c=drop_neighbors(big,.2,gen)
assert .19<c['dropped']/c['valid_before']<.21
f=TemporalContextInnovationFilter(8,4,enable_transition_gate=True)
with torch.no_grad():f.output_projection.weight.normal_();f.transition_gate[-1].weight.normal_()
local=torch.randn(3,8);ctx=torch.randn(3,2,8,requires_grad=True);pos=torch.tensor([[-1.,1.]]*3)
use=torch.tensor([[True,False],[False,True],[False,False]])
r=f(local,ctx,use,pos);mut=ctx.detach().clone();mut[~use]=1000
q=f(local,mut,use,pos)
for k in ['prior_mean','prior_variance','posterior_mean','posterior_variance','continuation_gate','continuation_gate_logits']:
    assert torch.equal(r[k],q[k]),k
assert (r['context_attention'][~use]==0).all() and torch.isfinite(r['posterior_feature']).all()
assert torch.equal(r['posterior_feature'][2],r['local_feature'][2])
r['posterior_feature'].sum().backward();assert ctx.grad[~use].abs().max()==0

# Forward adapter and gate loss see exact same dropped mask; dropped opposite label excluded.
batch['tcif_context_valid_mask']=use
kwargs=_tcif_context_kwargs(batch,torch.device('cpu'),False);assert torch.equal(kwargs['tcif_context_valid_mask'],use)
args=SimpleNamespace(tcif_transition_gate_tau=.75,tcif_transition_gate_conflict_target=0.)
payload=dict(regression_continuation_gate_logits=r['continuation_gate_logits'],ordinal_continuation_gate_logits=r['continuation_gate_logits'],
             regression_continuation_gate=r['continuation_gate'],ordinal_continuation_gate=r['continuation_gate'])
y=torch.tensor([1.,-1.,0.]);loss,stats=_compute_tcif_transition_gate_loss({'extras':{'tcif':payload}},batch,y,args)
batch2=dict(batch);batch2['tcif_context_raw_valence']=batch['tcif_context_raw_valence'].clone();batch2['tcif_context_raw_valence'][~use]=999
loss2,stats2=_compute_tcif_transition_gate_loss({'extras':{'tcif':payload}},batch2,y,args)
assert torch.equal(loss,loss2) and stats==stats2

# Plateau changes only after patience, same initial LRs, relative floor and actual val-MAE input.
p=nn.Parameter(torch.ones(1));o=torch.optim.AdamW([dict(params=[p],lr=2e-4)])
args=SimpleNamespace(scheduler='plateau',plateau_relative_min_factor=.03,min_lr=0.,plateau_factor=.5,plateau_patience=5,plateau_monitor='val_mae')
s=_build_lr_scheduler(o,args)
for _ in range(6):_step_lr_scheduler(s,args,0.,1.)
assert o.param_groups[0]['lr']==2e-4
_step_lr_scheduler(s,args,0.,1.);assert o.param_groups[0]['lr']==1e-4
with tempfile.TemporaryDirectory() as d:
    model=nn.Linear(2,1);args=SimpleNamespace(save_dir=str(Path(d)/'checkpoints'),enable_ema=True,ema_decay=.99,seed=123,neighbor_dropout=0.,
        scheduler='cosine',rdrop_views=1,rdrop_kl_weight=0.,checkpoint_selection_split='test',epochs=200,lr_warmup_epochs=1)
    controller=Controller(args,model)
    from torch.utils.data import DataLoader
    loaders=[DataLoader([1,2],generator=torch.Generator().manual_seed(i)) for i in [1,2]]
    controller.after_step(model,1);assert controller.ema.shadow is None
    controller.after_epoch(model,1,None,None,*loaders,torch.device('cpu'),0.,None)
    assert controller.ema.updates==0
    with torch.no_grad():model.weight.add_(.1)
    controller.after_step(model,2);assert controller.ema.updates==1
    raw={k:v.clone() for k,v in model.state_dict().items()};states=[l.generator.get_state() for l in loaders]
    def fake_validate(model,loader,*unused):
        list(loader);torch.rand(5);result=[None]*13;result[6]=.7;result[9]=.84;result[10]=.46;return result
    def fake_save(model,save_dir,epoch,val,test,best,args):
        assert epoch==2 and test['acc7']==.46 and args.ema_weight_source
        assert all(torch.equal(v,controller.ema.shadow[k]) for k,v in model.state_dict().items())
        return [('acc7',{}),('mae',{})]
    controller.after_epoch(model,2,fake_validate,fake_save,*loaders,torch.device('cpu'),0.,None)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in raw.items())
    assert all(torch.equal(x,l.generator.get_state()) for x,l in zip(states,loaders))
print('PASS EMA recurrence/frozen state/RNG/raw paired trajectory; neighbor mask/statistics/gradients/gate supervision/all-invalid; plateau val-MAE/patience/floor')
