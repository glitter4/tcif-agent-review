import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import torch
CODE=Path(__file__).resolve().parents[1]/'server-code'
sys.path.insert(0,str(CODE))
from models.tcif import TemporalContextInnovationFilter as TCIF
from context_variant import StandardContextFusion
from transfer_support import initialize,ExposureBatchSampler,RESET_PREFIXES
from grs_controls import extra_loss

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(12,12);self.head_cls7=torch.nn.Linear(12,7);self.head_signed_reg=torch.nn.Linear(12,1)
        self.tcif_regression=TCIF(12,4,enable_transition_gate=True,transition_gate_hidden_dim=4)
        self.tcif_ordinal=TCIF(12,4,enable_transition_gate=True,transition_gate_hidden_dim=4)
        self.tcif_context_reg_head=torch.nn.Linear(4,1);self.tcif_context_cls7_head=torch.nn.Linear(4,7)

with tempfile.TemporaryDirectory() as td:
    root=Path(td);torch.manual_seed(11);source=Toy()
    with torch.no_grad():
        for p in source.parameters():p.add_(.1)
    ck=root/'source.pth';torch.save(source.state_dict(),ck)
    ck.with_suffix('.json').write_text(json.dumps(dict(dataset='cmumosei',transfer_phase='source',checkpoint_epoch=4)))
    torch.manual_seed(123);model=Toy();fresh={k:v.clone() for k,v in model.state_dict().items()}
    args=SimpleNamespace(context_fusion='tcif',transfer_phase='target',save_dir=str(root/'target/ckpt'),
        dataset='cmumosi',checkpoint_selection_split='test',epochs=200,evaluate_test_each_epoch=True,
        transfer_init_checkpoint=str(ck),source_compute_control=False,transfer_reset_tcif=True)
    initialize(model,args)
    for k,v in model.state_dict().items():assert torch.equal(v,fresh[k] if k.startswith(RESET_PREFIXES) else source.state_dict()[k])
    x=torch.randn(3,12);ctx=torch.randn(3,2,12);mask=torch.tensor([[1,1],[1,0],[0,0]],dtype=torch.bool);pos=torch.zeros(3,2)
    out=model.tcif_regression(x,ctx,mask,pos)['feature'];assert torch.equal(out,x)
    out.square().sum().backward();assert model.tcif_regression.output_projection.weight.grad.abs().sum()>0

full=TCIF(3072,96,enable_transition_gate=True,transition_gate_hidden_dim=64)
ref=sum(p.numel() for p in full.parameters());std=StandardContextFusion(3072,96,ref)
assert abs(sum(p.numel() for p in std.fusion.parameters())-ref)/ref<.005
x=torch.randn(2,3072);c=torch.randn(2,2,3072);mask=torch.tensor([[1,0],[0,0]],dtype=torch.bool);pos=torch.zeros(2,2)
assert torch.equal(std(x,c,mask,pos)['feature'],x)
with torch.no_grad():std.fusion[-1].weight.normal_(std=.01)
a=std(x,c,mask,pos)['feature'];c[~mask]=1e5;b=std(x,c,mask,pos)['feature']
assert torch.equal(a,b) and torch.equal(a[1],x[1])
a.square().sum().backward();assert std.fusion[1].weight.grad.abs().sum()>0

sampler=ExposureBatchSampler(1284);reference=ExposureBatchSampler(1284)
examples=0;updates=0
for epoch in range(4):
    batches=list(sampler);assert batches==list(reference)
    assert len(batches)==1021 and len(batches[-1])==6 and all(len(b)==16 for b in batches[:-1])
    assert all(0<=i<1284 for b in batches for i in b)
    examples+=sum(map(len,batches));updates+=(len(batches)+1)//2
assert examples==65304 and updates==2044

centers=torch.arange(-3,4,dtype=torch.float32);z=torch.zeros(4,7,requires_grad=True);r=torch.zeros(4,requires_grad=True)
loss,stats=extra_loss('S',r,z,torch.tensor([.2,-.2,0.,1.]),centers)
assert abs(float(loss)-.005)<1e-7
loss.backward();assert r.grad[0]<0 and r.grad[1]>0 and r.grad[2]==r.grad[3]==0
print('PASS selective TCIF reset/rest inheritance, zero residual plus live learning; standard masked fusion active capacity; exact2044 steps/65304 exposures; original S boundary')
