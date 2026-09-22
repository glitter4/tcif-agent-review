import copy
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent/'source/server-code'))
import torch
from test_signed_reg_cls7_head import build_tiny_model,tiny_batch
from test_tcif import _tiny_tcif_context
from pathstudy_support import stage_mode,stage_groups
torch.set_num_threads(1)
torch.manual_seed(40)
model=build_tiny_model('signed_reg_cls7',signed_class_count=5,label_min=-1.,label_max=1.,enable_tcif=True,tcif_latent_dim=4,tcif_enable_transition_gate=True)
model.eval()
dim=model.head_input_dim
def run(detach):
    model.zero_grad(set_to_none=True)
    model.tcif_detach_context_features=detach
    torch.manual_seed(9)
    lr=torch.randn(2,dim,requires_grad=True);lc=torch.randn(2,dim,requires_grad=True)
    cr=torch.randn(4,dim,requires_grad=True);cc=torch.randn(4,dim,requires_grad=True)
    local={'y_reg':model.head_signed_reg(lr).squeeze(-1),'cls7_logits':model.head_cls7(lc),'extras':{'feat_task1':lr,'feat_task2':lc}}
    context={'extras':{'feat_task1':cr,'feat_task2':cc}}
    out=model._apply_tcif_to_local_output(local,context,torch.ones(2,2,dtype=torch.bool),torch.tensor([[-1.,1.],[-1.,1.]]),True)
    # Include context auxiliary so context gradient has a nonzero route even at identity TCIF initialization.
    loss=out['y_reg'].square().mean()+out['cls7_logits'].square().mean()+out['extras']['tcif']['context_y_reg'].square().mean()+out['extras']['tcif']['context_cls7_logits'].square().mean()
    loss.backward()
    assert lr.grad is not None and lc.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.tcif_regression.parameters())
    if detach:assert cr.grad is None and cc.grad is None
    else:assert cr.grad is not None and cr.grad.abs().sum()>0 and cc.grad is not None and cc.grad.abs().sum()>0
    return out['y_reg'].detach(),out['cls7_logits'].detach()
a=run(False);b=run(True)
assert all(torch.equal(x,y) for x,y in zip(a,b))
model.tcif_detach_context_features=False
args=SimpleNamespace(stage2_mode='head_only',lr=2.32e-5)
stage_mode(model,args)
assert all(p.requires_grad==n.startswith('head_cls7.') for n,p in model.named_parameters())
assert all(not m.training for n,m in model.named_modules() if n!='head_cls7')
batch=tiny_batch();ctx=_tiny_tcif_context(batch)
def forward():return model(batch['image'],batch['input_ids'],batch['attention_mask'],batch['audio_values'],batch['audio_attention_mask'],**ctx)
before={k:v.detach().clone() for k,v in model.state_dict().items()}
out=forward();reg=out['y_reg'].detach().clone()
opt=torch.optim.AdamW(stage_groups(model,[],args),weight_decay=.01)
torch.nn.functional.cross_entropy(out['cls7_logits'],torch.tensor([0,4])).backward();opt.step()
assert torch.equal(reg,forward()['y_reg'])
assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items() if not k.startswith('head_cls7.'))
assert not torch.equal(before['head_cls7.weight'],model.head_cls7.weight)
print('PATHSTUDY_CHECKS_PASSED: identical forward, stopped neighbor gradients, live center/TCIF gradients, head-only parameter and regression invariance',flush=True)
