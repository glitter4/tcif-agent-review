"""Mechanism tests, not training a new experiment."""
import json,sys,tempfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parent/'server-code'))
from update_audit import UpdateAudit
from metric_protocol import classes
from head_utils import continuous_to_cls7_hard
from text_only import TextModel,TextData,objective

x=torch.tensor([-9,-2.5,-1.5,-.5,0,.5,1.5,2.5,9],dtype=torch.float32)
assert np.array_equal(classes(x.numpy(),'legacy_away'),continuous_to_cls7_hard(x).numpy()-3)
assert np.array_equal(classes(x.numpy(),'nearest_even'),x.round().clamp(-3,3).numpy())
with tempfile.TemporaryDirectory() as d:
    a=nn.Linear(3,1);b=nn.Linear(3,1);b.load_state_dict(a.state_dict())
    oa=torch.optim.AdamW([dict(name='head',params=a.parameters(),lr=.01)])
    ob=torch.optim.AdamW([dict(name='head',params=b.parameters(),lr=.01)])
    log=UpdateAudit(a,oa,d);z=torch.ones(2,3)
    for step in range(3):
        a(z).sum().backward();b(z).sum().backward();state=torch.get_rng_state().clone()
        log.before(1);na=torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
        oa.step();ob.step();log.after(1,na,1.);oa.zero_grad();ob.zero_grad()
        assert torch.equal(state,torch.get_rng_state())
        assert all(torch.equal(u,v) for u,v in zip(a.parameters(),b.parameters()))
    rows=[json.loads(s) for s in (Path(d)/'optimizer_updates.jsonl').read_text().splitlines()]
    assert len(rows)==3 and rows[0]['first_update_of_epoch']['head']['update_l2']>0 and 'first_update_of_epoch' not in rows[1]
root='/path/to/user/datasets/MER-unibench/cmumosi-process';backbone='/path/to/user/models/AI-ModelScope_roberta-base'
data=TextData(root,'train',backbone);assert len(data)==1284
ids,mask,y,i=data[0];assert i==0 and ids.shape==(128,)
model=TextModel(backbone).cuda();model.train()
assert all(not p.requires_grad for p in model.bert.embeddings.parameters())
assert all(p.requires_grad for p in model.bert.encoder.layer.parameters())
r,z=model(ids[None].cuda(),mask[None].cuda());loss=objective(r,z,y[None].cuda());loss.backward()
assert torch.isfinite(loss) and z.shape==(1,7) and all(p.grad is None for p in model.bert.embeddings.parameters())
assert model.reg.weight.grad is not None and model.cls.weight.grad is not None
model.eval()
with torch.no_grad():
    a=model(ids[None].cuda(),mask[None].cuda());b=model(ids[None].cuda(),mask[None].cuda())
assert all(torch.equal(u,v) for u,v in zip(a,b))
print('PASS metrics exact mapping; logger no parameter/RNG perturbation; text-only dataset, frozen embeddings,12 trainable layers, loss gradients, single eval forward')
