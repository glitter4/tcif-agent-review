"""Isolated EMA readout and training-only neighbor masking; no new losses."""
import contextlib,copy,json,random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

def append(path,row):
    with Path(path).open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

@contextlib.contextmanager
def preserved_rng():
    python=random.getstate();numpy=np.random.get_state();cpu=torch.get_rng_state()
    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:yield
    finally:
        random.setstate(python);np.random.set_state(numpy);torch.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)

class EMA:
    def __init__(self,decay=.99):
        if decay!=.99:raise ValueError('Only preregistered decay=.99')
        self.decay=decay;self.shadow=None;self.updates=0
    @torch.no_grad()
    def initialize(self,model):
        assert self.shadow is None
        self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self,model):
        assert self.shadow is not None
        state=model.state_dict();assert state.keys()==self.shadow.keys()
        for k,v in state.items():
            if v.is_floating_point():self.shadow[k].lerp_(v.detach(),1-self.decay)
            else:self.shadow[k].copy_(v)
        self.updates+=1
    @contextlib.contextmanager
    def applied(self,model):
        assert self.shadow is not None
        backup={k:v.detach().clone() for k,v in model.state_dict().items()}
        training={m:m.training for m in model.modules()}
        with preserved_rng():
            try:
                model.load_state_dict(self.shadow,strict=True);yield
            finally:
                model.load_state_dict(backup,strict=True)
                for m,flag in training.items():m.training=flag

def drop_neighbors(batch,p,generator):
    if p not in [0.,.2,1.]:raise ValueError('Unsupported mask probability')
    mask=batch['tcif_context_valid_mask'].bool()
    sample=torch.rand(mask.shape,generator=generator,device='cpu').to(mask.device)
    kept=mask&(sample>=p)
    changed=dict(batch);changed['tcif_context_valid_mask']=kept
    return changed,dict(valid_before=int(mask.sum()),dropped=int((mask&~kept).sum()),valid_after=int(kept.sum()),
        no_context_before=int((~mask.any(1)).sum()),no_context_after=int((~kept.any(1)).sum()),n=mask.shape[0])

class Controller:
    def __init__(self,args,model):
        self.args=args;self.root=Path(args.save_dir).parent;self.root.mkdir(parents=True,exist_ok=True)
        self.ema=EMA(args.ema_decay) if args.enable_ema else None
        self.generator=torch.Generator(device='cpu').manual_seed(args.seed+2718)
        self.best={};self.steps=0
        assert not (args.enable_ema and args.neighbor_dropout)
        if args.enable_ema or args.neighbor_dropout:assert args.scheduler=='cosine'
        assert args.rdrop_views==1 and args.rdrop_kl_weight==0
        assert args.checkpoint_selection_split=='test' and args.epochs==200
        append(self.root/'regularization_protocol.jsonl',dict(ema=args.enable_ema,decay=args.ema_decay,
            warmup_epochs=args.lr_warmup_epochs,neighbor_dropout=args.neighbor_dropout,neighbor_rng_seed=args.seed+2718,
            ema_scope='every model parameter and floating buffer, integer buffers copied',ema_start='initialize after completed warmup; update after every subsequent optimizer.step',
            selection='test-selected legacyAcc7/MAE at configured eta; validation diagnostic only',inference='single model forward, neighbor dropout disabled'))
    def training_batch(self,batch,epoch,micro_step):
        if not self.args.neighbor_dropout:return batch
        changed,stats=drop_neighbors(batch,self.args.neighbor_dropout,self.generator)
        append(self.root/'neighbor_dropout.jsonl',dict(epoch=epoch,micro_step=micro_step,**stats))
        return changed
    def after_step(self,model,epoch):
        self.steps+=1
        if self.ema is not None and epoch>self.args.lr_warmup_epochs:self.ema.update(model)
    def after_epoch(self,model,epoch,validate,update_checkpoints,val_loader,test_loader,device,eff_alpha,contrast_head):
        if self.ema is None:return
        if epoch==self.args.lr_warmup_epochs:
            self.ema.initialize(model)
            append(self.root/'ema_epochs.jsonl',dict(epoch=epoch,initialized=True,updates=0,keys=len(self.ema.shadow)))
            return
        if epoch<self.args.lr_warmup_epochs:return
        args=SimpleNamespace(**vars(self.args));args.save_dir=str(self.root/'ema/checkpoints')
        args.ema_weight_source=True
        Path(args.save_dir).mkdir(parents=True,exist_ok=True)
        with self.ema.applied(model):
            values={};generators={loader.generator:loader.generator.get_state() for loader in [val_loader,test_loader] if loader.generator is not None}
            try:
                for split,loader in [('val',val_loader),('test',test_loader)]:
                    result=validate(model,loader,device,eff_alpha,args,contrast_head)
                    values[split]=dict(acc7=result[10],mae=result[6],acc2=result[9])
                promoted=update_checkpoints(model,args.save_dir,epoch,values['val'],values['test'],self.best,args)
            finally:
                for generator,state in generators.items():generator.set_state(state)
        append(self.root/'ema_epochs.jsonl',dict(epoch=epoch,initialized=False,updates=self.ema.updates,
            val=values['val'],test=values['test'],promoted=[x[0] for x in promoted],raw_weights_restored=True))
