"""Two stochastic views of the same inputs; KL applies only to class logits."""
import json
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def symmetric_kl(first,second):
    a=F.log_softmax(first,dim=-1);b=F.log_softmax(second,dim=-1)
    return .5*((a.exp()*(a-b)).sum(-1)+(b.exp()*(b-a)).sum(-1)).mean()


def combine(first_loss,second_loss,first_logits,second_logits,weight):
    kl=symmetric_kl(first_logits,second_logits) if weight else symmetric_kl(first_logits.detach(),second_logits.detach())
    return .5*(first_loss+second_loss)+weight*kl,kl


def versions(args,kwargs):
    return [(t,t._version) for t in list(args)+list(kwargs.values()) if torch.is_tensor(t)]


def assert_unchanged(snapshot):
    assert all(t._version==version for t,version in snapshot),'Forward modified shared input tensors in place'


def second_task(out,batch,y,args,reg_fn,cls_fn,gate_fn,centers,class_weights):
    reg=reg_fn(out['y_reg'],y,args.reg_loss_type)
    cls,_=cls_fn(out['cls7_logits'],y,args.cls7_loss_type,args.cls7_soft_tau,
        args.cls7_low_abs_sample_weight_threshold,args.cls7_low_abs_sample_weight,class_weights,centers=centers)
    gate,_=gate_fn(out,batch,y,args)
    return reg+args.cls7_loss_weight*cls+args.tcif_transition_gate_loss_weight*gate,(reg,cls,gate)


class Controller:
    def __init__(self,args,ids):
        assert args.rdrop_views==2 and args.rdrop_kl_weight in (0.,.1)
        assert args.output_head_mode=='signed_reg_cls7' and args.cls7_head_type=='flat'
        assert args.dataset=='cmumosi' and args.seed==123 and args.num_workers==0
        assert args.temporal_batch_mode=='shuffle' and args.target_sampler=='none'
        assert args.reg_loss_type=='l1' and args.cls7_loss_type=='soft_ce' and args.cls7_loss_weight==.75
        assert args.cls7_soft_tau==.3 and args.tcif_transition_gate_loss_weight==.05
        for key in ['alpha','sign_aux_weight','hier_sign_loss_weight','hier_mag_loss_weight','sign_marginal_loss_weight',
                    'signed_neutral_band_weight','zero_sign_margin_weight','reg_cls_mag_consistency_weight',
                    'cumulative_loss_weight','score_sign_aux_weight','lambda_oacr','temporal_contrast_weight','tcif_context_aux_weight']:
            assert getattr(args,key)==0,(key,getattr(args,key))
        self.args=args;self.ids=list(map(str,ids));self.index={s:i for i,s in enumerate(self.ids)}
        assert len(self.ids)==1284
        self.reference_loader=DataLoader(self.ids,batch_size=args.batch_size,shuffle=True,num_workers=0,
            generator=torch.Generator().manual_seed(args.seed))
        self.root=Path(args.save_dir).parent;self.root.mkdir(parents=True,exist_ok=True)
        self.steps=0
        protocol=dict(views=2,kl_weight=args.rdrop_kl_weight,kl='mean batch, half sum of KL(p||q) and KL(q||p)',
            active_base_losses='L1 + .75 softCE + .05 gate; average both views',
            same_input_tensors=True,reference_sampler='seed123 DataLoader with dedicated generator and1284 IDs',
            validation_inference='one deterministic eval forward',training_accuracy_log='first stochastic view',
            training_loss_parts_log='mean of the two supervised views; KL separately in rdrop_steps.jsonl')
        (self.root/'rdrop_protocol.json').write_text(json.dumps(protocol,indent=2))
        print('RDROP_PROTOCOL',protocol,flush=True)

    def begin_epoch(self):
        self.iterator=iter(self.reference_loader);self.started=time.perf_counter()
        self.n=0;self.batches=0;self.kl=0.;self.first=0.;self.second=0.;self.epoch_steps=0
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()

    def record(self,batch,first,second,kl,logits_first,logits_second,epoch,step):
        ids=list(map(str,batch['id']))
        assert ids==list(next(self.iterator)),'Sample order differs from original C0 seeded loader'
        n=len(ids);self.n+=n;self.batches+=1
        k=float(kl.detach());a=float(first.detach());b=float(second.detach())
        assert k>=-1e-6
        self.kl+=k*n;self.first+=a*n;self.second+=b*n
        row=dict(epoch=epoch+1,micro_step=step+1,sample_indices=[self.index[x] for x in ids],n=n,
            base_loss_first=a,base_loss_second=b,kl=k,weighted_kl=self.args.rdrop_kl_weight*k,
            max_logit_difference=float((logits_first.detach()-logits_second.detach()).abs().max()),
            same_inputs=True,data_order_matches_reference=True)
        with (self.root/'rdrop_steps.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')

    def step(self):self.steps+=1;self.epoch_steps+=1

    def end_epoch(self,epoch):
        assert self.n==1284 and self.batches==81
        try:next(self.iterator)
        except StopIteration:pass
        else:raise AssertionError('Reference loader has unexpected remainder')
        if torch.cuda.is_available():torch.cuda.synchronize()
        row=dict(epoch=epoch+1,optimizer_steps=self.epoch_steps,total_optimizer_steps=self.steps,
            unique_epoch_examples=self.n,microbatches=self.batches,stochastic_forward_calls=2*self.batches,
            model_center_exposures=2*self.n,mean_kl=self.kl/self.n,
            mean_base_loss_first=self.first/self.n,mean_base_loss_second=self.second/self.n,
            training_wall_seconds=time.perf_counter()-self.started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
        with (self.root/'rdrop_epochs.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
