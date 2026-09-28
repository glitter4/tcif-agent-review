"""Strict full/selective initialization and fixed-budget pretraining controls."""
import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Sampler
from context_variant import prepare
RESET_PREFIXES=('tcif_regression.','tcif_ordinal.','tcif_context_reg_head.','tcif_context_cls7_head.')

def initialize(model,args):
    variant=args.context_fusion
    prepare(model,variant)
    phase=args.transfer_phase
    out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    record=dict(phase=phase,context_fusion=variant,optimizer_restored=False,
        parameter_count=sum(p.numel() for p in model.parameters()),context_variant=getattr(model,'context_variant_record',None))
    if phase=='source':
        expected='cmumosi' if args.source_compute_control else 'cmumosei'
        assert args.dataset==expected and args.epochs==4
        assert not args.evaluate_test_each_epoch and args.save_latest_checkpoint and not args.transfer_init_checkpoint
        record.update(status='source_fresh',source_fixed_epochs=4,compute_control=args.source_compute_control)
    elif phase=='target':
        assert args.dataset=='cmumosi' and args.checkpoint_selection_split=='test' and args.epochs==200
        assert args.evaluate_test_each_epoch
        ck=Path(args.transfer_init_checkpoint);metadata=json.loads(ck.with_suffix('.json').read_text())
        expected='cmumosi' if args.source_compute_control else 'cmumosei'
        assert metadata['checkpoint_epoch']==4 and metadata['dataset']==expected and metadata['transfer_phase']=='source'
        assert metadata.get('context_fusion','tcif')==variant
        state=torch.load(ck,map_location='cpu',weights_only=False,mmap=True)
        state=state.get('model_state_dict',state.get('state_dict',state))
        fresh=model.state_dict()
        assert set(fresh)==set(state)
        assert all(fresh[k].shape==state[k].shape for k in state)
        reset={k:v.detach().cpu().clone() for k,v in fresh.items() if args.transfer_reset_tcif and k.startswith(RESET_PREFIXES)}
        if reset:assert variant=='tcif'
        model.load_state_dict(state,strict=True)
        if reset:model.load_state_dict(reset,strict=False)
        for k,v in model.state_dict().items():
            assert torch.equal(v.detach().cpu(),reset[k] if k in reset else state[k]),k
        if reset:
            for branch in [model.tcif_regression,model.tcif_ordinal]:
                assert torch.count_nonzero(branch.output_projection.weight)==0
                assert torch.count_nonzero(branch.output_projection.bias)==0
        record.update(status='passed',source_epoch=4,source_dataset=expected,strict_shapes=True,
            inherited_tensor_count=len(state)-len(reset),reset_tensor_names=sorted(reset),
            inherited_tensors_exact=True,reset_matches_fresh_seeded_initialization=bool(reset),output_projection_zero=bool(reset))
    else:
        assert not args.transfer_init_checkpoint and not args.transfer_reset_tcif
        record['status']='target_fresh'
    (out.parent/'initialization_audit.json').write_text(json.dumps(record,indent=2))
    print('INITIALIZATION_AUDIT',record,flush=True)

class ExposureBatchSampler(Sampler):
    """1020 batches16 plus one batch6 per epoch, four epochs."""
    def __init__(self,size,seed=123,total=16326,batch_size=16):
        self.size=size;self.total=total;self.batch=batch_size
        self.generator=torch.Generator().manual_seed(seed)
    def __len__(self):return (self.total+self.batch-1)//self.batch
    def __iter__(self):
        indices=[]
        while len(indices)<self.total:indices.extend(torch.randperm(self.size,generator=self.generator).tolist())
        indices=indices[:self.total]
        for i in range(0,self.total,self.batch):yield indices[i:i+self.batch]

def train_loader(original,dataset,args):
    if not (args.transfer_phase=='source' and args.source_compute_control):return original
    assert len(dataset)==1284 and args.batch_size==16 and args.grad_accum_steps==2 and args.num_workers==0
    return DataLoader(dataset,batch_sampler=ExposureBatchSampler(len(dataset),args.seed),num_workers=0)

def epoch_record(args,epoch,val_metrics):
    if args.transfer_phase=='source':
        with (Path(args.save_dir).parent/'source_val_trajectory.jsonl').open('a') as f:
            f.write(json.dumps(dict(epoch=epoch,val_metrics=val_metrics),allow_nan=False)+'\n')

class Trace:
    def __init__(self,args):
        self.args=args;self.root=Path(args.save_dir).parent;self.steps=0
    def begin_epoch(self):
        self.started=time.perf_counter();self.examples=0;self.batches=0;self.elements={};self.extra_sum=0.;self.weak_pos=0;self.weak_neg=0;self.empty=0;self.epoch_steps=0
    def batch(self,batch,extra=None):
        self.examples+=len(batch['id']);self.batches+=1
        for k in ['input_ids','cached_vision','cached_audio','tcif_context_input_ids','tcif_context_cached_vision','tcif_context_cached_audio']:
            if k in batch:self.elements[k]=self.elements.get(k,0)+batch[k].numel()
        if extra:
            self.extra_sum+=extra['extra_weighted'];self.weak_pos+=extra['weak_positive'];self.weak_neg+=extra['weak_negative']
            self.empty+=int(extra['weak_positive']+extra['weak_negative']==0)
    def step(self):self.steps+=1;self.epoch_steps+=1
    def end_epoch(self,epoch):
        if torch.cuda.is_available():torch.cuda.synchronize()
        record=dict(epoch=epoch+1,optimizer_steps=self.epoch_steps,total_optimizer_steps=self.steps,
            microbatches=self.batches,examples=self.examples,training_wall_seconds=time.perf_counter()-self.started,
            input_elements=self.elements,mean_S_weighted_loss=self.extra_sum/max(1,self.batches),
            weak_positive_exposures=self.weak_pos,weak_negative_exposures=self.weak_neg,empty_weak_batches=self.empty)
        with (self.root/'compute_trace.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
