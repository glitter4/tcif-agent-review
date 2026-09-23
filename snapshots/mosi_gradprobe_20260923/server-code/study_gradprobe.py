"""Four fixed accumulation steps, no optimizer step or weight mutation."""
import json
import math
import os
from pathlib import Path
import re
import torch

OUT=Path(os.environ.get('STUDY_PROBE_OUT','/tmp/tcif_mosi_gradprobe'))
PARAMS=[];NAMES=[];GROUPS={};BEFORE={};PENDING={};BATCH=[];ROWS=[];FORWARD=[]
EFFECTIVE=4
MICROBATCHES=EFFECTIVE*2


def group(name):
    if name.startswith('shared_specific_layers.'):
        if name.endswith('.phi'):return 'router_phi'
        if name.endswith('.scale'):return 'router_scale'
        match=re.search(r'\.experts\.(\d+)\.',name)
        if match:
            i=int(match.group(1))
            return 'expert_shared' if i==0 else ('expert_text' if i in (1,2) else ('expert_audio' if i in (3,4) else 'expert_vision'))
        return 'moe_nonexpert'
    if name.startswith('bert.'):
        match=re.search(r'encoder\.layer\.(\d+)\.',name)
        return 'text_layer_'+match.group(1) if match else 'text_other'
    if name.startswith('tcif_regression.'):
        return 'tcif_reg_gate' if '.transition_gate.' in name else 'tcif_regression'
    if name.startswith('tcif_ordinal.'):
        return 'tcif_cls_gate' if '.transition_gate.' in name else 'tcif_ordinal'
    if name.startswith('head_signed_reg.'):return 'head_regression'
    if name.startswith('head_cls7.'):return 'head_classifier'
    if name.startswith('task_pool_intensity.'):return 'task_pool_regression'
    if name.startswith('task_pool_polarity.'):return 'task_pool_classifier'
    return 'other_trainable'


def begin(model,args):
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'result.json').exists(), 'Probe result already exists'
    with (OUT/'probe.lock').open('x') as f:
        f.write(str(os.getpid()))
    global BEFORE
    BEFORE={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    (OUT/'configuration.json').write_text(json.dumps(dict(mode=os.environ['STUDY_MODE'],view=os.environ['STUDY_DIAG_VIEW'],
        checkpoint=os.environ['STUDY_INIT'],seed=args.seed,grad_accum_steps=args.grad_accum_steps,
        effective_batches=EFFECTIVE,number_of_microbatches=MICROBATCHES,
        sample_selection='first eight batches of seed123 train DataLoader, no test-error selection',
        parameter_update=False),indent=2))
    print('GRADPROBE_BEGIN',OUT,flush=True)


def enforce_model_mode(model):
    if os.environ['STUDY_DIAG_VIEW']=='eval':
        model.eval()
    else:
        mask_seed=int(os.environ['STUDY_DIAG_MASK_SEED'])
        torch.manual_seed(mask_seed)
        torch.cuda.manual_seed_all(mask_seed)


def setup_params(model):
    if PARAMS:return
    for n,p in model.named_parameters():
        if p.requires_grad:
            NAMES.append(n);PARAMS.append(p)
            GROUPS.setdefault(group(n),[]).append(len(PARAMS)-1)
    assert PARAMS
    assert len({i for ids in GROUPS.values() for i in ids})==len(PARAMS)


def tensor_grad(loss,divisor):
    if not loss.requires_grad:return [None]*len(PARAMS)
    gradients=torch.autograd.grad(loss/divisor,PARAMS,retain_graph=True,allow_unused=True)
    return [g.detach().cpu() if g is not None else None for g in gradients]


def stats(gradients):
    def l2(arr):return math.sqrt(sum(float(g.double().square().sum()) for g in arr if g is not None))
    total=[None if all(gradients[t][i] is None for t in gradients) else sum(
        (gradients[t][i] if gradients[t][i] is not None else torch.zeros_like(PARAMS[i],device='cpu')) for t in gradients)
        for i in range(len(PARAMS))]
    totalnorm=l2(total)
    records={}
    for name,indices in GROUPS.items():
        comps={k:[gradients[k][i] for i in indices] for k in gradients}
        norms={k:l2(v) for k,v in comps.items()}
        norms['total']=l2([total[i] for i in indices])
        a,b=comps['reg'],comps['cls_weighted']
        dot=sum(float(x.double().flatten().dot(y.double().flatten())) for x,y in zip(a,b) if x is not None and y is not None)
        if norms['reg']<1e-12 or norms['cls_weighted']<1e-12:
            cosine=None
            reason='missing_gradient_path' if not any(x is not None for x in a) or not any(y is not None for y in b) else 'near_zero_gradient'
        else:
            cosine=dot/(norms['reg']*norms['cls_weighted']);reason=None
        records[name]=dict(parameter_count=sum(PARAMS[i].numel() for i in indices),
            norm=norms,total_norm_share=norms['total']/totalnorm if totalnorm>0 else None,
            reg_cls_dot=dot,reg_cls_cosine=cosine,cosine_na_reason=reason,
            connected={k:any(x is not None for x in v) for k,v in comps.items()})
    return records,totalnorm


def capture(model,batch,output,loss_reg,loss_cls,loss_gate,loss_total,epoch,step,args):
    assert epoch==0 and step<len(range(MICROBATCHES))
    setup_params(model)
    divisor=args.grad_accum_steps
    terms=dict(reg=loss_reg,cls_weighted=args.cls7_loss_weight*loss_cls,
               gate_weighted=args.tcif_transition_gate_loss_weight*loss_gate)
    gradients={k:tensor_grad(v,divisor) for k,v in terms.items()}
    for k,arr in gradients.items():
        if k not in PENDING:PENDING[k]=[None]*len(arr)
        for i,g in enumerate(arr):
            if g is not None:PENDING[k][i]=g if PENDING[k][i] is None else PENDING[k][i]+g
    BATCH.append(dict(sample_ids=[str(x) for x in batch['id']],
        raw_labels=[float(x) for x in batch['raw_valence']],
        valid_neighbor_slots=int(batch['tcif_context_valid_mask'].sum()),
        batch_size=len(batch['id']),
        loss_reg=float(loss_reg.detach()),loss_cls=float(loss_cls.detach()),
        loss_gate=float(loss_gate.detach()),loss_total=float(loss_total.detach())))
    FORWARD.extend(dict(sample_id=str(s),y_reg=float(output['y_reg'][i].detach()),
        logits=output['cls7_logits'][i].detach().cpu().tolist()) for i,s in enumerate(batch['id']))
    if step%divisor==divisor-1:
        records,norm=stats(PENDING)
        base=len(ROWS)*divisor
        rows=BATCH[base:base+divisor]
        ROWS.append(dict(effective_index=len(ROWS),groups=records,
            global_preclip_norm=norm,clip_coefficient=min(1.,1./(norm+1e-6)),
            micro_loss_mean={key:sum(r[key] for r in rows)/len(rows) for key in ['loss_reg','loss_cls','loss_gate','loss_total']},
            sample_count=sum(r['batch_size'] for r in rows),valid_neighbor_slots=sum(r['valid_neighbor_slots'] for r in rows)))
        PENDING.clear()
        print('GRADPROBE_BATCH',len(ROWS),'preclip',norm,flush=True)
    if step+1==MICROBATCHES:
        for k,v in model.state_dict().items():assert torch.equal(BEFORE[k],v.detach().cpu()),'Changed model tensor: '+k
        data=dict(status='complete',configuration=json.loads((OUT/'configuration.json').read_text()),
                  grouped_parameter_names={k:[NAMES[i] for i in ids] for k,ids in GROUPS.items()},
                  batches=ROWS,microbatch_metadata=BATCH,forward=FORWARD,weights_unchanged=True)
        (OUT/'result.json').write_text(json.dumps(data,indent=2,allow_nan=False))
        print('GRADPROBE_COMPLETE',flush=True)
        raise SystemExit(0)
