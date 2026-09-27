"""Strict same-shape transfer initialization; never restore optimizer state."""
import json
from pathlib import Path
import torch


def initialize(model,args):
    phase=args.transfer_phase
    if phase=='source':
        assert args.dataset=='cmumosei' and args.epochs==4
        assert not args.evaluate_test_each_epoch
        assert args.save_latest_checkpoint and not args.transfer_init_checkpoint
        print('TRANSFER_SOURCE fixed_epoch4; train updates only; val diagnostics; no source test evaluation',flush=True)
        return
    if phase!='target':return
    assert args.dataset=='cmumosi' and args.checkpoint_selection_split=='test'
    assert args.evaluate_test_each_epoch and args.epochs==200
    checkpoint=Path(args.transfer_init_checkpoint)
    metadata=json.loads(checkpoint.with_suffix('.json').read_text())
    assert metadata['checkpoint_epoch']==4 and metadata['dataset']=='cmumosei'
    assert metadata.get('transfer_phase')=='source'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    state=state.get('model_state_dict',state.get('state_dict',state))
    target=model.state_dict()
    assert set(state)==set(target), 'Transfer architecture keys differ'
    for name,value in state.items():
        assert value.shape==target[name].shape,(name,value.shape,target[name].shape)
    model.load_state_dict(state,strict=True)
    for name,value in model.state_dict().items():
        assert torch.equal(value.detach().cpu(),state[name]),'Transfer tensor mismatch: '+name
    out=Path(args.save_dir);out.mkdir(parents=True,exist_ok=True)
    record=dict(status='passed',source_epoch=4,source_dataset='cmumosei',target_dataset='cmumosi',
        loaded_tensor_count=len(state),loaded_elements=sum(x.numel() for x in state.values()),
        exact_tensor_equality=True,optimizer_restored=False,source_selection='fixed epoch4, no test/val checkpoint selection')
    (out.parent/'initialization_audit.json').write_text(json.dumps(record,indent=2))
    print('TRANSFER_INITIALIZATION',record,flush=True)


def epoch_record(args,epoch,val_metrics):
    if args.transfer_phase!='source':return
    with (Path(args.save_dir).parent/'source_val_trajectory.jsonl').open('a') as f:
        f.write(json.dumps(dict(epoch=epoch,val_metrics=val_metrics),allow_nan=False)+'\n')
