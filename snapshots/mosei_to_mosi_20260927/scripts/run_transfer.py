"""One fixed-budget MOSEI source run followed by one D1 MOSI target run."""
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import time
import runner as e

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'server-code'
OUT=Path('/path/to/user/m4oe/tcif_mosei_to_mosi_20260927')
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
BASE=e.read(Path(__file__).with_name('D1_config.json'))
SOURCE_RUN=OUT/'source_mosei4'
SOURCE_CKPT=SOURCE_RUN/'checkpoints/recovery/latest_model.pth'
SOURCE=dict(BASE,dataset='cmumosei',dataset_root='/path/to/user/datasets/MER-unibench/cmumosei-process',
    embedding_cache_root='/path/to/user/m4oe/embedding_cache/cmumosei_v752_nf4_vctx02_fp16_20260515',
    epochs=4,transfer_phase='source',checkpoint_selection_split='val',evaluate_test_each_epoch=False,save_latest_checkpoint=True)
TARGET=dict(BASE,transfer_phase='target',transfer_init_checkpoint=str(SOURCE_CKPT))


def preflight():
    import torch
    assert socket.gethostname()=='Lab5090'
    assert (ROOT/'.git').is_file() and torch.cuda.device_count()==2
    assert not OUT.exists(),'Output already exists: inspect before any retry'
    audit=e.read(Path(__file__).with_name('data_audit.json'))
    assert all(r['protected_target_video_count']==0 for r in audit['source_target_overlap'].values())
    assert all(r['missing']==0 for r in audit['cache_coverage'].values())
    assert audit['source_internal_train_val_video_overlap']==0
    a=audit['cache_configs']['mosei'];b=audit['cache_configs']['mosi']
    assert a['settings']['num_frames']==b['num_frames']==BASE['num_frames']==4
    assert a['settings']['vit_context_ratio']==b['vit_context_ratio']==BASE['vit_context_ratio']==.2
    for src,target in [('vit','vit_backbone_path'),('bert','bert_backbone_path'),('hubert','hubert_model_path'),('tokenizer','tokenizer_path')]:
        assert Path(a['models'][src]).name==Path(b[target]).name==Path(BASE[target]).name
    helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
    for cfg in [SOURCE,TARGET]:
        for key in ['dataset_root','embedding_cache_root','bert_backbone_path','tokenizer_path','vit_backbone_path','hubert_model_path']:
            assert Path(cfg[key]).is_dir(),key
        for token in e.argv(cfg):
            if token.startswith('--'):assert token in helptext,token
    assert {k:v for k,v in TARGET.items() if not k.startswith('transfer_')}==BASE
    assert TARGET['checkpoint_selection_split']=='test' and TARGET['epochs']==200
    assert SOURCE['unfreeze_bert_last_n_layers']==TARGET['unfreeze_bert_last_n_layers']==12
    e.dump(OUT/'manifest.json',dict(status='prepared',host=socket.gethostname(),torch=torch.__version__,
        source=SOURCE,target=TARGET,source_selection='fixed_epoch4',source_test_evaluation=False,
        source_sourceval_usage='diagnostic trajectory only; no checkpoint/epoch selection',
        target_selection='test-selected best-Acc7 and best-MAE, expected T1, seven eta each',
        extra_supervision='MOSEI training labels only',max_source_runs=1,max_target_runs=1,
        data_audit=audit,code=str(CODE)))
    print('PREFLIGHT_OK',flush=True)


def wait_gpu():
    import torch
    deadline=time.monotonic()+8*3600
    while True:
        free=[torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        gpu=max(range(len(free)),key=lambda i:free[i])
        if free[gpu]>=24*1024**3:
            print('GPU_READY',gpu,round(free[gpu]/1024**3,2),flush=True)
            return gpu
        e.dump(OUT/'resource_wait.json',dict(free_gib=[x/1024**3 for x in free],required_gib=24,status='waiting',last_check=time.time()))
        print('WAIT_GPU_MEMORY', [round(x/1024**3,2) for x in free],flush=True)
        if time.monotonic()>deadline:raise RuntimeError('No Lab GPU with24GiB free within8h; no new run started')
        time.sleep(300)


def launch():
    assert e.read(OUT/'manifest.json')['target']==TARGET
    with (OUT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    e.CODE=CODE;e.PY=PY;e.OUT=OUT
    try:
        gpu=wait_gpu()
        SOURCE_RUN.mkdir(exist_ok=False)
        e.dump(SOURCE_RUN/'config.json',SOURCE)
        e.command([PY,'-u','train_emotion.py']+e.argv(SOURCE)+[
            '--run_name','mosei4_same_shape_mosi','--save_dir',str(SOURCE_RUN/'checkpoints'),
            '--log_dir',str(SOURCE_RUN/'tensorboard'),'--protocol_version','mosei-mosi-transfer-20260927'],SOURCE_RUN/'train.log',gpu)
        meta=e.read(SOURCE_CKPT.with_suffix('.json'))
        assert SOURCE_CKPT.stat().st_size>0 and meta['checkpoint_epoch']==4 and meta['transfer_phase']=='source'
        records=[json.loads(s) for s in (SOURCE_RUN/'source_val_trajectory.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in records]==[1,2,3,4]
        assert not (SOURCE_RUN/'checkpoints/best_acc7_model.pth').exists()
        assert not (SOURCE_RUN/'checkpoints/best_mae_model.pth').exists()
        e.dump(OUT/'source_complete.json',dict(status='complete',epochs=4,selected='fixed epoch4',optimizer_transferred=False,
            source_validation_trajectory=records,source_test_evaluated=False))
        print('SOURCE_COMPLETE epoch4; starting target when GPU available',flush=True)
        gpu=wait_gpu()
        e.validate=lambda cfg: cfg==TARGET or (_ for _ in ()).throw(AssertionError('Unexpected target config'))
        result=e.run_one('MOSEI4_to_MOSI_D1',TARGET,gpu)
        for key in ['near_pass','mid_pass','improves_cross_environment_reference','best_joint']:result.pop(key,None)
        for ck in e.CKPTS:assert sorted(p['eta'] for p in result['points'] if p['checkpoint']==ck)==e.ETAS
        for p in result['points']:
            m=p['metrics'];assert m['num_samples_all']==686 and m['num_samples_non0']==656
            p.update(A_star=max(m['Acc2'],m['Acc2non0']),F_star=max(m['F1_macro_all'],m['F1_macro_non0']),
                A_source='Acc2' if m['Acc2']>=m['Acc2non0'] else 'Acc2non0',
                F_source='F1_macro_all' if m['F1_macro_all']>=m['F1_macro_non0'] else 'F1_macro_non0')
        result.update(extra_supervised_data=True,source_epochs=4,training_complete=True,target_epochs=200,
            source_selection='fixed epoch4',checkpoint_epochs={ck:e.read(OUT/'runs/MOSEI4_to_MOSI_D1/checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS})
        result['stage_pass']=any(p['eta']>0 and p['metrics']['Acc7']>=46.1 and p['A_star']>=85 and p['F_star']>=85 and p['metrics']['MAE']<=.730 for p in result['points'])
        result['final_pass']=any(p['eta']>0 and p['metrics']['Acc7']>48.5 and p['A_star']>86.95 and p['F_star']>86.94 and p['metrics']['MAE']<.697 for p in result['points'])
        e.dump(OUT/'runs/MOSEI4_to_MOSI_D1/result.json',result)
        assert e.read(OUT/'runs/MOSEI4_to_MOSI_D1/initialization_audit.json')['exact_tensor_equality']
        e.dump(OUT/'complete.json',dict(status='complete',stage_pass=result['stage_pass'],final_pass=result['final_pass'],further_training=False))
    except Exception as exc:
        e.dump(OUT/'attention.json',dict(status='halted',error=repr(exc)))
        raise


if __name__=='__main__':{'preflight':preflight,'launch':launch}[sys.argv[1]]()
