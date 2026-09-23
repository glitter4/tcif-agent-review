"""Run no-update MOSI gradient probes on two assigned Lab GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import runner

ROOT=Path('/path/to/user/workspaces/m4oe-tcif-gradprobe-20260923')
OUT=Path('/path/to/user/m4oe/tcif_mosi_gradprobe_20260923_r2')
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
PARENTS={
 'D1':Path('/path/to/user/m4oe/tcif_literature_five_20260918/runs/D1'),
 'R1':Path('/path/to/user/m4oe/tcif_mosi_distribution_20260922/runs/R1_router_temp015')}
CASES={
 'D1_E_FULL':('D1','best_acc7_model','diag_full','eval',123),
 'D1_E_DETACH':('D1','best_acc7_model','diag_detach','eval',123),
 'D1_T123_FULL':('D1','best_acc7_model','diag_full','train',123),
 'D1_T123_DETACH':('D1','best_acc7_model','diag_detach','train',123),
 'D1_T124_FULL':('D1','best_acc7_model','diag_full','train',124),
 'D1_T124_DETACH':('D1','best_acc7_model','diag_detach','train',124),
 'R1_E58':('R1','best_acc7_model','diag_full','eval',123),
 'R1_T58_123':('R1','best_acc7_model','diag_full','train',123),
 'R1_T58_124':('R1','best_acc7_model','diag_full','train',124),
 'R1_E149':('R1','best_mae_model','diag_full','eval',123),
 'R1_T149_123':('R1','best_mae_model','diag_full','train',123),
 'R1_T149_124':('R1','best_mae_model','diag_full','train',124)}
LANES=[['D1_E_FULL','D1_T123_FULL','D1_T124_FULL','R1_E58','R1_T58_123','R1_T58_124'],
       ['D1_E_DETACH','D1_T123_DETACH','D1_T124_DETACH','R1_E149','R1_T149_123','R1_T149_124']]


def preflight():
    import torch
    assert socket.gethostname()=='lab-host'
    assert not OUT.exists()
    assert torch.cuda.device_count()==2
    assert (ROOT/'.git').is_file()
    for name,(parent,ck,*_) in CASES.items():
        source=PARENTS[parent]/'checkpoints'/(ck+'.pth')
        assert source.stat().st_size>0,(name,source)
        cfg=runner.read(PARENTS[parent]/'config.json')
        assert cfg['dataset']=='cmumosi' and cfg['seed']==123
        assert cfg['batch_size']==16 and cfg['grad_accum_steps']==2 and cfg['num_workers']==0
        assert cfg['checkpoint_selection_split']=='test'
        for key in ['dataset_root','embedding_cache_root','bert_backbone_path','vit_backbone_path']:
            assert Path(cfg[key]).is_dir(),key
    OUT.mkdir(parents=True)
    runner.dump(OUT/'manifest.json',dict(status='prepared',host=socket.gethostname(),cases=CASES,
        effective_batches=4,microbatch_size=16,optimizer_steps=0,
        selection='first 8 DataLoader batches with fixed seed123 training order; no test error selection',
        dropout_modes=['eval','train mask seed123','train mask seed124']))
    print('PREFLIGHT_OK',flush=True)


def lane(index):
    results=[]
    for name in LANES[index]:
        parent,ck,mode,view,maskseed=CASES[name]
        cfg=runner.read(PARENTS[parent]/'config.json')
        ckpt=PARENTS[parent]/'checkpoints'/(ck+'.pth')
        target=OUT/'runs'/name
        cfg=dict(cfg,save_dir=str(target/'checkpoints'))
        command=[PY,'-u',str(ROOT/'server-code/train_emotion.py')]+runner.argv({k:v for k,v in cfg.items() if k!='save_dir'})+[
            '--run_name','mosi_gradprobe_'+name,'--save_dir',str(target/'checkpoints'),
            '--log_dir',str(target/'tensorboard'),'--protocol_version','mosi-gradprobe-no-update-20260923']
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(index),STUDY_MODE=mode,STUDY_INIT=str(ckpt),
            STUDY_DIAG='1',STUDY_DIAG_VIEW=view,STUDY_DIAG_MASK_SEED=str(maskseed),
            STUDY_PROBE_OUT=str(target),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        target.parent.mkdir(parents=True,exist_ok=True)
        with (OUT/(name+'.log')).open('x') as f:
            result=subprocess.run(command,cwd=ROOT/'server-code',env=env,stdout=f,stderr=subprocess.STDOUT)
        status=dict(name=name,exit_code=result.returncode,result_exists=(target/'result.json').is_file())
        runner.dump(OUT/(name+'_status.json'),status)
        print('CASE_DONE',status,flush=True)
        results.append(status)
        if result.returncode!=0 or not status['result_exists']:
            break
    return results


if __name__=='__main__':
    preflight()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(lane,0);b=pool.submit(lane,1)
        results=a.result()+b.result()
    runner.dump(OUT/'complete.json',dict(status='complete' if all(x['exit_code']==0 and x['result_exists'] for x in results) and len(results)==len(CASES) else 'attention',results=results))
