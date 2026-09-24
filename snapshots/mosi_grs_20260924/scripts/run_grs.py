"""Bounded C1 MOSI G/R/S array with complete test-selected eta evaluation."""
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import runner as e

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'server-code'
OUT=Path('/path/to/user/m4oe/c1_runs/tcif_mosi_grs_20260924')
PY='/path/to/user/.conda/envs/m4oe/bin/python'
BASE=e.read(Path(__file__).with_name('D1_config.json'))
TASKS=['G','R','S']
CONFIGS={name:dict(BASE,grs_mode=name) for name in TASKS}


def target_point(p):
    m=p['metrics'];a=max(m['Acc2'],m['Acc2non0']);f=max(m['F1_macro_all'],m['F1_macro_non0'])
    return dict(p,A_star=a,F_star=f,A_source='Acc2' if m['Acc2']>=m['Acc2non0'] else 'Acc2non0',
        F_source='F1_macro_all' if m['F1_macro_all']>=m['F1_macro_non0'] else 'F1_macro_non0')


def passes(p,final=False):
    q=target_point(p);m=p['metrics']
    if p['eta']==0:return False
    if final:return m['Acc7']>48.5 and q['A_star']>86.95 and q['F_star']>86.94 and m['MAE']<.697
    return m['Acc7']>=46.1 and q['A_star']>=85 and q['F_star']>=85 and m['MAE']<=.730


def audit(points):
    assert len(points)==14
    for ck in e.CKPTS:assert sorted(p['eta'] for p in points if p['checkpoint']==ck)==e.ETAS
    for p in points:
        m=p['metrics']
        assert p['readout']=='expected' and p['T']==1
        assert m['num_samples_all']==686 and m['num_samples_non0']==656
        assert m['F1non0']==m['F1_macro_non0']
        assert all(math.isfinite(m[k]) for k in ['Acc7','MAE','Acc2','Acc2non0','F1_macro_all','F1_macro_non0'])


def validate(cfg):
    assert cfg in CONFIGS.values()
    assert {k:v for k,v in cfg.items() if k!='grs_mode'}==BASE


def preflight():
    import numpy as np
    import torch
    assert socket.gethostname()=='ln301'
    assert not OUT.exists(), 'Existing output requires inspection; no duplicate launch'
    assert BASE['dataset']=='cmumosi' and BASE['seed']==123
    assert BASE['epochs']==BASE['freeze_backbone_epochs']==200
    assert BASE['early_stop_patience']==201 and BASE['batch_size']==16 and BASE['grad_accum_steps']==2
    for key in ['dataset_root','embedding_cache_root','bert_backbone_path','tokenizer_path','vit_backbone_path','hubert_model_path']:
        assert Path(BASE[key]).is_dir(),key
    labels=np.load(Path(BASE['dataset_root'])/'label.npz',allow_pickle=True)
    for split,n in [('train',1284),('val',229),('test',686)]:
        ids=set(map(str,labels[split+'_corpus'].item()));assert len(ids)==n
        cached=set()
        for p in (Path(BASE['embedding_cache_root'])/split).glob('*/ids.json'):
            if (p.parent/'done.json').exists():cached.update(map(str,e.read(p)))
        assert ids<=cached
    helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
    for cfg in CONFIGS.values():
        validate(cfg)
        for token in e.argv(cfg):
            if token.startswith('--'):assert token in helptext,token
    e.dump(OUT/'manifest.json',dict(status='prepared',target_host='c1.hpcmaster.com',torch=torch.__version__,
        code=str(CODE),configs=CONFIGS,max_new_training_runs=3,automatic_combinations=False,
        selection='test-selected dual checkpoints; expected T1; seven eta each; eta0 diagnostic',
        A_star='max Acc2/all and nonzero from same point',F_star='max macro F1/all and nonzero from same point',
        stage_goal=dict(Acc7=46.1,A_star=85,F_star=85,MAE=.730),
        final_goal=dict(Acc7_gt=48.5,A_star_gt=86.95,F_star_gt=86.94,MAE_lt=.697),
        environment_note='Lab5090 connection timeout; use prior C1 D1 as same-environment reference, Lab D1 kept separately'))
    print('PREFLIGHT_OK',flush=True)


def worker():
    import torch
    import resource
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    task=TASKS[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    assert e.read(OUT/'manifest.json')['configs']==CONFIGS
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    # Preserve the scheduler-assigned device set; do not select another user's GPU.
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','0')
    deadline=time.monotonic()+3*3600
    while torch.cuda.mem_get_info(0)[0]<20*1024**3:
        if time.monotonic()>deadline:raise RuntimeError('Assigned GPU below20GiB free for3h; training not started')
        print('WAIT_GPU_MEMORY',task,flush=True);time.sleep(60)
    e.OUT=OUT;e.CODE=CODE;e.PY=PY;e.validate=validate
    result=e.run_one(task,CONFIGS[task],visible)
    audit(result['points'])
    for k in ['near_pass','mid_pass','improves_cross_environment_reference','best_joint']:result.pop(k,None)
    result.update(points=[target_point(p) for p in result['points']],stage_pass=any(passes(p) for p in result['points']),
        final_pass=any(passes(p,True) for p in result['points']),training_complete=True,execution_host=socket.gethostname())
    result['checkpoint_epochs']={ck:e.read(OUT/'runs'/task/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS}
    e.dump(OUT/'runs'/task/'result.json',result)
    e.dump(OUT/(task+'_complete.json'),dict(status='complete',stage_pass=result['stage_pass'],final_pass=result['final_pass']))
    print('GRS_COMPLETE',task,result['stage_pass'],result['final_pass'],flush=True)


if __name__=='__main__':{'preflight':preflight,'worker':worker}[sys.argv[1]]()
