"""One full detach trial plus two matched ten-epoch continuation arms."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import torch
import runner as e
from collect import ROOT,OUT,PY,SOURCES,CKPTS
BASE=e.read(SOURCES['D1']/'config.json')
CONFIGS={'DETACH':dict(BASE),
         'HEAD10':dict(BASE,epochs=10,lr_warmup_epochs=0,lr_min_factor=1.),
         'FULL10':dict(BASE,epochs=10,lr_warmup_epochs=0,lr_min_factor=1.)}
MODES={'DETACH':'detach','HEAD10':'head_only','FULL10':'full_continue'}


def worker(name):
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    e.CODE=ROOT/'server-code';e.OUT=OUT/'training';e.PY=PY
    e.validate=lambda cfg: cfg in CONFIGS.values() or (_ for _ in ()).throw(AssertionError('Unexpected config'))
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    assert torch.cuda.mem_get_info(0)[0]>=24*1024**3
    result=e.run_one(name,CONFIGS[name],gpu)
    result.pop('improves_cross_environment_reference',None)
    result.update(study_mode=MODES[name],training_complete=True,
        epochs_completed=CONFIGS[name]['epochs'],optimizer_resume=False)
    run=e.OUT/'runs'/name
    valrun=run/'validation_protocol';valrun.mkdir()
    (valrun/'checkpoints').symlink_to(run/'validation_checkpoints',target_is_directory=True)
    points=e.evaluate(valrun,dict(CONFIGS[name],checkpoint_selection_split='val'),gpu)
    # Separate validation-selected checkpoints and eta, never select with test metrics.
    options=[]
    for ck in CKPTS:
        for eta in e.ETAS:
            file=valrun/'eta_expected'/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/'val_results.json'
            obj=e.read(file);metrics=obj['metrics']
            options.append(dict(checkpoint=ck,eta=eta,val_mae=metrics['mae'],val_acc7=metrics['acc7']))
    chosen=min(options,key=lambda x:(x['val_mae'],-x['val_acc7'],x['checkpoint'],x['eta']))
    e.dump(valrun/'selection.json',chosen)
    e.dump(valrun/'result.json',dict(points=points,selected=chosen,
        caveat='HEAD10/FULL10 initialize from a test-selected D1 parent; not independent holdout confirmation'))
    result['checkpoint_epochs']={ck:e.read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in CKPTS}
    e.dump(run/'result.json',result)
    # Head-only arm must preserve every non-classifier tensor, including buffers.
    if name=='HEAD10':
        parent=torch.load(SOURCES['D1']/'checkpoints/best_acc7_model.pth',map_location='cpu',weights_only=False,mmap=True)
        checks={}
        for folder in ['checkpoints','validation_checkpoints']:
            for ck in CKPTS:
                state=torch.load(run/folder/(ck+'.pth'),map_location='cpu',weights_only=False,mmap=True)
                differences=[k for k in parent if not k.startswith('head_cls7.') and not torch.equal(parent[k],state[k])]
                assert not differences,differences
                checks[folder+'/'+ck]='all non-classifier tensors exactly unchanged'
        e.dump(run/'frozen_tensor_audit.json',checks)
    e.dump(run/'study_complete.json',dict(status='complete',mode=MODES[name]))


def launch():
    assert e.read(OUT/'reconstruction_audit.json')['status']=='passed'
    root=OUT/'training';root.mkdir(exist_ok=False)
    e.dump(root/'manifest.json',dict(configs=CONFIGS,modes=MODES,head_learning_rate=2.25e-5,
        short_schedule='constant LR, fresh optimizer, identical batch budget and seed',
        reference='D1 epoch59; test-selected parent',maximum_training_runs=3))
    def run(name,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),STUDY_MODE=MODES[name],OMP_NUM_THREADS='4',
            STUDY_INIT=str(SOURCES['D1']/'checkpoints/best_acc7_model.pth'),TORCH_SHOW_CPP_STACKTRACES='1')
        if name=='DETACH':env.pop('STUDY_INIT')
        with (root/(name+'_controller.log')).open('w') as f:
            proc=subprocess.run([PY,'-u',__file__,'worker',name],env=env,stdout=f,stderr=subprocess.STDOUT)
        status=dict(name=name,exit_code=proc.returncode)
        e.dump(root/(name+'_status.json'),status)
        return status
    def short_lane():return [run('HEAD10',1),run('FULL10',1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(run,'DETACH',0);b=pool.submit(short_lane)
        results=[a.result()]+b.result()
    e.dump(root/'complete.json',dict(results=results,further_submission=False))


if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2])
    else:launch()
