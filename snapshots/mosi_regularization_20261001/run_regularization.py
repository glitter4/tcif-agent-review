"""Three bounded independent studies. C0 reused; EMA raw is a same-run control."""
import json,os,queue,resource,socket,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runner as e
from protocol_helpers import complete_points

ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'server-code';WORK=Path(__file__).parent
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
OUT=Path('/path/to/user/m4oe/tcif_mosi_regularization_20261001')
C0=Path('/path/to/user/m4oe/tcif_mosei_to_mosi_20260927/runs/MOSEI4_to_MOSI_D1')
NAMES=['EMA','NEIGHBOR_DROP','PLATEAU']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):e.dump(p,x)
def configure():e.CODE=CODE;e.OUT=OUT;e.PY=PY;e.WORK=WORK

def preflight():
    import torch,numpy as np
    assert socket.gethostname()=='lab-gpu-host' and (ROOT/'.git').is_file()
    base=read(C0/'config.json');assert read(C0/'result.json')['training_complete']
    source=Path(base['transfer_init_checkpoint']);assert source.exists() and read(source.with_suffix('.json'))['checkpoint_epoch']==4
    permitted={'EMA':{'enable_ema','ema_decay'},'NEIGHBOR_DROP':{'neighbor_dropout'},'PLATEAU':{'scheduler','plateau_monitor','plateau_factor','plateau_patience','plateau_relative_min_factor'}}
    helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
    for name in NAMES:
        c=read(WORK/'configs'/(name+'.json'))
        assert {k for k,v in c.items() if base.get(k)!=v}==permitted[name]
        assert c['epochs']==200 and c['early_stop_patience']==201 and c['checkpoint_selection_split']=='test'
        assert all(t in helptext for t in e.argv(c) if t.startswith('--'))
    labels=np.load(Path(base['dataset_root'])/'label.npz',allow_pickle=True)
    for split,n in [('train',1284),('val',229),('test',686)]:
        ids=set(map(str,labels[split+'_corpus'].item()));assert len(ids)==n
        cached=set()
        for f in (Path(base['embedding_cache_root'])/split).glob('*/ids.json'):
            if (f.parent/'done.json').exists():cached.update(map(str,read(f)))
        assert ids<=cached
    dump(OUT/'preflight.json',dict(status='passed',names=NAMES,epochs=200,seed=123,source_epoch=4,
        selection='test-best legacyAcc7/MAE, eta.85; every selected checkpoint separate7eta',
        expected_EMA_updates=(200-1)*41,torch=torch.__version__,no_automatic_expansion=True))
    print('PREFLIGHT_PASS',flush=True)

def export_branches(name,run,gpu):
    for ck in e.CKPTS:
        root=OUT/'exports'/name/ck;root.mkdir(parents=True,exist_ok=False)
        links=root/'weights';links.mkdir()
        for suffix in ['.pth','.json']:(links/(ck+suffix)).symlink_to(run/'checkpoints'/(ck+suffix))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',STUDY_CODE=str(CODE),STUDY_EXPORT=str(root))
        command=[PY,'-u',str(WORK/'export_outputs.py'),'--checkpoints_root',str(links),'--results_root',str(root/'eval'),'--default_dataset','cmumosi',
            '--default_dataset_root',read(C0/'config.json')['dataset_root'],'--classification_readout','expected','--final_pred_eta','.8','--num_workers','0']
        with (root/'export.log').open('w') as f:subprocess.run(command,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def result_for(run,cfg,name,gpu):
    e.evaluate(run,cfg,gpu);points,val=complete_points(run)
    r=dict(id=name,config=cfg,points=points,validation_points=val,training_complete=True,target_epochs=200,
        extra_supervised_data='MOSEI source4 reused',checkpoint_epochs={ck:read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS})
    dump(run/'result.json',r);return r

def worker(name,gpu):
    import torch
    configure();deadline=time.monotonic()+12*3600
    while torch.cuda.mem_get_info(0)[0]<27*1024**3:
        if time.monotonic()>deadline:raise RuntimeError('GPU free memory insufficient for12h')
        time.sleep(300)
    cfg=read(WORK/'configs'/(name+'.json'));run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    with (run/'started.json').open('x') as f:json.dump(dict(name=name,gpu=gpu),f)
    dump(run/'config.json',cfg)
    e.command([PY,'-u','train_emotion.py']+e.argv(cfg)+['--run_name','mosi_regularization_'+name,'--protocol_version','mosi-regularization-20261001',
        '--save_dir',str(run/'checkpoints'),'--log_dir',str(run/'tensorboard')],run/'train.log',gpu)
    trace=[json.loads(x) for x in (run/'optimizer_updates.jsonl').read_text().splitlines()]
    assert len(trace)==8200 and trace[-1]['epoch']==200
    result_for(run,cfg,name+'_RAW' if name=='EMA' else name,gpu)
    if name=='EMA':
        entries=[json.loads(x) for x in (run/'ema_epochs.jsonl').read_text().splitlines()]
        assert len(entries)==200 and entries[-1]['updates']==8159
        dump(run/'ema/config.json',cfg)
        result_for(run/'ema',cfg,'EMA_AVERAGED',gpu)
    if name=='NEIGHBOR_DROP':
        records=[json.loads(x) for x in (run/'neighbor_dropout.jsonl').read_text().splitlines()]
        assert len(records)==16200 and all(x['valid_before']-x['dropped']==x['valid_after'] for x in records)
        export_branches('NEIGHBOR_DROP',run,gpu);export_branches('C0',C0,gpu)
        e.command([PY,str(WORK/'group_analysis.py'),str(OUT)],run/'group_analysis.log',gpu)
    dump(run/'study_complete.json',dict(status='complete',epochs=200,updates=8200,checkpoint_count=4 if name=='EMA' else 2))

def launch():
    assert read(OUT/'preflight.json')['status']=='passed';_,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    with (OUT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    q=queue.Queue()
    for n in NAMES:q.put(n)
    results=[]
    def lane(gpu):
        while True:
            try:n=q.get_nowait()
            except queue.Empty:return
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            with (OUT/(n+'_controller.log')).open('x') as f:r=subprocess.run([PY,'-u',__file__,'worker',n,str(gpu)],env=env,stdout=f,stderr=subprocess.STDOUT)
            status=dict(run=n,exit_code=r.returncode,gpu=gpu);dump(OUT/(n+'_status.json'),status);results.append(status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(lane,g) for g in [0,1]]
        for f in fs:f.result()
    dump(OUT/'complete.json',dict(status='complete' if all(r['exit_code']==0 for r in results) else 'completed_with_failures',results=results))
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'launch':launch}[sys.argv[1]]()
