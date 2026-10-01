"""Initial V2/V3 only. Later stages require the saved validation gate and fixed seed plan."""
import json,os,resource,socket,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runner as e
from protocol_helpers import complete_points
ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'server-code';WORK=Path(__file__).parent
OUT=Path('/path/to/user/m4oe/tcif_mosi_twohot_20261001');PY='/path/to/user/envs/m4oe-lab5090/bin/python'
NAMES=['V2_STANDARD_DROP','V3_TCIF_TWOHOT']
MIN_FREE_GIB=25
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):e.dump(p,x)
def configure():e.CODE=CODE;e.OUT=OUT;e.PY=PY;e.WORK=WORK
def preflight():
    import numpy as np
    assert socket.gethostname()=='lab-gpu-host' and (ROOT/'.git').is_file()
    helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
    for n in NAMES:
        cfg=read(WORK/'configs'/(n+'.json'));source=Path(cfg['transfer_init_checkpoint']);meta=read(source.with_suffix('.json'))
        assert source.is_file() and meta['checkpoint_epoch']==4 and meta['dataset']=='cmumosei'
        assert meta.get('context_fusion','tcif')==cfg.get('context_fusion','tcif')
        assert cfg['checkpoint_selection_split']=='test' and cfg['epochs']==200 and cfg['seed']==123 and cfg['neighbor_dropout']==.2
        assert all(t in helptext for t in e.argv(cfg) if t.startswith('--'))
        labels=np.load(Path(cfg['dataset_root'])/'label.npz',allow_pickle=True)
        for s,k in [('train',1284),('val',229),('test',686)]:assert len(labels[s+'_corpus'].item())==k
    dump(OUT/'preflight.json',dict(status='passed',runs=NAMES,source_epoch=4,selection='test-selected',extra_source_training=False))
    print('PREFLIGHT_PASS')
def export(name,run,gpu):
    for ck in e.CKPTS:
        root=OUT/'exports'/name/ck;root.mkdir(parents=True,exist_ok=False);links=root/'weights';links.mkdir()
        for suffix in ['.pth','.json']:(links/(ck+suffix)).symlink_to(run/'checkpoints'/(ck+suffix))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',STUDY_CODE=str(CODE),STUDY_EXPORT=str(root))
        cmd=[PY,'-u',str(WORK/'export_outputs.py'),'--checkpoints_root',str(links),'--results_root',str(root/'eval'),'--default_dataset','cmumosi',
             '--classification_readout','expected','--final_pred_eta','.4','--num_workers','0']
        with (root/'export.log').open('w') as f:subprocess.run(cmd,cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def worker(name,gpu):
    dispatch=OUT/'external_dispatch.json'
    if dispatch.exists() and name in read(dispatch).get('runs',{}):
        # A queued local slot has been explicitly moved to another host.
        # Do not allocate a GPU or claim local training; wait for audited return receipt.
        deadline=time.monotonic()+24*3600
        while True:
            entry=read(dispatch).get('runs',{}).get(name)
            if entry is None:
                # User withdrew the dispatch while the local slot was waiting.
                break
            if entry.get('finalized'):
                assert (OUT/'runs'/name/'result.json').exists()
                dump(OUT/(name+'_external_receipt.json'),entry)
                return
            if entry.get('failed'):raise RuntimeError('External training failed; preserve remote failure, no local retry')
            if time.monotonic()>deadline:raise RuntimeError('External dispatch receipt pending24h; no local training attempted')
            time.sleep(300)
    import torch
    configure();deadline=time.monotonic()+12*3600
    while torch.cuda.mem_get_info(0)[0]<MIN_FREE_GIB*1024**3:
        if time.monotonic()>deadline:raise RuntimeError('GPU unavailable12h')
        time.sleep(300)
    cfg=read(WORK/'configs'/(name+'.json'));run=OUT/'runs'/name;run.mkdir(parents=True,exist_ok=True)
    with (run/'started.json').open('x') as f:json.dump(dict(name=name,gpu=gpu),f)
    dump(run/'config.json',cfg)
    e.command([PY,'-u','train_emotion.py']+e.argv(cfg)+['--run_name','mosi_twohot_'+name,'--protocol_version','mosi-twohot-20261001',
        '--save_dir',str(run/'checkpoints'),'--log_dir',str(run/'tensorboard')],run/'train.log',gpu)
    tr=[json.loads(s) for s in (run/'optimizer_updates.jsonl').read_text().splitlines()];assert len(tr)==8200 and tr[-1]['epoch']==200
    e.evaluate(run,cfg,gpu);points,val=complete_points(run)
    dump(run/'result.json',dict(id=name,config=cfg,points=points,validation_points=val,training_complete=True,target_epochs=200,
        extra_supervised_data='corresponding MOSEI fixedepoch4',checkpoint_epochs={ck:read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS}))
    export(name,run,gpu)
    dump(run/'study_complete.json',dict(status='complete',epochs=200,updates=8200))
def launch():
    assert read(OUT/'preflight.json')['status']=='passed';_,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    with (OUT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    def one(name,gpu):
        with (OUT/(name+'_controller.log')).open('x') as f:
            r=subprocess.run([PY,'-u',__file__,'worker',name,str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
        status=dict(run=name,exit_code=r.returncode,gpu=gpu);dump(OUT/(name+'_status.json'),status);return status
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(one,n,i) for i,n in enumerate(NAMES)];statuses=[j.result() for j in jobs]
    if all(r['exit_code']==0 for r in statuses):
        subprocess.run([PY,str(WORK/'branch_analysis.py')],check=True)
    dump(OUT/'initial_complete.json',dict(status='complete' if all(r['exit_code']==0 for r in statuses) else 'completed_with_failures',results=statuses,
        next='read validation decision before matched baseline or fixedseed follow-up; no automatic new stage in this launcher'))
if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'launch':launch}[sys.argv[1]]()
