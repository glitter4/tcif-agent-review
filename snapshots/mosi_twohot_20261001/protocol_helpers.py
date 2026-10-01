"""Exactly three authorized target runs, no retries or adaptive expansion."""
import csv,json,os,queue,resource,socket,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import runner as e
from metric_protocol import audit

ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'server-code'
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
OUT=Path('/path/to/user/m4oe/tcif_mosi_protocol_lr_20260930')
C0=Path('/path/to/user/m4oe/tcif_mosei_to_mosi_20260927/runs/MOSEI4_to_MOSI_D1')
CONFIG_DIR=Path(__file__).parent/'configs'
NAMES=['A_FUSION_LR','B_TEXT_LR','TEXT_ONLY']
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):e.dump(p,x)
def augment(p):
    m=p['metrics'];p.update(A_star=max(m['Acc2'],m['Acc2non0']),F_star=max(m['F1_macro_all'],m['F1_macro_non0']),
        A_source='Acc2' if m['Acc2']>=m['Acc2non0'] else 'Acc2non0',F_source='F1_macro_all' if m['F1_macro_all']>=m['F1_macro_non0'] else 'F1_macro_non0')
    return p
def complete_points(run):
    points=[];validation=[]
    for ck in e.CKPTS:
        for eta in e.ETAS:
            for split,dest,n in [('test',points,686),('val',validation,229)]:
                root=run/'eta_expected'/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck
                rows=list(csv.DictReader((root/(split+'_details.csv')).open()));assert len(rows)==n
                a=audit([float(r['true_value']) for r in rows],[float(r['pred_value']) for r in rows]);b=a['binary']
                m=dict(Acc7=a['Acc7_legacy'],Acc7_nearest_even=a['Acc7_nearest_even'],MAE=a['MAE'],Acc2=b['all']['Acc2'],Acc2non0=b['nonzero']['Acc2'],
                    F1_macro_all=b['all']['macro_F1'],F1_macro_non0=b['nonzero']['macro_F1'],F1_weighted_all=b['all']['weighted_F1'],F1_weighted_non0=b['nonzero']['weighted_F1'],
                    num_samples_all=b['all']['n'],num_samples_non0=b['nonzero']['n'])
                dest.append(augment(dict(checkpoint=ck,eta=eta,readout='expected',T=1,metrics=m)))
    return points,validation

def preflight():
    import numpy as np
    import torch
    assert socket.gethostname()=='lab-gpu-host' and (ROOT/'.git').is_file()
    base=read(C0/'config.json');assert read(C0/'result.json')['training_complete']
    source=Path(base['transfer_init_checkpoint']);assert source.is_file() and read(source.with_suffix('.json'))['checkpoint_epoch']==4
    for name in NAMES[:2]:
        cfg=read(CONFIG_DIR/(name+'.json'))
        diff={k for k,v in cfg.items() if v!=base[k]}
        assert diff==({'lr','bert_last_layer_lr_ratio'} if name.startswith('A_') else {'bert_last_layer_lr_ratio'})
        assert cfg['router_lr']==2e-4 and cfg['tcif_transition_gate_lr']==6e-5 and cfg['checkpoint_selection_split']=='test'
        assert cfg['seed']==123 and cfg['epochs']==200 and cfg['evaluate_test_each_epoch']
        helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
        assert all(t in helptext for t in e.argv(cfg) if t.startswith('--'))
    labels=np.load(Path(base['dataset_root'])/'label.npz',allow_pickle=True)
    for split,n in [('train',1284),('val',229),('test',686)]:
        ids=set(map(str,labels[split+'_corpus'].item()));assert len(ids)==n
        cached=set()
        for f in (Path(base['embedding_cache_root'])/split).glob('*/ids.json'):
            if (f.parent/'done.json').exists():cached.update(map(str,read(f)))
        assert ids<=cached
    dump(OUT/'preflight.json',dict(status='passed',host=socket.gethostname(),torch=torch.__version__,runs=NAMES,
        seed=123,epochs=200,source_epoch=4,selection='test-selected legacyAcc7 at eta.85; dual complete7eta; nearest-even supplementary',
        no_automatic_expansion=True))
    print('PREFLIGHT_PASS',flush=True)

def worker(name,gpu):
    import torch
    deadline=time.monotonic()+12*3600
    while torch.cuda.mem_get_info(0)[0]<24*1024**3:
        if time.monotonic()>deadline:raise RuntimeError('Assigned GPU insufficient free memory for12h')
        time.sleep(300)
    e.CODE=CODE;e.OUT=OUT;e.PY=PY
    run=OUT/'runs'/name
    if name=='TEXT_ONLY':
        cfg=read(CONFIG_DIR/'TEXT_ONLY_spec.json');run.mkdir(parents=True,exist_ok=True);dump(run/'config.json',cfg)
        e.command([PY,'-u',str(Path(__file__).with_name('text_only.py')),'--out',str(run),'--dataset-root',cfg['dataset_root'],'--backbone',cfg['backbone']],run/'train.log',gpu)
        result=read(run/'result.json')
        for p in result['points']+result['validation_points']:augment(p)
    else:
        cfg=read(CONFIG_DIR/(name+'.json'));run.mkdir(parents=True,exist_ok=True)
        with (run/'started.json').open('x') as f:json.dump(dict(name=name,gpu=gpu),f)
        dump(run/'config.json',cfg)
        e.command([PY,'-u','train_emotion.py']+e.argv(cfg)+['--run_name','mosi_protocol_'+name,'--protocol_version','mosi-protocol-lr-20260930',
            '--save_dir',str(run/'checkpoints'),'--log_dir',str(run/'tensorboard')],run/'train.log',gpu)
        trace=[json.loads(s) for s in (run/'optimizer_updates.jsonl').read_text().splitlines()]
        assert len(trace)==8200 and trace[-1]['epoch']==200
        e.evaluate(run,cfg,gpu)
        points,val=complete_points(run)
        result=dict(id=name,config=cfg,points=points,validation_points=val,training_complete=True,target_epochs=200,
            extra_supervised_data='MOSEI fixed4epochs reused',checkpoint_epochs={ck:read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS})
    dump(run/'result.json',result);dump(run/'study_complete.json',dict(status='complete',epochs=200,updates=8200))

def launch():
    assert read(OUT/'preflight.json')['status']=='passed'
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    with (OUT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    tasks=queue.Queue()
    for n in NAMES:tasks.put(n)
    statuses=[]
    def lane(gpu):
        while not tasks.empty():
            try:name=tasks.get_nowait()
            except queue.Empty:return
            with (OUT/(name+'_controller.log')).open('x') as f:
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
                r=subprocess.run([PY,'-u',__file__,'worker',name,str(gpu)],env=env,stdout=f,stderr=subprocess.STDOUT)
            status=dict(run=name,exit_code=r.returncode,gpu=gpu);dump(OUT/(name+'_status.json'),status);statuses.append(status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(lane,g) for g in [0,1]]
        for f in futures:f.result()
    dump(OUT/'complete.json',dict(status='complete' if all(r['exit_code']==0 for r in statuses) else 'completed_with_failures',results=statuses))

if __name__=='__main__':
    if sys.argv[1]=='worker':worker(sys.argv[2],int(sys.argv[3]))
    else:{'preflight':preflight,'launch':launch}[sys.argv[1]]()
