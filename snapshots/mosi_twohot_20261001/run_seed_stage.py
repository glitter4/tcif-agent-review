"""Fixed four target-seed runs selected only by the preregistered validation gate."""
import json,os,queue,resource,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import run_twohot as r
OUT=r.OUT;WORK=Path(__file__).parent
def prepare():
    assert r.read(OUT/'initial_complete.json')['status']=='complete'
    decision=r.read(OUT/'twohot_expansion_decision.json')
    # This stage is the already-decided failed-twohot branch. Passing case needs V4 first.
    assert decision['twohot_expansion'] is False
    bases={'V1_TCIF_DROP':r.read(Path('/path/to/user/m4oe/tcif_mosi_regularization_20261001/runs/NEIGHBOR_DROP/config.json')),
           'V2_STANDARD_DROP':r.read(OUT/'runs/V2_STANDARD_DROP/config.json')}
    names=[]
    for seed in [40,41]:
        for arm,base in bases.items():
            name=arm+'_S'+str(seed);cfg=dict(base,seed=seed)
            assert {k for k in cfg if cfg[k]!=base[k]}=={'seed'}
            assert cfg['epochs']==200 and cfg['neighbor_dropout']==.2 and cfg['checkpoint_selection_split']=='test'
            assert cfg.get('cls7_soft_target','distance')=='distance'
            assert not (OUT/'runs'/name/'started.json').exists()
            r.dump(WORK/'configs'/(name+'.json'),cfg);names.append(name)
    r.dump(OUT/'seed_stage_plan.json',dict(status='prepared',validation_gate=False,no_V4=True,names=names,seeds=[40,41],
        comparison='V1 vs V2, same Lab environment, original source4 seed123 fixed',new_training_runs=4,early_stop=False))
    print('SEED_STAGE_PREPARED',names)
def launch():
    plan=r.read(OUT/'seed_stage_plan.json');assert len(plan['names'])==4 and plan['validation_gate'] is False
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    with (OUT/'seed_stage.lock').open('x') as f:f.write(str(os.getpid()))
    tasks=queue.Queue()
    for n in plan['names']:tasks.put(n)
    statuses=[]
    def lane(gpu):
        while True:
            try:n=tasks.get_nowait()
            except queue.Empty:return
            with (OUT/(n+'_controller.log')).open('x') as f:
                p=subprocess.run([r.PY,'-u',str(WORK/'run_twohot.py'),'worker',n,str(gpu)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4'),stdout=f,stderr=subprocess.STDOUT)
            status=dict(run=n,exit_code=p.returncode,gpu=gpu);r.dump(OUT/(n+'_status.json'),status);statuses.append(status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(lane,g) for g in [0,1]]
        for j in jobs:j.result()
    r.dump(OUT/'seed_stage_complete.json',dict(status='complete' if all(s['exit_code']==0 for s in statuses) else 'completed_with_failures',results=statuses,
        next='audit all3seeds and paired video bootstrap; evaluate fixed weak-reweight trigger; do not stop whole task here'))
if __name__=='__main__':{'prepare':prepare,'launch':launch}[sys.argv[1]]()
