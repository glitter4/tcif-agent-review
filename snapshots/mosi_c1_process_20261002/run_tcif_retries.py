"""Only unresolved TCIF seeds40/41, explicitly authorized reruns; no standard-fusion jobs."""
import os,sys,socket
from pathlib import Path
ROOT=Path('/path/to/user/workspaces/m4oe-tcif-mosi-seed-retries-20261002')
WORK=ROOT/'.codex-jobs/twohot';sys.path.insert(0,str(WORK))
import run_twohot as r
OUT=Path('/path/to/user/m4oe/c1_runs/tcif_mosi_seed_retries_20261002')
PY='/path/to/user/.conda/envs/m4oe/bin/python'
NAMES=['C1_V1_S40','C1_V1_S41']
def configure():
    r.ROOT=ROOT;r.CODE=ROOT/'server-code';r.WORK=WORK;r.OUT=OUT;r.PY=PY;r.MIN_FREE_GIB=20
def preflight():
    import subprocess,numpy as np,torch
    configure();assert socket.gethostname()=='ln301'
    helptext=subprocess.check_output([PY,str(r.CODE/'train_emotion.py'),'--help'],text=True)
    for name,seed in zip(NAMES,[40,41]):
        cfg=r.read(WORK/'configs'/(name+'.json'))
        assert cfg['seed']==seed and cfg['epochs']==200 and cfg['neighbor_dropout']==.2
        assert cfg.get('context_fusion','tcif')=='tcif' and cfg.get('weak_main_weight',1)==1
        assert cfg.get('cls7_soft_target','distance')=='distance' and cfg['checkpoint_selection_split']=='test'
        assert cfg['tcif_enable_transition_gate'] and cfg['tcif_transition_gate_loss_weight']==.05
        source=Path(cfg['transfer_init_checkpoint']);assert source.is_file() and r.read(source.with_suffix('.json'))['checkpoint_epoch']==4
        assert all(t in helptext for t in r.e.argv(cfg) if t.startswith('--'))
        labels=np.load(Path(cfg['dataset_root'])/'label.npz',allow_pickle=True)
        for split,n in [('train',1284),('val',229),('test',686)]:
            ids=set(map(str,labels[split+'_corpus'].item()));assert len(ids)==n
            present=set()
            for p in (Path(cfg['embedding_cache_root'])/split).glob('*/ids.json'):
                if (p.parent/'done.json').exists():present.update(map(str,r.read(p)))
            assert ids<=present
    r.dump(OUT/'preflight.json',dict(status='passed',host='c1.hpcmaster.com',torch=torch.__version__,names=NAMES,
        source_epoch=4,source_training_repeated=False,known_completed_experiments_repeated=False,
        rerun_authorization='user requested rerun of experiments without obtained results; ordinary fusion skipped',
        original_Lab_status='unverified while host unreachable; any recovered original results retained separately'))
    print('TCIF_RETRIES_PREFLIGHT_PASS',NAMES)
def worker():
    configure();idx=int(os.environ['SLURM_ARRAY_TASK_ID']);assert idx in [0,1];name=NAMES[idx]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert visible is not None
    r.dump(OUT/(name+'_slurm_started.json'),dict(job=os.environ['SLURM_JOB_ID'],name=name,node=socket.gethostname(),cuda_visible_devices=visible))
    try:r.worker(name,visible)
    except BaseException as exc:
        r.dump(OUT/(name+'_status.json'),dict(status='failed',error=repr(exc),job=os.environ['SLURM_JOB_ID']))
        raise
    r.dump(OUT/(name+'_status.json'),dict(status='complete',exit_code=0,job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':{'preflight':preflight,'worker':worker}[sys.argv[1]]()
