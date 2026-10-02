"""Run only the already-authorized, previously unstarted REWEIGHT1P5 trial on c1."""
import json,os,socket,subprocess,sys
from pathlib import Path
ROOT=Path('/path/to/user/workspaces/m4oe-tcif-weak-reweight-20261002')
WORK=ROOT/'.codex-jobs/twohot';sys.path.insert(0,str(WORK))
import run_twohot as r
OUT=Path('/path/to/user/m4oe/c1_runs/tcif_weak_reweight_20261002')
PY='/path/to/user/.conda/envs/m4oe/bin/python'
NAME='REWEIGHT1P5'
def configure():
    r.ROOT=ROOT;r.CODE=ROOT/'server-code';r.WORK=WORK;r.OUT=OUT;r.PY=PY;r.MIN_FREE_GIB=20
def preflight():
    import numpy as np,torch
    configure();assert socket.gethostname()=='ln301'
    decision=r.read(OUT/'reweight_trigger.json');assert decision['triggered'] and decision['split']=='val'
    cfg=r.read(WORK/'configs'/(NAME+'.json'))
    assert cfg['seed']==123 and cfg['epochs']==200 and cfg['weak_main_weight']==1.5 and cfg['checkpoint_selection_split']=='test'
    assert cfg['neighbor_dropout']==.2 and cfg['tcif_transition_gate_loss_weight']==.05 and cfg['cls7_loss_weight']==.75
    source=Path(cfg['transfer_init_checkpoint']);assert source.is_file() and not source.name.endswith('.partial')
    state=torch.load(str(source),map_location='cpu',weights_only=False,mmap=True)
    assert len(state)==795 and sum(v.numel() for v in state.values())==394229721
    assert all(torch.is_tensor(v) for v in state.values());del state
    meta=r.read(source.with_suffix('.json'));assert meta['checkpoint_epoch']==4 and meta['dataset']=='cmumosei'
    for key in ['dataset_root','embedding_cache_root','bert_backbone_path','vit_backbone_path','hubert_model_path','tokenizer_path']:assert Path(cfg[key]).is_dir()
    assert (Path(cfg['dataset_root'])/'transcription-engchi-polish.csv').is_file()
    labels=np.load(Path(cfg['dataset_root'])/'label.npz',allow_pickle=True);counts={}
    for s,n in [('train',1284),('val',229),('test',686)]:
        ids=set(map(str,labels[s+'_corpus'].item()));assert len(ids)==n
        present=set()
        for p in (Path(cfg['embedding_cache_root'])/s).glob('*/ids.json'):
            if (p.parent/'done.json').exists():present.update(map(str,r.read(p)))
        assert ids<=present;counts[s]=n
    helptext=subprocess.check_output([PY,str(r.CODE/'train_emotion.py'),'--help'],text=True)
    assert all(t in helptext for t in r.e.argv(cfg) if t.startswith('--'))
    r.dump(OUT/'preflight.json',dict(status='passed',target_host='c1.hpcmaster.com',torch=torch.__version__,counts=counts,source_tensors=795,
        source_elements=394229721,source_epoch=4,run=NAME,training_budget=200,selection='test-selected dual7eta',
        comparison_limit='parent V1 ran on Lab5090/PyTorch2.7.1; this trial c1/PyTorch2.1.2, not a same-environment causal comparison'))
    print('REWEIGHT_C1_PREFLIGHT_PASS')
def worker():
    configure();visible=os.environ.get('CUDA_VISIBLE_DEVICES');assert visible is not None
    r.dump(OUT/'slurm_started.json',dict(job=os.environ['SLURM_JOB_ID'],node=socket.gethostname(),cuda_visible_devices=visible))
    try:r.worker(NAME,visible)
    except BaseException as exc:
        r.dump(OUT/'status.json',dict(status='failed',error=repr(exc),job=os.environ['SLURM_JOB_ID']))
        raise
    r.dump(OUT/'status.json',dict(status='complete',exit_code=0,job=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':{'preflight':preflight,'worker':worker}[sys.argv[1]]()
