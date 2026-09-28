"""Bounded transfer follow-up: three interventions, missing2x2 cells, compute control."""
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import math
import os
from pathlib import Path
from queue import Queue,Empty
import resource
import subprocess
import sys
import time
import runner as e

ROOT=Path(__file__).resolve().parents[2]
CODE=ROOT/'server-code'
WORK=Path(__file__).resolve().parent
OUT=Path('/path/to/user/m4oe/tcif_transfer_followup_20260928')
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
OLD=Path('/path/to/user/m4oe/tcif_mosei_to_mosi_20260927')
SOURCE=OLD/'source_mosei4/checkpoints/recovery/latest_model.pth'
REFERENCE=OLD/'runs/MOSEI4_to_MOSI_D1'
BASE=e.read(WORK/'D1_config.json')
COMMON=dict(BASE,context_fusion='tcif',transfer_phase='target',transfer_init_checkpoint=str(SOURCE))
STANDARD=dict(BASE,context_fusion='standard',tcif_enable_transition_gate=False,tcif_transition_gate_loss_weight=0.)
B_SOURCE=OUT/'source_runs/B_SOURCE/checkpoints/recovery/latest_model.pth'
C_SOURCE=OUT/'source_runs/COMPUTE_SOURCE/checkpoints/recovery/latest_model.pth'
TARGETS={
 'E1_ROUTER':dict(COMMON,router_lr=5e-5),
 'E2_S':dict(COMMON,transfer_use_s=True),
 'E3_RESET':dict(COMMON,transfer_reset_tcif=True),
 'C_T':dict(BASE,context_fusion='tcif',transfer_phase='none'),
 'B_T':dict(STANDARD,transfer_phase='none'),
 'B_ST':dict(STANDARD,transfer_phase='target',transfer_init_checkpoint=str(B_SOURCE)),
 'COMPUTE_T':dict(BASE,context_fusion='tcif',transfer_phase='target',source_compute_control=True,transfer_init_checkpoint=str(C_SOURCE))}
SOURCE_CONFIGS={
 'B_SOURCE':dict(STANDARD,dataset='cmumosei',dataset_root='/path/to/user/datasets/MER-unibench/cmumosei-process',
    embedding_cache_root='/path/to/user/m4oe/embedding_cache/cmumosei_v752_nf4_vctx02_fp16_20260515',
    epochs=4,transfer_phase='source',checkpoint_selection_split='val',evaluate_test_each_epoch=False,save_latest_checkpoint=True),
 'COMPUTE_SOURCE':dict(BASE,context_fusion='tcif',epochs=4,transfer_phase='source',source_compute_control=True,
    checkpoint_selection_split='val',evaluate_test_each_epoch=False,save_latest_checkpoint=True)}
ORDER=['E1_ROUTER','E2_S','E3_RESET','C_T','B_T','B_ST','COMPUTE_T']


def augment(points):
    for p in points:
        m=p['metrics'];p.update(A_star=max(m['Acc2'],m['Acc2non0']),F_star=max(m['F1_macro_all'],m['F1_macro_non0']),
            A_source='Acc2' if m['Acc2']>=m['Acc2non0'] else 'Acc2non0',F_source='F1_macro_all' if m['F1_macro_all']>=m['F1_macro_non0'] else 'F1_macro_non0')
    return points


def validate(cfg):assert cfg in TARGETS.values()


def preflight():
    import numpy as np
    import torch
    assert not OUT.exists(),'Existing output: inspect before resubmission'
    assert (ROOT/'.git').is_file() and torch.cuda.device_count()==2
    assert e.read(OLD/'complete.json')['status']=='complete'
    assert e.read(SOURCE.with_suffix('.json'))['checkpoint_epoch']==4
    audit=e.read(WORK/'data_audit.json')
    assert all(r['protected_target_video_count']==0 for r in audit['source_target_overlap'].values())
    assert all(r['missing']==0 for r in audit['cache_coverage'].values())
    weak=e.read(WORK/'weak_audit.json');assert len(weak)==6
    assert all(sum(x['groups'][k]['sign_errors'] for k in ['weak_pos','weak_neg'])>=30 for x in weak)
    assert TARGETS['E1_ROUTER']==dict(COMMON,router_lr=5e-5)
    assert TARGETS['E2_S']==dict(COMMON,transfer_use_s=True)
    assert TARGETS['E3_RESET']==dict(COMMON,transfer_reset_tcif=True)
    helptext=subprocess.check_output([PY,str(CODE/'train_emotion.py'),'--help'],text=True)
    for cfg in list(TARGETS.values())+list(SOURCE_CONFIGS.values()):
        for key in ['dataset_root','embedding_cache_root','bert_backbone_path','vit_backbone_path','hubert_model_path','tokenizer_path']:
            assert Path(cfg[key]).is_dir(),key
        for token in e.argv(cfg):
            if token.startswith('--'):assert token in helptext,token
    for name,root,cache,expected in [('mosi',BASE['dataset_root'],BASE['embedding_cache_root'],[1284,229,686]),
        ('mosei',SOURCE_CONFIGS['B_SOURCE']['dataset_root'],SOURCE_CONFIGS['B_SOURCE']['embedding_cache_root'],[16326,1871])]:
        labels=np.load(Path(root)/'label.npz',allow_pickle=True)
        for split,n in zip(['train','val','test'],expected):
            ids=set(map(str,labels[split+'_corpus'].item()));assert len(ids)==n
            present=set()
            for f in (Path(cache)/split).glob('*/ids.json'):
                if (f.parent/'done.json').exists():present.update(map(str,e.read(f)))
            assert ids<=present
    e.dump(OUT/'manifest.json',dict(status='prepared',code=str(CODE),target_configs=TARGETS,source_configs=SOURCE_CONFIGS,
        ordered_jobs=ORDER,max_new_target_runs=7,max_new_pretraining_runs=2,
        reused_C_ST=str(REFERENCE),reused_C_source=str(SOURCE),weak_label_audit=weak,
        reason_new_C_T='Lab D1 stopped at142; a complete200 epoch C_T is required for matched2x2',
        interaction_protocol='same checkpoint type, same eta/readout; main eta.8 for both checkpoint types',
        compute_match='2044 source updates and65304 center-example exposures exactly; also measure input tensor elements and wall time',
        source_validation_diagnostic='fixed expected T1 eta.8 only, not checkpoint selection',
        source_overlap_audit=audit,torch=torch.__version__))
    print('PREFLIGHT_OK',flush=True)


def gpu_guard():
    import torch
    deadline=time.monotonic()+12*3600
    while torch.cuda.mem_get_info(0)[0]<24*1024**3:
        if time.monotonic()>deadline:raise RuntimeError('Assigned Lab GPU lacks24GiB for12h')
        print('WAIT_GPU_MEMORY',flush=True);time.sleep(300)


def source_train(name,gpu):
    cfg=SOURCE_CONFIGS[name];run=OUT/'source_runs'/name
    run.mkdir(parents=True,exist_ok=False);e.dump(run/'config.json',cfg)
    gpu_guard()
    e.command([PY,'-u','train_emotion.py']+e.argv(cfg)+['--run_name',name,'--save_dir',str(run/'checkpoints'),
        '--log_dir',str(run/'tensorboard'),'--protocol_version','transfer-followup-source-20260928'],run/'train.log',gpu)
    ck=run/'checkpoints/recovery/latest_model.pth';assert ck.stat().st_size>0
    assert e.read(ck.with_suffix('.json'))['checkpoint_epoch']==4
    trace=[json.loads(s) for s in (run/'compute_trace.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in trace]==[1,2,3,4]
    assert sum(r['optimizer_steps'] for r in trace)==2044
    assert sum(r['examples'] for r in trace)==65304
    e.dump(run/'complete.json',dict(status='complete',source_epochs=4,optimizer_steps=2044,examples=65304,trace=trace))


def source_val(weights,output,gpu):
    output.mkdir(parents=True,exist_ok=False)
    e.command([PY,'-u','eval_all_mosei_maefixed.py','--checkpoints_root',str(weights),'--results_root',str(output),
        '--validation_only','--override_dataset','cmumosei','--override_dataset_root','/path/to/user/datasets/MER-unibench/cmumosei-process',
        '--override_embedding_cache_root','/path/to/user/m4oe/embedding_cache/cmumosei_v752_nf4_vctx02_fp16_20260515',
        '--classification_readout','expected','--final_pred_eta','.8','--num_workers','0'],output/'source_val.log',gpu)
    result={}
    for f in output.glob('*/val_results.json'):
        m=e.read(f)['metrics'];assert m['num_samples']==1871
        result[f.parent.name]={k:m[k] for k in ['num_samples','mae','acc7','acc2','binary_f1']}
    assert result
    e.dump(output/'summary.json',result)
    return result


def drift(run,source):
    import torch
    src=torch.load(source,map_location='cpu',weights_only=False,mmap=True)
    names=[k for k in src if k.startswith('shared_specific_layers.') and (k.endswith('.phi') or k.endswith('.scale'))]
    results={}
    for ck in e.CKPTS:
        target=torch.load(run/'checkpoints'/(ck+'.pth'),map_location='cpu',weights_only=False,mmap=True)
        results[ck]={}
        for n in names:
            a=src[n].double().flatten();b=target[n].double().flatten();delta=(b-a).norm()
            results[ck][n]=dict(source_norm=float(a.norm()),delta_norm=float(delta),relative_delta=float(delta/(a.norm()+1e-12)),
                cosine=float(torch.nn.functional.cosine_similarity(a,b,dim=0)))
    e.dump(run/'router_drift.json',results)


def paired_repairs(run):
    def klass(v):return math.floor(v+.5) if v>=0 else -math.floor(-v+.5)
    rows=[]
    for ck in e.CKPTS:
        for eta in e.ETAS:
            tag='eta_'+f'{eta:.1f}'.replace('.','p')
            for split in ['val','test']:
                a={r['id']:r for r in csv.DictReader((REFERENCE/'eta_expected'/tag/ck/(split+'_details.csv')).open())}
                b={r['id']:r for r in csv.DictReader((run/'eta_expected'/tag/ck/(split+'_details.csv')).open())}
                assert set(a)==set(b)
                groups={}
                for label,fn in [('weak_pos',lambda y:0<y<.5),('weak_neg',lambda y:-.5<y<0),('nonweak_nonzero',lambda y:abs(y)>=.5),('zero',lambda y:y==0)]:
                    stats=dict(n=0,sign_fixed=0,new_sign_errors=0,acc7_fixed=0,new_acc7_errors=0,sign_fixed_and_acc7_correct=0)
                    for sid,r in a.items():
                        y=float(r['true_value']);assert abs(y-float(b[sid]['true_value']))<1e-6
                        if not fn(y):continue
                        pa=float(r['pred_value']);pb=float(b[sid]['pred_value']);ca=(pa>=0)==(y>=0);cb=(pb>=0)==(y>=0)
                        ka=klass(pa)==klass(y);kb=klass(pb)==klass(y)
                        stats['n']+=1;stats['sign_fixed']+=int(not ca and cb);stats['new_sign_errors']+=int(ca and not cb)
                        stats['acc7_fixed']+=int(not ka and kb);stats['new_acc7_errors']+=int(ka and not kb)
                        stats['sign_fixed_and_acc7_correct']+=int(not ca and cb and kb)
                    groups[label]=stats
                rows.append(dict(checkpoint=ck,eta=eta,split=split,groups=groups))
    e.dump(run/'paired_repairs.json',rows)


def worker(name):
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    e.CODE=CODE;e.PY=PY;e.OUT=OUT;e.validate=validate
    gpu=os.environ['CUDA_VISIBLE_DEVICES']
    if name=='B_ST':source_train('B_SOURCE',gpu)
    if name=='COMPUTE_T':source_train('COMPUTE_SOURCE',gpu)
    gpu_guard()
    result=e.run_one(name,TARGETS[name],gpu)
    for key in ['near_pass','mid_pass','improves_cross_environment_reference','best_joint']:result.pop(key,None)
    assert len(result['points'])==14
    for ck in e.CKPTS:assert sorted(p['eta'] for p in result['points'] if p['checkpoint']==ck)==e.ETAS
    for p in result['points']:assert p['metrics']['num_samples_all']==686 and p['metrics']['num_samples_non0']==656
    result['points']=augment(result['points']);run=OUT/'runs'/name
    result.update(training_complete=True,target_epochs=200,checkpoint_epochs={ck:e.read(run/'checkpoints'/(ck+'.json'))['checkpoint_epoch'] for ck in e.CKPTS})
    trace=[json.loads(s) for s in (run/'compute_trace.jsonl').read_text().splitlines()]
    assert len(trace)==200 and sum(r['optimizer_steps'] for r in trace)==8200
    assert sum(r['examples'] for r in trace)==256800
    e.dump(run/'result.json',result)
    if 'transfer_init_checkpoint' in TARGETS[name]:drift(run,Path(TARGETS[name]['transfer_init_checkpoint']))
    if name=='E1_ROUTER':source_val(run/'checkpoints',run/'source_validation',gpu)
    if name in ['E1_ROUTER','E2_S','E3_RESET']:paired_repairs(run)
    e.dump(run/'study_complete.json',dict(status='complete',optimizer_steps=8200,examples=256800))


def finalize(gpu):
    # Existing C_ST and source4 are evaluated only on source val as additional diagnostics.
    e.CODE=CODE;e.PY=PY
    source_val(REFERENCE/'checkpoints',OUT/'reference_source_validation',gpu)
    links=OUT/'source_reference_weights';links.mkdir()
    for ext in ['.pth','.json']:(links/('best_acc7_model'+ext)).symlink_to(SOURCE.with_suffix(ext))
    source_val(links,OUT/'source_initial_validation',gpu)
    refs={'C_ST':e.read(REFERENCE/'result.json'),'C_T':e.read(OUT/'runs/C_T/result.json'),
          'B_T':e.read(OUT/'runs/B_T/result.json'),'B_ST':e.read(OUT/'runs/B_ST/result.json')}
    for r in refs.values():augment(r['points'])
    cells=[]
    for ck in e.CKPTS:
        for eta in e.ETAS:
            selected={name:next(p for p in r['points'] if p['checkpoint']==ck and p['eta']==eta) for name,r in refs.items()}
            interaction={}
            for metric in ['Acc7','MAE','A_star','F_star']:
                def q(name):
                    p=selected[name];v=p[metric] if metric.endswith('_star') else p['metrics'][metric]
                    return -v if metric=='MAE' else v
                interaction[metric]=(q('C_ST')-q('B_ST'))-(q('C_T')-q('B_T'))
            cells.append(dict(checkpoint_type=ck,eta=eta,cells=selected,interaction_Q=interaction))
    e.dump(OUT/'factorial.json',dict(rows=cells,main_eta=.8,MAE_Q='negative MAE',single_seed=True))


def launch():
    assert e.read(OUT/'manifest.json')['target_configs']==TARGETS
    with (OUT/'launch.lock').open('x') as f:f.write(str(os.getpid()))
    q=Queue()
    for name in ORDER:q.put(name)
    def lane(gpu):
        results=[]
        while True:
            try:name=q.get_nowait()
            except Empty:return results
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',TORCH_SHOW_CPP_STACKTRACES='1')
            with (OUT/(name+'_controller.log')).open('x') as f:
                proc=subprocess.run([PY,'-u',__file__,'worker',name],env=env,stdout=f,stderr=subprocess.STDOUT)
            status=dict(name=name,exit_code=proc.returncode)
            e.dump(OUT/(name+'_status.json'),status);results.append(status)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(lane,0);b=pool.submit(lane,1);results=a.result()+b.result()
    if all(r['exit_code']==0 for r in results):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4')
        with (OUT/'finalize.log').open('x') as f:
            proc=subprocess.run([PY,'-u',__file__,'finalize'],env=env,stdout=f,stderr=subprocess.STDOUT)
        e.dump(OUT/'complete.json',dict(status='complete' if proc.returncode==0 else 'needs_finalization',results=results,automatic_expansion=False))
    else:e.dump(OUT/'complete.json',dict(status='completed_with_failures',results=results,automatic_expansion=False))


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='worker':worker(sys.argv[2])
    elif mode=='finalize':finalize(os.environ['CUDA_VISIBLE_DEVICES'])
    else:{'preflight':preflight,'launch':launch}[mode]()
