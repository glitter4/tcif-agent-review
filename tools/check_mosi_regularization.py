"""Verify all three complete runs, EMA pairing, masking, and scheduler decisions."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mosi_regularization_20261001'
RUNS=['EMA','NEIGHBOR_DROP','PLATEAU']
VIEWS=['EMA','EMA/ema','NEIGHBOR_DROP','PLATEAU']
ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def close(a,b,tol=1e-9):assert abs(a-b)<=tol,(a,b)
def validate():
    assert read(BASE/'complete.json')['status']=='complete'
    assert all(x['exit_code']==0 for x in read(BASE/'complete.json')['results'])
    audit={};orders={};c0=read(BASE/'C0_reference.json')['config']
    diffs={'EMA':{'enable_ema','ema_decay'},'NEIGHBOR_DROP':{'neighbor_dropout'},'PLATEAU':{'scheduler','plateau_monitor','plateau_factor','plateau_patience','plateau_relative_min_factor'}}
    for name in RUNS:
        path=BASE/'runs'/name;cfg=read(path/'config.json');init=read(path/'initialization_audit.json')
        assert {k for k,v in cfg.items() if c0.get(k)!=v}==diffs[name]
        assert cfg['checkpoint_selection_split']=='test' and cfg['early_stop_patience']==201
        assert init['source_epoch']==4 and init['loaded_tensor_count']==795 and init['exact_tensor_equality'] and not init['optimizer_restored']
        tr=lines(path/'optimizer_updates.jsonl');orders[name]=lines(path/'data_order.jsonl');sched=lines(path/'scheduler_epochs.jsonl')
        assert len(tr)==8200 and len(orders[name])==16200 and len(sched)==200
        assert [x['optimizer_step'] for x in tr]==list(range(1,8201))
        assert [x['epoch'] for x in sched]==list(range(1,201))
        for ep in range(1,201):
            batch=orders[name][(ep-1)*81:ep*81]
            assert [x['micro_step'] for x in batch]==list(range(1,82)) and all(x['epoch']==ep for x in batch)
            assert [len(x['positions']) for x in batch]==[16]*80+[4]
            assert sorted(i for x in batch for i in x['positions'])==list(range(1284))
            steps=tr[(ep-1)*41:ep*41];assert all(x['epoch']==ep and x['lr']==sched[ep-1]['before'] for x in steps)
            assert 'first_update_of_epoch' in steps[0]
            assert steps[0]['first_update_of_epoch']['backbone']['active_elements']==0
            for x in steps:assert math.isfinite(x['preclip_global_norm']) and x['clipped']==(x['preclip_global_norm']>1)
            if ep>1:assert sched[ep-2]['after']==sched[ep-1]['before']
        audit[name]=dict(epochs=200,updates=8200,microbatches=16200,center_exposures=256800,clip_fraction=sum(x['clipped'] for x in tr)/8200,
            mean_preclip_norm=statistics.mean(x['preclip_global_norm'] for x in tr),lr_reduction_epochs=[x['epoch'] for x in sched if x['after']!=x['before']])
    assert orders['EMA']==orders['NEIGHBOR_DROP']==orders['PLATEAU']
    ema=lines(BASE/'runs/EMA/ema_epochs.jsonl');assert len(ema)==200 and ema[0]['initialized'] and ema[0]['keys']==795 and ema[0]['updates']==0
    for ep,x in enumerate(ema[1:],2):assert x['epoch']==ep and x['updates']==41*(ep-1) and x['raw_weights_restored']
    audit['EMA']['shadow_updates']=ema[-1]['updates'];assert ema[-1]['updates']==8159
    er=read(BASE/'runs/EMA/ema/result.json')
    assert er['checkpoint_epochs']['best_acc7_model']==min(ema[1:],key=lambda x:(-x['test']['acc7'],x['test']['mae'],x['epoch']))['epoch']
    assert er['checkpoint_epochs']['best_mae_model']==min(ema[1:],key=lambda x:(x['test']['mae'],-x['test']['acc7'],x['epoch']))['epoch']
    drop=lines(BASE/'runs/NEIGHBOR_DROP/neighbor_dropout.jsonl');assert len(drop)==16200
    for i,x in enumerate(drop):
        assert x['epoch']==i//81+1 and x['micro_step']==i%81+1
        assert 0<=x['dropped']<=x['valid_before']<=2*x['n']
        assert x['valid_before']-x['dropped']==x['valid_after'] and x['no_context_before']<=x['no_context_after']<=x['n']
    audit['NEIGHBOR_DROP']['drop_fraction']=sum(x['dropped'] for x in drop)/sum(x['valid_before'] for x in drop)
    audit['NEIGHBOR_DROP']['valid_neighbors_before']=sum(x['valid_before'] for x in drop)
    audit['NEIGHBOR_DROP']['dropped_neighbors']=sum(x['dropped'] for x in drop)
    assert .19<audit['NEIGHBOR_DROP']['drop_fraction']<.21
    # Independent replay of ReduceLROnPlateau defaults: mode=min/rel threshold1e-4/patience5/factor.5/floor.03.
    sched=lines(BASE/'runs/PLATEAU/scheduler_epochs.jsonl');initial=sched[0]['before'];lr=dict(initial);best=float('inf');bad=0
    for x in sched:
        assert x['monitor']=='val_mae' and x['before']==lr
        if x['val_mae']<best*(1-1e-4):best=x['val_mae'];bad=0
        else:bad+=1
        if bad>5:lr={k:max(v*.5,initial[k]*.03) for k,v in lr.items()};bad=0
        for k in lr:close(x['after'][k],lr[k],1e-15)
    audit['PLATEAU']['final_lr']=lr
    for view in VIEWS:
        path=BASE/'runs'/view;r=read(path/'result.json');assert r['training_complete'] and r['target_epochs']==200
        for ck,ep in r['checkpoint_epochs'].items():assert ep==read(path/'checkpoints'/(ck+'.json'))['checkpoint_epoch']
        for split,n,nz in [('points',686,656),('validation_points',229,216)]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:assert sorted(p['eta'] for p in r[split] if p['checkpoint']==ck)==ETAS
            for p in r[split]:
                m=p['metrics'];assert (m['num_samples_all'],m['num_samples_non0'])==(n,nz) and p['readout']=='expected' and p['T']==1
                assert p['A_star']==max(m['Acc2'],m['Acc2non0'])==m[p['A_source']]
                assert p['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[p['F_source']]
                assert all(math.isfinite(v) for v in m.values())
    groups=read(BASE/'neighbor_group_analysis.json')['rows'];assert len(groups)==28
    r=read(BASE/'runs/NEIGHBOR_DROP/result.json')
    reference=read(BASE/'C0_reference.json')
    for row in groups:
        p=next(p for p in r['points' if row['split']=='test' else 'validation_points'] if p['checkpoint']==row['checkpoint'] and p['eta']==row['eta'])
        for direction in ['local_to_posterior','C0_to_drop']:
            m=row['groups']['all'][direction];close(m['Acc7'],p['metrics']['Acc7'],1e-5);close(m['MAE'],p['metrics']['MAE'],1e-5)
        rp=next(p for p in reference['points' if row['split']=='test' else 'validation_points'] if p['checkpoint']==row['checkpoint'] and p['eta']==row['eta'])
        m=row['groups']['all']['C0_local_to_posterior'];close(m['Acc7'],rp['metrics']['Acc7'],1e-5);close(m['MAE'],rp['metrics']['MAE'],1e-5)
    for folder in [BASE,ROOT/'snapshots/mosi_regularization_20261001']:
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            assert p.suffix not in ['.pth','.pt','.npz','.npy','.pkl','.gz']
            text=p.read_text(encoding='utf-8');assert not any(s in text for s in ['/home/admin123','/data/user/hd57166','C:/Users/','C:\\Users\\'])
    audit['data_order_equal_microbatches']=16200
    return audit
if __name__=='__main__':
    print(json.dumps(validate(),indent=2));print('PASS 56test+56val,3x200epochs,EMA8159,mask and plateau replay,paired data order and group metrics')
