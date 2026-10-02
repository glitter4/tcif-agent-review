"""Standard-library audit of completed c1 process records; no retraining or hashing."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/mosi_c1_process_20261002'
NAMES=['REWEIGHT1P5','C1_V1_S40','C1_V1_S41'];ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def validate():
    report={};receipts=read(BASE/'completion_receipts.json')
    for n in NAMES:
        p=BASE/'runs'/n;r=read(p/'result.json');cfg=read(p/'config.json');init=read(p/'initialization_audit.json')
        assert receipts[n]['status']=='complete' and receipts[n]['exit_code']==0
        assert r['training_complete'] and r['target_epochs']==200 and cfg['checkpoint_selection_split']=='test'
        assert init['loaded_tensor_count']==795 and init['exact_tensor_equality'] and init['source_epoch']==4 and not init['optimizer_restored']
        updates=lines(p/'optimizer_updates.jsonl');order=lines(p/'data_order.jsonl');masks=lines(p/'neighbor_dropout.jsonl');schedule=lines(p/'scheduler_epochs.jsonl')
        epochs=lines(p/'epoch_metrics.jsonl');lambdas=lines(p/'lambda_history.jsonl')
        assert len(updates)==8200 and len(order)==len(masks)==16200 and len(schedule)==len(epochs)==len(lambdas)==200
        assert [x['optimizer_step'] for x in updates]==list(range(1,8201))
        assert [x['epoch'] for x in epochs]==[x['epoch'] for x in lambdas]==list(range(1,201))
        samples=[]
        for ep in range(1,201):
            batches=order[(ep-1)*81:ep*81];steps=updates[(ep-1)*41:ep*41]
            assert [x['micro_step'] for x in batches]==list(range(1,82)) and all(x['epoch']==ep for x in batches)
            assert [len(x['positions']) for x in batches]==[16]*80+[4]
            assert sorted(i for x in batches for i in x['positions'])==list(range(1284))
            assert all(x['epoch']==ep and x['lr']==schedule[ep-1]['before'] for x in steps)
            assert schedule[ep-1]['epoch']==ep
            if ep>1:assert schedule[ep-2]['after']==schedule[ep-1]['before']
            assert 'first_update_of_epoch' in steps[0];samples.append(steps[0]['first_update_of_epoch'])
            assert samples[-1]['backbone']['active_elements']==0 and samples[-1]['backbone']['update_l2']==0
        assert all(x['clipped']==(x['preclip_global_norm']>1) and math.isfinite(x['preclip_global_norm']) for x in updates)
        for i,x in enumerate(masks):
            assert x['epoch']==i//81+1 and x['micro_step']==i%81+1
            assert x['valid_before']-x['dropped']==x['valid_after'] and 0<=x['dropped']<=x['valid_before']<=2*x['n']
        summary=read(p/'checkpoints/checkpoint_selection_summary.json');assert summary['policy']['selection_split']=='test'
        for ck,key in [('best_acc7_model','best_test_acc7'),('best_mae_model','best_test_mae')]:
            meta=read(p/'checkpoints'/(ck+'.json'))
            assert meta['checkpoint_epoch']==r['checkpoint_epochs'][ck]==summary[key]['epoch']
        for split,nall,nz in [('points',686,656),('validation_points',229,216)]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:assert sorted(x['eta'] for x in r[split] if x['checkpoint']==ck)==ETAS
            for x in r[split]:
                m=x['metrics'];assert m['num_samples_all']==nall and m['num_samples_non0']==nz
                assert all(math.isfinite(v) for v in m.values()) and x['readout']=='expected' and x['T']==1
                assert x['A_star']==max(m['Acc2'],m['Acc2non0'])==m[x['A_source']]
                assert x['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[x['F_source']]
        report[n]=dict(epochs=200,updates=8200,microbatches=16200,center_exposures=256800,
            clip_fraction=sum(x['clipped'] for x in updates)/len(updates),mean_preclip_norm=statistics.mean(x['preclip_global_norm'] for x in updates),
            neighbor_drop_fraction=sum(x['dropped'] for x in masks)/sum(x['valid_before'] for x in masks),
            mean_sampled_relative_update={g:statistics.mean(x[g]['relative_update_l2'] for x in samples) for g in samples[0]},
            checkpoint_epochs=r['checkpoint_epochs'],epoch_curve_precision='printed rounded values; not exact checkpoint selection proof')
    p=BASE/'runs/REWEIGHT1P5';rows=lines(p/'weak_main_batches.jsonl');meta=read(p/'weak_weighting.json');mean=meta['mean_weight']
    assert len(rows)==16200 and meta['normalization_split']=='train' and meta['gate_loss_weight']==.05
    assert meta['training_n']==1284 and meta['weak_count']==172 and abs(mean-(1+.5*172/1284))<1e-12
    for ep in range(1,201):
        batch=rows[(ep-1)*81:ep*81]
        assert all(x['epoch']==ep for x in batch)
        assert sum(x['n'] for x in batch)==1284
        assert sum(x['weak_positive'] for x in batch)==84 and sum(x['weak_negative'] for x in batch)==88 and sum(x['zero'] for x in batch)==53
        assert abs(sum(x['weight_sum'] for x in batch)-1284)<.002
        for x in batch:
            nw=x['weak_positive']+x['weak_negative'];assert abs(x['weight_sum']-(x['n']+.5*nw)/mean)<1e-5
            if nw==0:
                assert abs(x['weighted_reg']-x['original_reg']/mean)<1e-5
                assert abs(x['weighted_cls']-x['original_cls']/mean)<1e-5
    report['REWEIGHT1P5']['weighting_verified']=dict(mean=mean,weak_per_epoch=172,zero_per_epoch=53,training_mean_normalized=True)
    assert len(read(BASE/'reweight_analysis.json')['rows'])==28
    for folder in [BASE,ROOT/'snapshots/mosi_c1_process_20261002']:
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            assert p.suffix not in ['.pth','.pt','.npz','.npy','.gz','.log']
            s=p.read_text(encoding='utf-8');assert not any(t in s for t in ['/home/admin123','/data/user/hd57166','C:/Users/','C:\\Users\\'])
    return report
if __name__=='__main__':
    print(json.dumps(validate(),indent=2));print('PASS3x200 epochs/8200 updates/16200 batches,42test+42val,checkpoint provenance,mask and weighting logs')
