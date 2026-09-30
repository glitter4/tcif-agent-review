"""Validate protocol/LR artifacts, including interrupted A; no ML dependencies."""
import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mosi_protocol_lr_20260930'
NAMES=['A_FUSION_LR','B_TEXT_LR','TEXT_ONLY']
ETAS=[0,.2,.4,.6,.8,.9,1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def close(a,b):assert abs(a-b)<1e-8,(a,b)
def validate():
    report={};orders={}
    for name in NAMES:
        run=BASE/'runs'/name;r=read(run/'result.json');tr=lines(run/'optimizer_updates.jsonl');order=lines(run/'data_order.jsonl');orders[name]=order
        interrupted=name=='A_FUSION_LR';assert r['training_complete']==(not interrupted)
        assert len(tr)==(6886 if interrupted else 8200) and tr[-1]['epoch']==(168 if interrupted else 200)
        assert len(order)==(13607 if interrupted else 16200)
        assert [x['optimizer_step'] for x in tr]==list(range(1,len(tr)+1))
        if interrupted:assert r['completed_epochs']==167 and r['evaluation_complete'] and 'SIGSEGV' in r['failure']
        for i,x in enumerate(order):
            assert x['epoch']==i//81+1 and x['micro_step']==i%81+1
            assert len(x['positions'])==(4 if i%81==80 else 16)
        for epoch in range(167 if interrupted else 200):
            assert sorted(p for x in order[epoch*81:(epoch+1)*81] for p in x['positions'])==list(range(1284))
        samples=[x for x in tr if 'first_update_of_epoch' in x]
        assert len(samples)==(168 if interrupted else 200)
        assert [x['epoch'] for x in samples]==list(range(1,len(samples)+1))
        for x in tr:
            assert math.isfinite(x['preclip_global_norm']) and x['clipped']==(x['preclip_global_norm']>1)
        for x in samples:
            for g,m in x['first_update_of_epoch'].items():
                assert all(math.isfinite(v) for v in m.values())
                if g=='backbone':assert m['active_elements']==0 and m['update_l2']==0
        if name!='TEXT_ONLY':
            config=read(run/'config.json');base=read(BASE/'C0_reference.json')['config'] if (BASE/'C0_reference.json').exists() else None
            if base is not None:
                assert {k for k in config if config[k]!=base[k]}==({'lr','bert_last_layer_lr_ratio'} if interrupted else {'bert_last_layer_lr_ratio'})
            init=read(run/'initialization_audit.json');assert init['exact_tensor_equality'] and init['loaded_tensor_count']==795 and init['source_epoch']==4 and not init['optimizer_restored']
            lr=tr[0]['lr'];close(lr['router'],2e-4);close(lr['tcif_transition_gate'],6e-5)
            close(lr['head'],3e-5 if interrupted else 7.5e-6);close(lr['bert_last_layers'],3.75e-6 if interrupted else 1.5e-5)
        for split,n,nz in [('points',686,656),('validation_points',229,216)]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:
                assert sorted(p['eta'] for p in r[split] if p['checkpoint']==ck)==ETAS
            for p in r[split]:
                m=p['metrics'];assert (m['num_samples_all'],m['num_samples_non0'])==(n,nz)
                assert p['readout']=='expected' and p['T']==1
                close(p['A_star'],max(m['Acc2'],m['Acc2non0']));close(p['F_star'],max(m['F1_macro_all'],m['F1_macro_non0']))
                close(p['A_star'],m[p['A_source']]);close(p['F_star'],m[p['F_source']])
                assert all(math.isfinite(v) for v in m.values() if isinstance(v,(int,float)))
        report[name]=dict(training_complete=not interrupted,updates=len(tr),microbatch_records=len(order),
            fully_completed_epochs=167 if interrupted else 200,first_update_samples=len(samples),
            clipping_fraction=sum(x['clipped'] for x in tr)/len(tr),mean_preclip_norm=statistics.mean(x['preclip_global_norm'] for x in tr),
            mean_sampled_relative_update={g:statistics.mean(x['first_update_of_epoch'][g]['relative_update_l2'] for x in samples) for g in samples[0]['first_update_of_epoch']})
    assert orders['B_TEXT_LR']==orders['TEXT_ONLY']
    assert orders['A_FUSION_LR']==orders['B_TEXT_LR'][:len(orders['A_FUSION_LR'])]
    report['data_order']='B/TEXT exact16200; A matches first13607 logged batches (last entered batch not necessarily completed)'
    for folder,expected in [('lab_metric_audit',727),('mosei_metric_audit',140)]:
        r=read(BASE/folder/'summary.json');assert len(r['rows'])==expected
        for p in r['rows']:
            close((p['Acc7_nearest_even']-p['Acc7_legacy'])*p['n']/100,p['wrong_to_right']-p['right_to_wrong'])
    for folder in [BASE,ROOT/'snapshots/mosi_protocol_lr_20260930']:
        for p in folder.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            assert p.suffix not in ['.pth','.pt','.npz','.npy','.pkl','.gz']
            s=p.read_text(encoding='utf-8')
            assert not any(x in s for x in ['/home/admin123','/data/user/hd57166','C:/Users/','C:\\Users\\'])
    return report
if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
    print('PASS:42 test+42 val; interrupted A retained; complete B/TEXT; optimizer trace and paired order;867 fixed-file metric audits')
