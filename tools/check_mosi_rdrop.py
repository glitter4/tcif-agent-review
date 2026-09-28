"""Validate published R-Drop results and complete training traces without ML packages."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/mosi_rdrop_20260929'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def vec(p):return dict(Acc7=p['metrics']['Acc7'],MAE=p['metrics']['MAE'],A_star=p['A_star'],F_star=p['F_star'])
def get(r,split,ck,eta):return next(p for p in r[split] if p['checkpoint']==ck and p['eta']==eta)
def close(a,b):assert abs(a-b)<1e-9,(a,b)
def main():
    runs={n:read(BASE/(n+'.json')) for n in ['C0','C1','C2']}
    audit=read(BASE/'audit.json');steps={}
    assert read(BASE/'complete.json')['status']=='complete'
    for n,r in runs.items():
        assert r['training_complete'] and r['target_epochs']==200
        for split,counts in [('points',(686,656)),('validation_points',(229,216))]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:
                assert sorted(p['eta'] for p in r[split] if p['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
            for p in r[split]:
                m=p['metrics'];assert (m['num_samples_all'],m['num_samples_non0'])==counts
                assert p['T']==1 and p['readout']=='expected'
                assert p['A_star']==max(m['Acc2'],m['Acc2non0'])==m[p['A_source']]
                assert p['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[p['F_source']]
                assert all(math.isfinite(x) for x in m.values() if isinstance(x,(float,int)))
        if n=='C0':continue
        assert {k:v for k,v in r['config'].items() if not k.startswith('rdrop_')}==runs['C0']['config']
        w=0 if n=='C1' else .1
        assert r['config']['rdrop_views']==2 and r['config']['rdrop_kl_weight']==w
        init=read(BASE/n/'initialization_audit.json')
        assert init['exact_tensor_equality'] and not init['optimizer_restored'] and init['source_epoch']==4 and init['loaded_tensor_count']==795
        epochs=lines(BASE/n/'rdrop_epochs.jsonl');steps[n]=lines(BASE/n/'rdrop_steps.jsonl')
        assert len(epochs)==200 and len(steps[n])==16200
        assert sum(x['optimizer_steps'] for x in epochs)==8200
        assert sum(x['stochastic_forward_calls'] for x in epochs)==32400
        assert sum(x['model_center_exposures'] for x in epochs)==513600
        for ep,e in enumerate(epochs,1):
            assert e['epoch']==ep and e['total_optimizer_steps']==ep*41
            batch=steps[n][(ep-1)*81:ep*81]
            assert [x['micro_step'] for x in batch]==list(range(1,82))
            assert [x['n'] for x in batch]==[16]*80+[4]
            assert all(x['epoch']==ep for x in batch)
            assert sorted(i for x in batch for i in x['sample_indices'])==list(range(1284))
            for x in batch:
                assert x['same_inputs'] and x['data_order_matches_reference'] and x['max_logit_difference']>0
                assert x['kl']>=-1e-6
                close(x['weighted_kl'],w*x['kl'])
                assert all(math.isfinite(x[k]) for k in ['kl','base_loss_first','base_loss_second'])
            close(e['mean_kl'],sum(x['kl']*x['n'] for x in batch)/1284)
        close(audit[n]['training_wall_seconds'],sum(x['training_wall_seconds'] for x in epochs))
        close(audit[n]['mean_kl_first10'],sum(x['mean_kl'] for x in epochs[:10])/10)
        close(audit[n]['mean_kl_last10'],sum(x['mean_kl'] for x in epochs[-10:])/10)
    assert all((a['epoch'],a['micro_step'],a['sample_indices'])==(b['epoch'],b['micro_step'],b['sample_indices']) for a,b in zip(steps['C1'],steps['C2']))
    for row in read(BASE/'comparison.json')['rows']:
        n,ck,eta=row['run'],row['checkpoint'],row['eta']
        for split,key in [('points','test_delta'),('validation_points','validation_delta')]:
            a=vec(get(runs['C0'],split,ck,eta));b=vec(get(runs[n],split,ck,eta))
            for k in a:close(row[key][k],b[k]-a[k])
            if split=='points':
                assert row['guard_pass']==(b['Acc7']>=a['Acc7']-.3 and b['MAE']<=a['MAE']+.005)
                assert row['polarity_gain']==(b['A_star']>a['A_star'] and b['F_star']>a['F_star'])
    for row in read(BASE/'C2_vs_C1.json'):
        for split,key in [('points','test_delta'),('validation_points','validation_delta')]:
            a=vec(get(runs['C1'],split,row['checkpoint'],row['eta']));b=vec(get(runs['C2'],split,row['checkpoint'],row['eta']))
            for k in a:close(row[key][k],b[k]-a[k])
    for directory in [BASE,ROOT/'snapshots/mosi_rdrop_20260929']:
        for p in directory.rglob('*'):
            if not p.is_file():continue
            assert p.suffix not in ['.pth','.pt','.npz','.npy','.pkl','.gz']
            text=p.read_text(encoding='utf-8')
            assert not any(x in text for x in ['/home/admin123','/data/user/hd57166','C:/Users/','C:\\Users\\'])
    print('PASS: 42 test + 42 val points, 400 epochs, 32400 paired microbatch records, strict source initialization, budgets, KL and matched comparisons.')
if __name__=='__main__':main()
