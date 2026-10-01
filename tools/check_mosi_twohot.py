import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'results/mosi_twohot_20261001'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
    orders=[];masks=[]
    for name in ['V2_STANDARD_DROP','V3_TCIF_TWOHOT']:
        p=BASE/'runs'/name;r=read(p/'result.json');tr=lines(p/'optimizer_updates.jsonl')
        assert r['training_complete'] and len(tr)==8200 and tr[-1]['epoch']==200
        o=lines(p/'data_order.jsonl');m=lines(p/'neighbor_dropout.jsonl');assert len(o)==len(m)==16200;orders.append(o);masks.append(m)
        for split,n,nz in [('points',686,656),('validation_points',229,216)]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:assert sorted(p['eta'] for p in r[split] if p['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
            for q in r[split]:
                x=q['metrics'];assert (x['num_samples_all'],x['num_samples_non0'])==(n,nz)
                assert all(math.isfinite(v) for v in x.values()) and q['A_star']==max(x['Acc2'],x['Acc2non0']) and q['F_star']==max(x['F1_macro_all'],x['F1_macro_non0'])
    assert orders[0]==orders[1] and masks[0]==masks[1]
    d=read(BASE/'twohot_expansion_decision.json');assert d['twohot_expansion'] is False and not all(d['conditions'].values())
    assert d['split']=='val' and d['checkpoint']=='best_mae_model' and d['eta']==.4
    b=read(BASE/'bootstrap_seed123_primary.json');assert b['video_groups']==31 and b['replicates']==10000 and b['rng_seed']==20261001
    for root in [BASE,ROOT/'snapshots/mosi_twohot_20261001']:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                assert p.suffix not in ['.pth','.pt','.npz','.npy','.gz']
                s=p.read_text(encoding='utf-8');assert not any(x in s for x in ['/home/admin123','/data/user/hd57166','C:/Users/','C:\\Users\\'])
    print('PASS initial28test+28val, paired order/masks, failed validation gate,31-video bootstrap; task not yet complete')
if __name__=='__main__':main()
