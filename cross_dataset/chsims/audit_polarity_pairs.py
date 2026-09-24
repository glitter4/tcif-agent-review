"""Paired error attribution from existing outputs only. No inference or training."""
import bisect
import json
import struct
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
BOUNDARIES=[f32(x) for x in [-.7,-.1,.1,.7]]
def cls(x):return bisect.bisect_left(BOUNDARIES,max(-1,min(1,x)))
def load(run,ck,detach=False):
    root=REPO/'results'/('chsims_pathstudy_20260922' if detach else 'chsims_diagnostics_20260922')/'predictions'
    rows=[json.loads(s) for s in (root/run/(ck+'_test.jsonl')).read_text().splitlines()]
    result={}
    for row in rows:
        k=max(range(5),key=lambda k:row['logits'][k])
        reg=f32(row['y_reg_clipped']);center=f32(row['centers'][k]);y=f32(row['raw_label'])
        p=f32(f32(f32(.2)*reg)+f32(f32(.8)*center))
        nz=y!=0;binary=(p>=0)==(y>=0);acc5=cls(p)==cls(y)
        neighbors=[n for n in row['neighbors'] if n['valid']]
        group='no_neighbors' if not neighbors else 'conflicting' if any(y*n['raw_label']<0 for n in neighbors) else 'nonconflicting'
        strength='zero' if not nz else 'weak_nonzero_abs_le_0.2' if abs(y)<=f32(.2) else 'moderate_abs_le_0.6' if abs(y)<=f32(.6) else 'strong_abs_gt_0.6'
        assert k==2 or ((p>=0)==(center>=0)), 'High-eta sign dominance violated'
        result[row['sample_id']]={'label':y,'pred':p,'head_class':k,'neutral_head':k==2,'reg_positive':reg>=0,'binary_correct':binary,'acc5_correct':acc5,'nonzero':nz,'neighbors':group,'strength':strength}
    assert len(result)==457 and sum(r['nonzero'] for r in result.values())==388
    return result
def changes(before,after,ids,metric):
    fixed=[i for i in ids if not before[i][metric] and after[i][metric]]
    broken=[i for i in ids if before[i][metric] and not after[i][metric]]
    return {'n':len(ids),'fixed':len(fixed),'new_errors':len(broken),'net':len(fixed)-len(broken),'fixed_ids':fixed,'new_error_ids':broken}
def audit(seed,bck,dck,tag):
    before=load(f'B0_S{seed}',bck);after=load(f'DETACH_S{seed}',dck,True)
    assert before.keys()==after.keys()
    assert all(before[i]['label']==after[i]['label'] and before[i]['neighbors']==after[i]['neighbors'] for i in before)
    ids=list(before);nonzero=[i for i in ids if before[i]['nonzero']]
    errors=[i for i in nonzero if not after[i]['binary_correct']]
    attribution={}
    for neutral in [False,True]:
        selected=[i for i in errors if after[i]['neutral_head']==neutral]
        attribution['neutral_head_reg_sign_error' if neutral else 'nonneutral_head_wrong_polarity']={
            'count':len(selected),'acc5_correct':sum(after[i]['acc5_correct'] for i in selected),'ids':selected}
    groups={'all':ids,'nonzero':nonzero,'zero':[i for i in ids if not before[i]['nonzero']]}
    for key in ['neighbors','strength']:
        for value in sorted({r[key] for r in before.values()}):groups[key+':'+value]=[i for i in ids if before[i][key]==value]
    for b in [False,True]:
        for a in [False,True]:groups[f'neutral_head:{b}->{a}']=[i for i in ids if before[i]['neutral_head']==b and after[i]['neutral_head']==a]
    stats={name:{metric:changes(before,after,items,metric) for metric in ['binary_correct','acc5_correct']} for name,items in groups.items()}
    return {'seed':seed,'comparison':tag,'baseline_checkpoint':bck,'detach_checkpoint':dck,'readout':'argmax','eta':.8,'metrics_group_changes':stats,
            'detach_nonzero_binary_error_count':len(errors),'error_attribution':attribution,
            'paired_rows':[{'sample_id':i,'baseline':before[i],'detach':after[i]} for i in ids]}
def main():
    results=[audit(40,'best_mae_model','best_acc7_model','reported_representatives'),audit(41,'best_acc7_model','best_mae_model','reported_representatives')]
    for seed in [40,41]:
        for ck in ['best_acc7_model','best_mae_model']:results.append(audit(seed,ck,ck,'matched_checkpoint_rule'))
    r=results[1];s=r['metrics_group_changes']
    assert r['detach_nonzero_binary_error_count']==66
    assert s['all']['acc5_correct']['net']==13 and s['nonzero']['binary_correct']['net']==5 and s['all']['binary_correct']['net']==14
    payload={'protocol':'Existing test-selected checkpoints; fixed eta=.8 for paired mechanism diagnosis, not a replacement for complete test eta selection.',
        'mechanism':'For 5/7 < eta < 1, nonneutral class-center sign dominates clamped regression; neutral class uses regression sign. At eta=1 neutral prediction is zero.',
        'bins':'Boundaries [-.7,-.1,.1,.7], float32, bucketize right=False. Groups can overlap; fixes and new errors use identical sample IDs.',
        'comparisons':results}
    out=ROOT/'results/chsims_polarity_pair_20260923/audit.json';out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in results[:2]:
        print('seed',r['seed'],'errors',r['detach_nonzero_binary_error_count'],{k:v['count'] for k,v in r['error_attribution'].items()})
        for name in ['all','nonzero','zero','neighbors:conflicting','neighbors:nonconflicting']:
            print(name,{k:{x:v[x] for x in ['n','fixed','new_errors','net']} for k,v in r['metrics_group_changes'][name].items()})
if __name__=='__main__':main()
