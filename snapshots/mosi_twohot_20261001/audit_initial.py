"""Bounded completed-run audit and validation-decision verification; no training."""
import json,math,tarfile
from pathlib import Path
ROOT=Path('/path/to/user/m4oe/tcif_mosi_twohot_20261001')
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
    assert read(ROOT/'initial_complete.json')['status']=='complete'
    names=['V2_STANDARD_DROP','V3_TCIF_TWOHOT'];order={};masks={};summary={};files=[]
    for n in names:
        p=ROOT/'runs'/n;r=read(p/'result.json');tr=lines(p/'optimizer_updates.jsonl');order[n]=lines(p/'data_order.jsonl');masks[n]=lines(p/'neighbor_dropout.jsonl')
        init=read(p/'initialization_audit.json');assert init['source_epoch']==4 and init['exact_tensor_equality'] and not init['optimizer_restored']
        assert init['loaded_tensor_count']==(767 if n==names[0] else 795)
        assert r['training_complete'] and len(tr)==8200 and tr[-1]['epoch']==200 and len(order[n])==16200 and len(masks[n])==16200
        for split,count,nz in [('points',686,656),('validation_points',229,216)]:
            assert len(r[split])==14
            for ck in ['best_acc7_model','best_mae_model']:
                assert sorted(x['eta'] for x in r[split] if x['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
                assert r['checkpoint_epochs'][ck]==read(p/'checkpoints'/(ck+'.json'))['checkpoint_epoch']
            for x in r[split]:
                m=x['metrics'];assert (m['num_samples_all'],m['num_samples_non0'])==(count,nz)
                assert all(math.isfinite(v) for v in m.values()) and x['A_star']==max(m['Acc2'],m['Acc2non0']) and x['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])
        for ep in range(200):
            batch=order[n][81*ep:81*(ep+1)]
            assert [x['micro_step'] for x in batch]==list(range(1,82)) and all(x['epoch']==ep+1 for x in batch)
            assert [len(x['positions']) for x in batch]==[16]*80+[4]
            assert sorted(i for x in batch for i in x['positions'])==list(range(1284))
        for m in masks[n]:assert m['valid_before']-m['dropped']==m['valid_after']
        summary[n]=dict(epochs=200,updates=8200,microbatches=16200,strict_tensors=init['loaded_tensor_count'],clip_fraction=sum(x['clipped'] for x in tr)/8200)
        for f in ['result.json','config.json','initialization_audit.json','study_complete.json','optimizer_groups.json','optimizer_updates.jsonl','data_order.jsonl','neighbor_dropout.jsonl']:
            files.append(p/f)
        files+=list((p/'checkpoints').glob('*.json'))
    assert order[names[0]]==order[names[1]] and masks[names[0]]==masks[names[1]]
    ref=Path('/path/to/user/m4oe/tcif_mosi_regularization_20261001/runs/NEIGHBOR_DROP')
    assert order[names[0]]==lines(ref/'data_order.jsonl') and masks[names[0]]==lines(ref/'neighbor_dropout.jsonl')
    decision=read(ROOT/'twohot_expansion_decision.json');d=decision['details'];weak=['weak_positive','weak_negative']
    expected=dict(classification_MAE_improves=d['classification_expectation']['MAE']<d['reference_classification_expectation']['MAE'],
        final_MAE_improves=d['final']['MAE']<d['reference_final']['MAE'],Acc7_guard=d['final']['Acc7_legacy']>=d['reference_final']['Acc7_legacy']-.3,
        each_weak_sign_error_not_worse=all(d['groups'][g]['final_sign_errors']<=d['groups'][g]['reference_final_sign_errors'] for g in weak),
        combined_weak_sign_error_improves=sum(d['groups'][g]['final_sign_errors'] for g in weak)<sum(d['groups'][g]['reference_final_sign_errors'] for g in weak))
    assert expected==decision['conditions'] and decision['twohot_expansion']==all(expected.values())
    summary['mask_and_order_equal_to_V1']=True;summary['twohot_expansion']=decision['twohot_expansion'];summary['decision_conditions']=expected
    (ROOT/'initial_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    for f in ['initial_audit.json','initial_complete.json','target_bias_audit.json','twohot_expansion_decision.json','branch_analysis.json','manifest.json','test_video_groups.json','V1_validation_groups.json']:
        files.append(ROOT/f)
    with tarfile.open('/tmp/mosi-twohot-initial-results.tgz','w:gz') as t:
        for f in files:assert f.is_file() and not f.is_symlink();t.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    print(json.dumps(summary))
if __name__=='__main__':main()
