"""Validate fixed-budget extra-data transfer and complete target eta evaluation."""
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/mosei_to_mosi_20260927'
def read(name):return json.loads((DATA/name).read_text(encoding='utf-8'))

def main():
    r=read('result.json');source=read('source_complete.json');init=read('initialization_audit.json')
    assert source['status']=='complete' and source['epochs']==4 and source['selected']=='fixed epoch4'
    assert source['source_test_evaluated'] is False and not source['optimizer_transferred']
    assert [x['epoch'] for x in source['source_validation_trajectory']]==[1,2,3,4]
    assert init['status']=='passed' and init['exact_tensor_equality'] and not init['optimizer_restored']
    assert init['loaded_tensor_count']==795 and init['loaded_elements']==394229721
    sm=read('source_checkpoint_metadata.json');assert sm['checkpoint_epoch']==4 and sm['transfer_phase']=='source' and sm['dataset']=='cmumosei'
    audit=read('data_audit.json')
    assert all(x['protected_target_video_count']==0 and x['protected_target_sample_count']==0 for x in audit['source_target_overlap'].values())
    assert all(x['missing']==0 for x in audit['cache_coverage'].values())
    assert audit['source_internal_train_val_video_overlap']==0
    assert {k:v for k,v in r['config'].items() if not k.startswith('transfer_')}==read('D1_config.json')
    manifest=read('manifest.json');assert manifest['source_selection']=='fixed_epoch4' and manifest['source_test_evaluation'] is False
    assert manifest['source']['epochs']==4 and manifest['target']['epochs']==200
    assert r['training_complete'] and r['source_epochs']==4 and r['target_epochs']==200 and r['extra_supervised_data']
    pts=r['points'];assert len(pts)==14
    meta=read('target_checkpoint_metadata.json')
    for ck in ['best_acc7_model','best_mae_model']:
        assert sorted(x['eta'] for x in pts if x['checkpoint']==ck)==[0,.2,.4,.6,.8,.9,1]
        assert meta[ck]['checkpoint_epoch']==r['checkpoint_epochs'][ck] and meta[ck]['transfer_phase']=='target'
    for p in pts:
        m=p['metrics'];assert p['readout']=='expected' and p['T']==1
        assert m['num_samples_all']==686 and m['num_samples_non0']==656
        assert p['A_star']==max(m['Acc2'],m['Acc2non0'])==m[p['A_source']]
        assert p['F_star']==max(m['F1_macro_all'],m['F1_macro_non0'])==m[p['F_source']]
        assert p['F_source'] in ('F1_macro_all','F1_macro_non0')
        assert all(math.isfinite(m[k]) for k in ['Acc7','MAE','Acc2','Acc2non0','F1_macro_all','F1_macro_non0'])
    stage=any(p['eta']>0 and p['metrics']['Acc7']>=46.1 and p['A_star']>=85 and p['F_star']>=85 and p['metrics']['MAE']<=.730 for p in pts)
    final=any(p['eta']>0 and p['metrics']['Acc7']>48.5 and p['A_star']>86.95 and p['F_star']>86.94 and p['metrics']['MAE']<.697 for p in pts)
    assert stage==r['stage_pass']==read('complete.json')['stage_pass']
    assert final==r['final_pass']==read('complete.json')['final_pass']
    print('PASS: fixed source4/target200, no source-test evaluation,795 transferred tensors, compatible config/data audit,14 eta points and same-point A*/macro-F*.')

if __name__=='__main__':main()
