"""Audit completed MOSI interventions and their separate selection protocols."""
import json
import math
from check_mosi_diagnostics import DATA, ETAS, metrics, score


def main():
    studies=json.loads((DATA/'study_results.json').read_text(encoding='utf-8'))
    val_audit=json.loads((DATA/'val_selection_audit.json').read_text())
    assert {r['id'] for r in studies}=={'DETACH','HEAD10','FULL10'}
    point_count=0;sample_count=0
    for run in studies:
        name=run['id']
        for protocol in ['test_selected','validation_selected']:
            block=run[protocol];points=block['points']
            assert len(points)==14
            for ck in ['best_acc7_model','best_mae_model']:
                assert sorted(p['eta'] for p in points if p['checkpoint']==ck)==ETAS
                assert 1<=block['checkpoint_epochs'][ck]<= (200 if name=='DETACH' else 10)
            for p in points:
                m=p['metrics']
                assert m['num_samples_all']==686 and m['num_samples_non0']==656
                assert m['F1non0']==m['F1_macro_non0']
                assert p['readout']=='expected' and p['T']==1
                assert all(math.isfinite(m[k]) for k in ['MAE','Acc7','Acc2non0','F1non0'])
            point_count+=len(points)
        candidates=val_audit[name]['candidates']
        assert len(candidates)==14
        selected=min(candidates,key=lambda r:(r['val_mae'],-r['val_acc7'],r['checkpoint'],r['eta']))
        assert selected==run['validation_selected']['selected']==val_audit[name]['selected']
        for ck in ['best_acc7_model','best_mae_model']:
            for split,count in [('val',229),('test',686)]:
                rows=[json.loads(line) for line in (DATA/'samples'/name/ck/(split+'.jsonl')).read_text().splitlines()]
                parent={r['sample_id']:r for r in [json.loads(l) for l in (DATA/'samples/D1/best_acc7_model'/(split+'.jsonl')).read_text().splitlines()]}
                assert len(rows)==count and {r['sample_id'] for r in rows}==set(parent)
                sample_count+=count
                if name=='HEAD10':
                    assert max(abs(r['y_reg_raw']-parent[r['sample_id']]['y_reg_raw']) for r in rows)<=1e-6
                if split=='test':
                    for eta in ETAS:
                        result=metrics(rows,[score(r,eta) for r in rows])
                        expected=next(p['metrics'] for p in run['test_selected']['points'] if p['checkpoint']==ck and p['eta']==eta)
                        assert all(abs(result[k]-expected[k])<=1e-5 for k in result),(name,ck,eta)
        verification=run['output_verification']
        assert verification['status']=='passed' and len(verification['eta_points'])==28
        assert all(p['max_prediction_difference']<=1e-5 for p in verification['eta_points'])
        steps=8200 if name=='DETACH' else 410
        gradients=[json.loads(l) for l in (DATA/'training_process'/name/'gradient_norms.jsonl').read_text().splitlines()]
        assert len(gradients)==steps==verification['gradient_norm_summary']['optimizer_steps']
        assert all(math.isfinite(p['preclip_norm']) for p in gradients)
        if name=='HEAD10':
            assert len(run['frozen_tensor_audit'])==4
            assert run['gradient_probe']['status']=='not_applicable'
    assert point_count==84 and sample_count==5490
    print('PASS: three completed studies, 84 test/validation-selected eta points, 5490 additional samples, frozen regression, validation-only selection and all optimizer-step records.')


if __name__=='__main__':main()
