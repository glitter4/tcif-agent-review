"""Standard-library checks for raw CH-SIMS output coverage and diagnostic claims."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'results/chsims_diagnostics_20260922'
def load(path):return json.loads(path.read_text(encoding='utf-8'))

def main():
    report=load(D/'diagnostics.json')
    checks=report['reconstruction_checks']
    assert len(checks)==224
    assert all(r['binary_changed']==r['bins_changed']==0 and r['max_abs_prediction_delta']<=2e-5 for r in checks)
    assert len(report['calibration_grid'])==1120
    assert len(report['val_selected_scale_test_eta_candidates'])==112
    total=0
    for run in ['B0_S40','C003_S40','B0_S41','C003_S41']:
        for ck in ['best_acc7_model','best_mae_model']:
            for split,n in [('val',456),('test',457)]:
                path=D/'predictions'/run/(ck+'_'+split+'.jsonl')
                rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                assert len(rows)==n and len({r['sample_id'] for r in rows})==n
                assert all(r['split']==split and len(r['logits'])==len(r['centers'])==5 for r in rows)
                if split=='test':assert sum(r['raw_label']!=0 for r in rows)==388
                for r in rows:
                    assert math.isclose(r['y_reg_clipped'],max(-1.,min(1.,r['y_reg_raw'])),abs_tol=1e-12)
                    assert all(math.isfinite(v) for v in r['logits']+[r['raw_label'],r['y_reg_raw']])
                    assert all(n['sample_id']!=r['sample_id'] for n in r['neighbors'] if n['valid'])
                    for key in ['regression_relative_residual','ordinal_relative_residual']:
                        assert r['tcif'][key]>=0 and math.isfinite(r['tcif'][key])
                total+=n
    assert total==7304
    for choice in report['val_selected_scale_test_eta_candidates']:
        rows=[r for r in report['calibration_grid'] if all(r[k]==choice[k] for k in ['run','checkpoint','readout','eta'])]
        assert len(rows)==10
        for split in ['val','test']:
            selected=[r for r in rows if r['split']==split]
            assert sorted(r['scale'] for r in selected)==[.9,.95,1.,1.05,1.1]
            base=next(r['metrics'] for r in selected if r['scale']==1)
            for r in selected:
                assert all(r['metrics'][key]==base[key] for key in ['Acc2','Acc2non0','F1','F1non0'])
        val=[r for r in rows if r['split']=='val']
        base=next(r['metrics'] for r in val if r['scale']==1)
        eligible=[r for r in val if r['metrics']['MAE']<=base['MAE']+1e-12]
        selected=min(eligible,key=lambda r:(-r['metrics']['Acc5'],r['metrics']['MAE'],abs(r['scale']-1)))
        assert selected['scale']==choice['scale_selected_on_val']
    process=load(D/'process_records.json')
    for run,record in process.items():
        for key in ['loss_parts_rounded_in_log','argmax_epoch_metrics','optimizer_lr_history','temporal_history']:
            assert len(record[key])==50,(run,key)
    print('PASS: 7304 raw records; 224 reconstructed grids; 1120 calibration points; fixed polarity/F1; 4x50 process records.')

if __name__=='__main__':main()
