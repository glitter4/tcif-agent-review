"""Re-score existing fixed prediction files only; never select or run a model."""
import csv,json
from pathlib import Path
from metric_protocol import audit

BASE=Path('/path/to/user/m4oe')
OUT=BASE/'tcif_mosi_protocol_lr_20260930'
ROOTS=['tcif_mosi_budget_20260916','tcif_mosi_followup_20260916','tcif_mosi_distribution_20260922',
       'tcif_mosi_diagnostics_20260922','tcif_mosei_to_mosi_20260927','tcif_transfer_followup_20260928',
       'tcif_rdrop_20260929','tcif_parameter_sensitivity_20260910','tcif_mosi_seed_lab_20260919']
def main():
    rows=[];skipped=[]
    for root in ROOTS:
        for p in sorted((BASE/root).rglob('*_details.csv')):
            if p.name not in ['test_details.csv','val_details.csv'] or 'validation_protocol' in p.parts:continue
            samples=list(csv.DictReader(p.open()))
            if not samples or not {'true_value','pred_value'}<=set(samples[0]):continue
            n=len(samples);dataset='mosi' if n in [686,229] else 'mosei' if n in [4659,1871] else None
            if dataset is None:skipped.append(dict(file=str(p.relative_to(BASE)),n=n));continue
            result=audit([float(x['true_value']) for x in samples],[float(x['pred_value']) for x in samples])
            relative=p.relative_to(BASE).with_suffix('.json')
            dest=OUT/'metric_audit/changes'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(json.dumps(result.pop('changed_positions')))
            oldfile=p.with_name(p.name.replace('_details.csv','_results.json'))
            old=json.loads(oldfile.read_text()) if oldfile.exists() else {}
            old=old.get('metrics',old)
            if 'acc7' in old:assert abs(result['Acc7_legacy']/100-old['acc7'])<1e-6,(p,result['Acc7_legacy'],old['acc7'])
            rows.append(dict(file=str(p.relative_to(BASE)),dataset=dataset,split=p.name.split('_')[0],**result))
    report=dict(status='completed_for_available_files',scope=ROOTS,selection='fixed existing points, no reselection; file provenance distinguishes target sweeps/source val diagnostics',
        historical_Acc7_reproduced=True,rows=rows,skipped=skipped)
    (OUT/'metric_audit/summary.json').write_text(json.dumps(report,indent=2))
    print('AUDIT_FILES',len(rows),'MOSI',sum(r['dataset']=='mosi' for r in rows),'MOSEI',sum(r['dataset']=='mosei' for r in rows))
    for r in rows:
        if 'MOSEI4_to_MOSI_D1/eta_expected/eta_0p8/best_acc7_model/test_details' in r['file']:print(r)
if __name__=='__main__':main()
