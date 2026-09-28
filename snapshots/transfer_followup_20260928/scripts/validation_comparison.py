"""Fixed eta.8 validation diagnostics for experiment1, not model selection."""
from pathlib import Path
import run_followup as r

out=r.OUT/'source_initial_target_validation'
r.e.CODE=r.CODE;r.e.PY=r.PY
if not (out/'summary.json').exists():
    out.mkdir(exist_ok=True)
    r.e.command([r.PY,'-u','eval_all_mosei_maefixed.py','--checkpoints_root',str(r.OUT/'source_reference_weights'),
        '--results_root',str(out),'--validation_only','--override_dataset','cmumosi',
        '--override_dataset_root',r.BASE['dataset_root'],'--override_embedding_cache_root',r.BASE['embedding_cache_root'],
        '--classification_readout','expected','--final_pred_eta','.8','--num_workers','0'],out/'eval.log','0')
    m=r.e.read(out/'best_acc7_model/val_results.json')['metrics'];assert m['num_samples']==229
    r.e.dump(out/'summary.json',{k:m[k] for k in ['num_samples','mae','acc7','acc2','binary_f1']})
rows={'source_initial':{'source_val':r.e.read(r.OUT/'source_initial_validation/summary.json')['best_acc7_model'],
                         'target_val':r.e.read(out/'summary.json')}}
for name,run in [('C_ST',r.REFERENCE),('E1_ROUTER',r.OUT/'runs/E1_ROUTER')]:
    source=r.e.read(r.OUT/'reference_source_validation/summary.json') if name=='C_ST' else r.e.read(run/'source_validation/summary.json')
    rows[name]={}
    for ck in r.e.CKPTS:
        m=r.e.read(run/'eta_expected/eta_0p8'/ck/'val_results.json')['metrics'];assert m['num_samples']==229
        rows[name][ck]=dict(source_val=source[ck],target_val={k:m[k] for k in ['num_samples','mae','acc7','acc2','binary_f1']})
r.e.dump(r.OUT/'validation_comparison.json',dict(eta=.8,readout='expected',T=1,rows=rows,selection=False))
print('VALIDATION_COMPARISON_COMPLETE')
