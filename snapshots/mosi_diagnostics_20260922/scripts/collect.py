from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import resource

ROOT=Path('/path/to/user/workspaces/m4oe-tcif-diagnostics-20260922')
OUT=Path('/path/to/user/m4oe/tcif_mosi_diagnostics_20260922')
PY='/path/to/user/envs/m4oe-lab5090/bin/python'
SOURCES={
 'D1':Path('/path/to/user/m4oe/tcif_literature_five_20260918/runs/D1'),
 'R1':Path('/path/to/user/m4oe/tcif_mosi_distribution_20260922/runs/R1_router_temp015')}
EVALS={
 'D1':Path('/path/to/user/m4oe/tcif_recovery_20260918/eval/D1/eta_expected'),
 'R1':SOURCES['R1']/'eta_expected'}
CKPTS=['best_acc7_model','best_mae_model']


def lane(name,gpu):
    results=[]
    for ck in CKPTS:
        run=OUT/'exports'/name/ck
        run.mkdir(parents=True,exist_ok=False)
        links=run/'weights';links.mkdir()
        for suffix in ('.pth','.json'):
            (links/(ck+suffix)).symlink_to(SOURCES[name]/'checkpoints'/(ck+suffix))
        args=[PY,'-u',str(ROOT/'.codex-jobs/diagnostics/export_outputs.py'),
              '--checkpoints_root',str(links),'--results_root',str(run/'eval'),
              '--default_dataset','cmumosi','--default_dataset_root','/path/to/user/datasets/MER-unibench/cmumosi-process',
              '--classification_readout','expected','--final_pred_eta','0.6','--num_workers','0']
        env=dict(os.environ,STUDY_CODE=str(ROOT/'server-code'),STUDY_EXPORT=str(run),
                 CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
        with (run/'export.log').open('w') as f:
            subprocess.run(args,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=ROOT/'server-code')
        results.append(dict(run=name,checkpoint=ck,status='complete'))
        print('EXPORTED',name,ck,flush=True)
    return results


if __name__=='__main__':
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    OUT.mkdir(exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(lane,'D1',0);b=pool.submit(lane,'R1',1)
        try:
            results=a.result()+b.result()
            (OUT/'export_complete.json').write_text(json.dumps(results,indent=2))
        except Exception as exc:
            (OUT/'export_failure.json').write_text(json.dumps(dict(error=repr(exc))))
            raise
