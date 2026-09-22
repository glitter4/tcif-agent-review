"""Export completed study checkpoints once; preserve existing exports."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from collect import ROOT,OUT,PY,CKPTS


def main():
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(min(65536,hard),hard))
    gpu=sys.argv[1]
    for name in sys.argv[2:]:
        source=OUT/'training/runs'/name
        assert (source/'study_complete.json').exists(),name
        for ck in CKPTS:
            run=source/'exports'/ck
            if (run/'complete.json').exists():
                print('ALREADY_EXPORTED',name,ck,flush=True)
                continue
            run.mkdir(parents=True,exist_ok=False)
            weights=run/'weights';weights.mkdir()
            for suffix in ('.pth','.json'):
                (weights/(ck+suffix)).symlink_to(source/'checkpoints'/(ck+suffix))
            env=dict(os.environ,STUDY_CODE=str(ROOT/'server-code'),STUDY_EXPORT=str(run),
                CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
            args=[PY,'-u',str(ROOT/'.codex-jobs/diagnostics/export_outputs.py'),
                '--checkpoints_root',str(weights),'--results_root',str(run/'eval'),
                '--default_dataset','cmumosi','--default_dataset_root','/path/to/user/datasets/MER-unibench/cmumosi-process',
                '--classification_readout','expected','--final_pred_eta','0.6','--num_workers','0']
            with (run/'export.log').open('w') as f:
                subprocess.run(args,cwd=ROOT/'server-code',env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            counts={s:len((run/(s+'.jsonl')).read_text().splitlines()) for s in ['val','test']}
            assert counts==dict(val=229,test=686)
            (run/'complete.json').write_text(json.dumps(dict(status='complete',counts=counts)))
            print('EXPORTED',name,ck,flush=True)


if __name__=='__main__':main()
