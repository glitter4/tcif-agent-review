"""Serialization/evaluator compatibility check, not an experimental training run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from run_followup import ROOT,OUT,CODE,PY,SOURCE,BASE
sys.path.insert(0,str(CODE))
from context_variant import StandardContextFusion

run=OUT/'standard_eval_check_r2';run.mkdir(exist_ok=False)
weights=run/'weights';weights.mkdir()
src=torch.load(SOURCE,map_location='cpu',weights_only=False,mmap=True)
state={k:v for k,v in src.items() if not k.startswith(('tcif_regression.','tcif_ordinal.'))}
feature,latent=src['tcif_regression.output_projection.weight'].shape
target=sum(v.numel() for k,v in src.items() if k.startswith('tcif_regression.'))
for name in ['tcif_regression','tcif_ordinal']:
    m=StandardContextFusion(feature,latent,target)
    state.update({name+'.'+k:v for k,v in m.state_dict().items()})
ck=weights/'best_acc7_model.pth';torch.save(state,ck)
meta=json.loads(SOURCE.with_suffix('.json').read_text())
meta.update(BASE)
meta.update(context_fusion='standard',tcif_enable_transition_gate=False,tcif_transition_gate_loss_weight=0.,checkpoint_epoch=0)
ck.with_suffix('.json').write_text(json.dumps(meta))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
with (run/'eval.log').open('w') as f:
    subprocess.run([PY,str(CODE/'eval_all_mosei_maefixed.py'),'--checkpoints_root',str(weights),
        '--results_root',str(run/'evaluation'),'--validation_only','--classification_readout','expected','--final_pred_eta','.8','--num_workers','0'],
        cwd=CODE,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
p=run/'evaluation/best_acc7_model/val_results.json'
metrics=json.loads(p.read_text())['metrics'];assert metrics['num_samples']==229
assert 0<=metrics['acc7']<=1 and 0<=metrics['mae']<6
assert not (run/'evaluation/best_acc7_model/test_results.json').exists()
(run/'passed.json').write_text(json.dumps(dict(status='passed',strict_state_load=True,validation_only=True,validation_samples=229,
    note='Converted temporary state is a compatibility check, not a trained baseline or experiment result')))
ck.unlink()
print('PASS standard-context serialization and actual validation-only evaluator; no test split invoked')
