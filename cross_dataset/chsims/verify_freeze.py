import csv,json,sys
from pathlib import Path
import torch
run=Path(sys.argv[1]);cfg=json.loads((run/'config.json').read_text())['config']
base=torch.load(cfg['init_checkpoint'],map_location='cpu')
source=Path(cfg['init_checkpoint']).parent.parent
checks=[]
for path in sorted((run/'checkpoints').glob('best_*_model.pth')):
    current=torch.load(path,map_location='cpu')
    assert set(current)==set(base)
    changed=[k for k in base if not torch.equal(base[k],current[k])]
    assert changed and all(k.startswith('head_cls7.') for k in changed),changed
    for split in ['val','test']:
        def load(p):
            with p.open() as f:return {r['id']:float(r['pred_value']) for r in csv.DictReader(f)}
        old=load(source/'eta_argmax/eta_0p0/best_mae_model'/(split+'_details.csv'))
        new=load(run/'eta_argmax/eta_0p0'/path.stem/(split+'_details.csv'))
        assert old.keys()==new.keys()
        delta=max(abs(old[k]-new[k]) for k in old)
        assert delta<2e-5,delta
        checks.append({'checkpoint':path.stem,'split':split,'max_regression_delta':delta,'changed_parameters':changed})
    del current
(run/'freeze_verification.json').write_text(json.dumps(checks,indent=2))
print('FROZEN_PARAMETERS_AND_REGRESSION_VERIFIED',flush=True)
