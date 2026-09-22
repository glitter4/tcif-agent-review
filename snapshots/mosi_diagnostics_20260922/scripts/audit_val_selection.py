import json
from pathlib import Path
root=Path('/path/to/user/m4oe/tcif_mosi_diagnostics_20260922/training/runs')
output={}
for name in ['DETACH','HEAD10','FULL10']:
    base=root/name/'validation_protocol'
    options=[]
    for ck in ['best_acc7_model','best_mae_model']:
        for eta in [0,.2,.4,.6,.8,.9,1]:
            p=base/'eta_expected'/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/'val_results.json'
            m=json.loads(p.read_text())['metrics']
            options.append(dict(checkpoint=ck,eta=eta,val_mae=m['mae'],val_acc7=m['acc7']))
    expected=min(options,key=lambda x:(x['val_mae'],-x['val_acc7'],x['checkpoint'],x['eta']))
    actual=json.loads((base/'selection.json').read_text())
    assert expected==actual
    output[name]=dict(status='passed',selected=actual,candidates=options)
(root.parent/'val_selection_audit.json').write_text(json.dumps(output,indent=2))
print('PASS validation-only selection for three studies')
