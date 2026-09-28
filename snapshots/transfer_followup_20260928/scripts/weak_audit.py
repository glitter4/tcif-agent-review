import csv,json
from pathlib import Path
root=Path('/path/to/user/m4oe/tcif_mosei_to_mosi_20260927/runs/MOSEI4_to_MOSI_D1/eta_expected')
rows=[]
for ck in ['best_acc7_model','best_mae_model']:
 for eta in [.6,.8,1.]:
  f=root/('eta_'+f'{eta:.1f}'.replace('.','p'))/ck/'test_details.csv'
  items=list(csv.DictReader(f.open()))
  groups={}
  for name,select in [('weak_pos',lambda y:0<y<.5),('weak_neg',lambda y:-.5<y<0),('nonweak_nonzero',lambda y:abs(y)>=.5)]:
   selected=[r for r in items if select(float(r['true_value']))]
   groups[name]=dict(n=len(selected),sign_errors=sum((float(r['true_value'])>=0)!=(float(r['pred_value'])>=0) for r in selected))
  rows.append(dict(checkpoint=ck,eta=eta,groups=groups))
Path('/tmp/transfer_weak_audit_20260928.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
