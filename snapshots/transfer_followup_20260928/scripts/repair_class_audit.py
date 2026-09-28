"""Resolve whether sign repairs retain, gain or lose the correct ordinal class."""
import csv
import math
import run_followup as r

def k(x):return math.floor(x+.5) if x>=0 else -math.floor(-x+.5)
run=r.OUT/'runs/E2_S';rows=[]
for ck in r.e.CKPTS:
 for eta in r.e.ETAS:
  for split in ['val','test']:
   tag='eta_'+f'{eta:.1f}'.replace('.','p')
   a={x['id']:x for x in csv.DictReader((r.REFERENCE/'eta_expected'/tag/ck/(split+'_details.csv')).open())}
   b={x['id']:x for x in csv.DictReader((run/'eta_expected'/tag/ck/(split+'_details.csv')).open())};assert set(a)==set(b)
   groups={}
   for name,select in [('weak_pos',lambda y:0<y<.5),('weak_neg',lambda y:-.5<y<0),('nonweak_nonzero',lambda y:abs(y)>=.5),('zero',lambda y:y==0)]:
    rec=dict(n=0,baseline_abs_error_sum=0.,new_abs_error_sum=0.,sign_fixes=0,new_sign_errors=0,
       fixed_sign_class_transition={t:0 for t in ['correct_to_correct','wrong_to_correct','correct_to_wrong','wrong_to_wrong']})
    for sid,x in a.items():
     y=float(x['true_value']);old=float(x['pred_value']);new=float(b[sid]['pred_value'])
     assert y==float(b[sid]['true_value'])
     if not select(y):continue
     rec['n']+=1;rec['baseline_abs_error_sum']+=abs(old-y);rec['new_abs_error_sum']+=abs(new-y)
     oldsign=(old>=0)==(y>=0);newsign=(new>=0)==(y>=0)
     rec['new_sign_errors']+=int(oldsign and not newsign)
     if not oldsign and newsign:
      rec['sign_fixes']+=1
      transition=('correct' if k(old)==k(y) else 'wrong')+'_to_'+('correct' if k(new)==k(y) else 'wrong')
      rec['fixed_sign_class_transition'][transition]+=1
    rec['baseline_mae']=rec['baseline_abs_error_sum']/rec['n'] if rec['n'] else None
    rec['new_mae']=rec['new_abs_error_sum']/rec['n'] if rec['n'] else None
    groups[name]=rec
   rows.append(dict(checkpoint=ck,eta=eta,split=split,groups=groups))
r.e.dump(run/'paired_repairs_detailed.json',rows)
print('REPAIR_CLASS_AUDIT_COMPLETE')
