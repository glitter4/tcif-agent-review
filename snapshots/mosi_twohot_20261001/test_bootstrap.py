import numpy as np
from video_bootstrap import paired,aggregate,metrics
from metric_protocol import audit
y=np.array([-2,-.2,.1,2,0,1],dtype=np.float32);p=np.array([-1,-.1,-.1,2,.1,1],dtype=np.float32)
groups=np.array([0,0,1,1,2,2]);result=paired(y,p,p,groups,1000)
assert all(v['delta']==0 and v['interval95_percentile']==[0.,0.] for v in result['metrics'].values())
m=metrics(aggregate(y,p,groups).sum(0,keepdims=True));a=audit(y,p)
assert abs(m['Acc7_legacy'][0]-a['Acc7_legacy'])<1e-10
assert abs(m['MAE'][0]-a['MAE'])<1e-10
assert abs(m['macro_F1_nonzero'][0]-a['binary']['nonzero']['macro_F1'])<1e-10
r=paired(y,y,p,groups,1000);s=paired(y,p,y,groups,1000)
for k in r['metrics']:
    assert abs(r['metrics'][k]['delta']+s['metrics'][k]['delta'])<1e-10
    assert np.allclose(r['metrics'][k]['interval95_percentile'],-np.array(s['metrics'][k]['interval95_percentile'])[::-1])
print('PASS identical-model zero CI, metric equality, paired reversal, reproducible grouped draws')
