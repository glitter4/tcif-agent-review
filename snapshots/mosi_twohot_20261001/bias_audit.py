"""Aggregate target-distribution bias using train/val labels only; no model inference."""
import json,sys
from pathlib import Path
import numpy as np
def audit(y):
    y=np.asarray(y,dtype=np.float64);c=np.arange(-3,4,dtype=np.float64)
    q=np.exp(-np.abs(y[:,None]-c)/.3);q/=q.sum(1,keepdims=True);e=q@c
    upper=np.clip(np.searchsorted(c,y,side='right'),1,6);lower=upper-1;frac=y-c[lower]
    t=np.zeros_like(q);t[np.arange(len(y)),lower]=1-frac;t[np.arange(len(y)),upper]+=frac
    assert np.max(np.abs(t@c-y))<1e-12 and np.all(t>=0) and np.allclose(t.sum(1),1)
    masks=dict(all=np.ones(len(y),bool),weak_positive=(y>0)&(y<.5),weak_negative=(y<0)&(y>-.5),
        other_nonzero=np.abs(y)>=.5,zero=y==0,strong=np.abs(y)>=2,endpoints=np.abs(y)==3)
    return {g:dict(n=int(m.sum()),mean_signed_bias=float(np.mean((e-y)[m])) if m.any() else None,
        mean_absolute_bias=float(np.mean(np.abs(e-y)[m])) if m.any() else None,
        max_absolute_bias=float(np.max(np.abs(e-y)[m])) if m.any() else None,
        pushed_up=int(((e-y>1e-12)&m).sum()),pushed_down=int(((e-y < -1e-12)&m).sum()),
        two_hot_max_expectation_error=float(np.max(np.abs(t@c-y)[m])) if m.any() else None) for g,m in masks.items()}
if __name__=='__main__':
    data=np.load(Path(sys.argv[1])/'label.npz',allow_pickle=True)
    result={s:audit([float(r['val']) for r in data[s+'_corpus'].item().values()]) for s in ['train','val']}
    c=np.arange(-3,4);examples=[]
    for y in [.1,.2,.8,-.2,0.,3.,-3.]:
        q=np.exp(-np.abs(y-c)/.3);q/=q.sum();examples.append(dict(y=y,distance_expectation=float(q@c),two_hot_expectation=y))
    out=dict(formula='distance exp(-abs(y-c)/.3); two_hot linear interpolation',splits=result,examples=examples,
        note='target math only; not evidence of learned model performance; no test-label analysis')
    Path(sys.argv[2]).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
