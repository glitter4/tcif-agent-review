"""Evaluation-only mappings. Never imported into training label construction."""
import numpy as np

def classes(values, rule):
    x=np.asarray(values,dtype=np.float32)
    if not np.isfinite(x).all():raise ValueError('Non-finite labels or predictions')
    if rule=='legacy_away':q=np.where(x>=0,np.floor(x+np.float32(.5)),np.ceil(x-np.float32(.5)))
    elif rule=='nearest_even':q=np.round(x)
    else:raise ValueError(rule)
    return np.clip(q,-3,3).astype(np.int64)

def binary_metrics(y,p):
    y=np.asarray(y);p=np.asarray(p)
    result={}
    for name,mask in [('all',np.ones(len(y),dtype=bool)),('nonzero',y!=0)]:
        a=y[mask]>=0;b=p[mask]>=0
        if not len(a):result[name]={'n':0};continue
        f=[];support=[]
        for c in [False,True]:
            tp=int(((a==c)&(b==c)).sum());fp=int(((a!=c)&(b==c)).sum());fn=int(((a==c)&(b!=c)).sum())
            f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            support.append(int((a==c).sum()))
        result[name]=dict(n=len(a),Acc2=float((a==b).mean()*100),macro_F1=float(np.mean(f)*100),weighted_F1=float(np.dot(f,support)/len(a)*100))
    return result

def audit(y,p):
    y=np.asarray(y,dtype=np.float32);p=np.asarray(p,dtype=np.float32)
    if y.shape!=p.shape or y.ndim!=1 or not len(y):raise ValueError('Expected equal nonempty vectors')
    a,b=classes(y,'legacy_away'),classes(p,'legacy_away')
    c,d=classes(y,'nearest_even'),classes(p,'nearest_even')
    old=a==b;new=c==d
    changed=np.flatnonzero((a!=c)|(b!=d))
    return dict(n=len(y),Acc7_legacy=float(old.mean()*100),Acc7_nearest_even=float(new.mean()*100),
        wrong_to_right=int((~old&new).sum()),right_to_wrong=int((old&~new).sum()),
        label_class_changes=int((a!=c).sum()),prediction_class_changes=int((b!=d).sum()),
        MAE=float(np.abs(y.astype(np.float64)-p.astype(np.float64)).mean()),binary=binary_metrics(y,p),
        changed_positions=[dict(position=int(i),old_true=int(a[i]),new_true=int(c[i]),old_pred=int(b[i]),new_pred=int(d[i]),old_correct=bool(old[i]),new_correct=bool(new[i])) for i in changed])

def reconstruct(rows,eta):
    logits=np.asarray([r['cls7_logits'] for r in rows],dtype=np.float32)
    z=logits-logits.max(axis=-1,keepdims=True);prob=np.exp(z);prob/=prob.sum(axis=-1,keepdims=True)
    cls=(prob*np.arange(-3,4,dtype=np.float32)).sum(axis=-1)
    reg=np.asarray([r['y_reg_clipped'] for r in rows],dtype=np.float32)
    return np.float32(1-eta)*reg+np.float32(eta)*cls
