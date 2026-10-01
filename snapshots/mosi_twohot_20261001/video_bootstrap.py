"""Paired original-video bootstrap, conditional on already test-selected checkpoints."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from metric_protocol import classes
from branch_analysis import predictions,read

def aggregate(y,p,groups):
    ids=np.unique(groups);out=[]
    for g in ids:
        mask=groups==g;a=y[mask];b=p[mask];truth=a>=0;pred=b>=0;nz=a!=0
        row=[len(a),float(np.abs(a.astype(np.float64)-b.astype(np.float64)).sum()),int((classes(a,'legacy_away')==classes(b,'legacy_away')).sum()),int((classes(a,'nearest_even')==classes(b,'nearest_even')).sum())]
        for valid in [np.ones(len(a),bool),nz]:
            t=truth[valid];q=pred[valid]
            row += [int((t&q).sum()),int((~t&~q).sum()),int((~t&q).sum()),int((t&~q).sum())]
        out.append(row)
    return np.asarray(out,dtype=np.float64)
def metrics(total):
    n=total[:,0];result=dict(MAE=total[:,1]/n,Acc7_legacy=total[:,2]/n*100,Acc7_nearest_even=total[:,3]/n*100)
    for name,start in [('all',4),('nonzero',8)]:
        tp,tn,fp,fn=total[:,start:start+4].T
        pos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
        neg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tn),where=(2*tn+fp+fn)>0)
        denom=tp+tn+fp+fn
        result['Acc2_'+name]=np.divide(tp+tn,denom,out=np.full_like(tp,np.nan),where=denom>0)*100
        result['macro_F1_'+name]=(pos+neg)*50
        result['weighted_F1_'+name]=np.divide(pos*(tp+fn)+neg*(tn+fp),denom,out=np.full_like(tp,np.nan),where=denom>0)*100
    result['A_star']=np.maximum(result['Acc2_all'],result['Acc2_nonzero'])
    result['F_star']=np.maximum(result['macro_F1_all'],result['macro_F1_nonzero'])
    return result
def paired(y,a,b,groups,replicates=10000,seed=20261001):
    ga=aggregate(y,a,groups);gb=aggregate(y,b,groups);assert ga.shape==gb.shape
    rng=np.random.default_rng(seed);counts=rng.multinomial(len(ga),np.full(len(ga),1/len(ga)),size=replicates)
    ma=metrics(counts@ga);mb=metrics(counts@gb);pa=metrics(ga.sum(0,keepdims=True));pb=metrics(gb.sum(0,keepdims=True))
    result={}
    for k in ma:
        d=ma[k]-mb[k];finite=np.isfinite(d)
        result[k]=dict(method=float(pa[k][0]),control=float(pb[k][0]),delta=float(pa[k][0]-pb[k][0]),
            interval95_percentile=np.quantile(d[finite],[.025,.975]).tolist() if finite.any() else None,finite_replicates=int(finite.sum()))
    return dict(n=len(y),video_groups=len(ga),replicates=replicates,rng_seed=seed,metrics=result,
        interpretation='descriptive paired-video uncertainty conditional on selected weights/eta; does not correct test-selection or source-seed uncertainty')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--control',required=True);p.add_argument('--groups',required=True);p.add_argument('--eta',type=float,default=.4);p.add_argument('--output',required=True)
    p.add_argument('--method-csv',required=True);p.add_argument('--control-csv',required=True)
    args=p.parse_args();m=read(Path(args.method));c=read(Path(args.control));y,_,_,a=predictions(m,args.eta);z,_,_,b=predictions(c,args.eta)
    assert np.array_equal(y,z) and [r['position'] for r in m]==[r['position'] for r in c]==list(range(len(y)))
    def original(path):
        rows=list(csv.DictReader(Path(path).open()));labels=np.asarray([float(x['true_value']) for x in rows],dtype=np.float32)
        assert np.array_equal(y,labels)
        return np.asarray([float(x['pred_value']) for x in rows],dtype=np.float32)
    a=original(args.method_csv);b=original(args.control_csv)
    groups=np.asarray(json.loads(Path(args.groups).read_text())['video_group_by_position']);assert len(groups)==len(y)
    result=paired(y,a,b,groups);Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('PAIRED_BOOTSTRAP_COMPLETE',result['video_groups'])
