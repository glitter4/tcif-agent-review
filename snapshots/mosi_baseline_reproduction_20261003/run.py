"""c1-only MOSI reproduction driver; original EMOE model and training objective."""
import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'vendor/EMOE-main'))


class Data(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data['regression_labels'])
    def __getitem__(self,i):
        d = self.data
        return (torch.as_tensor(d['text_bert'][i],dtype=torch.float32),
                torch.as_tensor(d['audio'][i],dtype=torch.float32),
                torch.as_tensor(d['vision'][i],dtype=torch.float32),
                torch.as_tensor(d['regression_labels'][i],dtype=torch.float32).reshape(1),i)


def metrics(p,y):
    p,y = np.asarray(p).flatten(),np.asarray(y).flatten()
    nz = y!=0
    rnd = lambda x: np.sign(x)*np.floor(np.abs(x)+.5)
    pc,yc = p.clip(-3,3),y.clip(-3,3)
    return dict(n=len(y),mae=float(np.abs(p-y).mean()),
        acc7_numpy=float((pc.round()==yc.round()).mean()*100),
        acc7_project=float((rnd(pc)==rnd(yc)).mean()*100),
        acc2_non0=float(((p[nz]>0)==(y[nz]>0)).mean()*100),
        f1_weighted_non0=float(f1_score(y[nz]>0,p[nz]>0,average='weighted')*100),
        f1_macro_non0=float(f1_score(y[nz]>0,p[nz]>0,average='macro')*100),
        acc2_zero_positive=float(((p>=0)==(y>=0)).mean()*100))


@torch.no_grad()
def evaluate(model,loader,device):
    model.eval()
    preds,ys,indices,losses = [],[],[],[]
    for t,a,v,y,i in loader:
        p = model(t.to(device),a.to(device),v.to(device))['logits_c'].cpu()
        preds.append(p.numpy()); ys.append(y.numpy()); indices.append(i.numpy())
        losses.append(F.l1_loss(p,y).item())
    p,y,idx = np.concatenate(preds),np.concatenate(ys),np.concatenate(indices)
    order = idx.argsort(); p,y = p[order],y[order]
    m = metrics(p,y); m['batch_l1']=round(float(np.mean(losses)),4)
    return m,p,y


def emoe_loss(o,y):
    w=o['channel_weight']
    # Official order: language, vision, audio.
    dist=torch.cat([(o['logits_'+m]-y).square() for m in ('l','v','a')],1)
    inv=1/(dist+.1); target=(inv/inv.sum(1,keepdim=True)).detach()
    sim=(target-w).square().mean()
    prob=w.clamp_min(1e-9); ent=(3*(prob*prob.log()).sum(1)).mean()
    teacher=sum(o[m+'_proj']*w[:,i:i+1] for i,m in enumerate(('l','v','a'))).detach()
    ud=(o['c_proj'].softmax(-1)-teacher.softmax(-1)).square().mean()
    return F.l1_loss(o['logits_c'],y)+sum(F.l1_loss(o['logits_'+m],y) for m in ('l','v','a'))/3+.1*(ent+.1*sim)+.1*ud


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['emoe','fine'],required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--data',required=True)
    parser.add_argument('--bert',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--epochs',type=int)
    args=parser.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'GPU allocation is required'
    device=torch.device('cuda')
    with open(args.data,'rb') as f: data=pickle.load(f)
    audit={}
    for split,n in [('train',1284),('valid',229),('test',686)]:
        d=data[split]
        assert len(d['regression_labels'])==n,(split,len(d['regression_labels']))
        for key in ('audio','vision'):
            # Official EMOE cleaning: negative infinity in audio only.
            if key=='audio': d[key][d[key]==-np.inf]=0
            assert np.isfinite(d[key]).all(),(split,key,'nonfinite input')
        audit[split]={k:list(np.shape(d[k])) for k in ['text_bert','audio','vision','regression_labels']}
        assert np.isfinite(d['regression_labels']).all()
    idsets=[set(map(str,data[s]['id'])) for s in ['train','valid','test']]
    assert not(idsets[0]&idsets[1] or idsets[0]&idsets[2] or idsets[1]&idsets[2]),'split overlap'
    dims=[768,data['train']['audio'].shape[-1],data['train']['vision'].shape[-1]]
    if args.model=='emoe':
        from trains.singleTask.model.emoe import EMOE
        cfg=json.loads((ROOT/'vendor/EMOE-main/config/config.json').read_text())
        params={**cfg['datasetCommonParams']['mosi']['unaligned'],**cfg['emoe']['commonParams'],**cfg['emoe']['datasetParams']['mosi']}
        params.update(dataset_name='mosi',need_data_aligned=False,pretrained=args.bert,feature_dims=dims)
        model=EMOE(SimpleNamespace(**params)).to(device)
        bs=16; maxepochs=args.epochs or 100
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5)
    else:
        from fine import FINE
        model=FINE(args.bert,dims,data['train']['regression_labels']).to(device)
        bs=32; maxepochs=args.epochs or 100
        critic_params=[p for c in model.critics() for p in c.parameters()]
        crit_ids={id(p) for p in critic_params}; bert_ids={id(p) for p in model.bert.parameters()}
        optimizer=torch.optim.AdamW([
            {'params':list(model.bert.parameters()),'lr':4e-5},
            {'params':[p for p in model.parameters() if id(p) not in crit_ids|bert_ids],'lr':6e-5}],weight_decay=.01)
        critic_optimizer=torch.optim.AdamW(critic_params,lr=6e-5,weight_decay=.01)
        params=dict(batch_size=32,main_lr=6e-5,bert_lr=4e-5,epochs=maxepochs,
                    experts=4,queries=4,top_k=3,feature_dims=dims)
    loaders={s:DataLoader(Data(data[s]),batch_size=bs,shuffle=(s=='train' or args.model=='emoe'),num_workers=0) for s in data}
    if args.model=='fine':
        total=maxepochs*len(loaders['train']); warmup=max(1,int(.1*total))
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step: min((step+1)/warmup,max(0,(total-step)/(total-warmup))))
    meta=dict(arguments=vars(args),settings=params,data=audit,torch=torch.__version__,
              parameters=sum(p.numel() for p in model.parameters()),
              eta='not applicable: scalar regression only; no classification/regression mixture',
              input_source='MMSA standard MOSI mirror; FINE paper feature dimensions differ')
    (out/'protocol.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)
    best={'test_acc7_numpy':-1.,'test_acc7_project':-1.,'test_mae':float('inf')}
    selected={}; bestval=float('inf'); valepoch=0
    for epoch in range(1,maxepochs+1):
        start=time.time(); model.train(); optimizer.zero_grad(); losses=[]
        if args.model=='fine': critic_optimizer.zero_grad()
        for step,(t,a,v,y,_) in enumerate(loaders['train']):
            t,a,v,y=[x.to(device) for x in (t,a,v,y)]
            if args.model=='emoe':
                if step%10==0: optimizer.zero_grad()
                output=model(t,a,v); loss=emoe_loss(output,y)
                assert torch.isfinite(loss), 'nonfinite loss'
                loss.backward()
                if (step+1)%10==0: optimizer.step()
            else:
                optimizer.zero_grad(); critic_optimizer.zero_grad()
                output=model(t,a,v,y); loss=output['loss']
                assert torch.isfinite(loss+output['critic_loss']), 'nonfinite loss'
                (loss+output['critic_loss']).backward()
                optimizer.step(); critic_optimizer.step(); scheduler.step()
            losses.append(loss.item())
            if step==0: print(f'epoch={epoch} first_batch_loss={loss.item():.6f}',flush=True)
        vm,_,_=evaluate(model,loaders['valid'],device)
        tm,p,y=evaluate(model,loaders['test'],device)
        np.savez_compressed(out/f'predictions_epoch_{epoch:03}.npz',prediction=p,label=y,ids=np.asarray(data['test']['id']).astype(str))
        row=dict(epoch=epoch,seconds=time.time()-start,train_loss=float(np.mean(losses)),valid=vm,test=tm)
        print(json.dumps(row),flush=True)
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        for key,metric in [('test_acc7_numpy','acc7_numpy'),('test_acc7_project','acc7_project'),('test_mae','mae')]:
            value=tm[metric]
            better=value<best[key] if key=='test_mae' else value>best[key]
            if better:
                best[key]=value
                selected[key]=dict(epoch=epoch,metrics=tm,predictions=f'predictions_epoch_{epoch:03}.npz')
                # numpy/project Acc7 can choose different epochs; preserve each explicitly.
                torch.save(dict(model=model.state_dict(),epoch=epoch,arguments=vars(args)),out/(key+'.pt'))
        if vm['batch_l1']<=bestval-1e-6:
            bestval=vm['batch_l1']; valepoch=epoch
            selected['official_val_mae_reference']=dict(epoch=epoch,metrics=tm,predictions=f'predictions_epoch_{epoch:03}.npz')
        (out/'selected.json').write_text(json.dumps(selected,indent=2))
        if args.model=='emoe':
            scheduler.step(vm['batch_l1'])
            if epoch-valepoch>=10:break
    (out/'DONE.json').write_text(json.dumps(dict(epochs=epoch,selected=selected),indent=2))
    print('TRAINING_COMPLETE',flush=True)


if __name__=='__main__':main()
