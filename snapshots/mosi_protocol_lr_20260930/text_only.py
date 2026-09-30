"""Bounded MOSI-only center-text diagnostic; test-selected dual checkpoints and seven eta."""
import argparse,json,random,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader

ROOT=Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'server-code').is_dir() else Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'server-code'))
from datasets.emotion_dataset import CMUMOSIProcessDataset
from transformers import AutoModel
from head_utils import continuous_to_cls7_soft,soft_cross_entropy,compute_final_prediction
from train_emotion import _build_lr_scheduler
from update_audit import UpdateAudit
from metric_protocol import audit

ETAS=[0,.2,.4,.6,.8,.9,1]
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

class TextData(Dataset):
    def __init__(self,root,split,backbone):
        self.original=CMUMOSIProcessDataset(root=root,split=split,tokenizer_path=backbone,max_length=128,temporal_context_radius=0)
        self.ids=self.original.ids
        assert self.original.tokenizer is not None
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        ids,mask=self.original.encoded_text[i]
        return ids,mask,torch.tensor(float(self.original.labels[self.ids[i]]['val']),dtype=torch.float32),i

class TextModel(nn.Module):
    def __init__(self,backbone):
        super().__init__();self.bert=AutoModel.from_pretrained(backbone,local_files_only=True)
        for p in self.bert.parameters():p.requires_grad=False
        layers=self.bert.encoder.layer
        assert len(layers)==12
        for p in layers.parameters():p.requires_grad=True
        self.dropout=nn.Dropout(.5);h=self.bert.config.hidden_size
        self.reg=nn.Linear(h,1);self.cls=nn.Linear(h,7)
    def forward(self,ids,mask):
        h=self.bert(input_ids=ids,attention_mask=mask).last_hidden_state
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        pooled=self.dropout(pooled)
        return self.reg(pooled).squeeze(-1),self.cls(pooled)

def objective(r,z,y):return (r-y).abs().mean()+.75*soft_cross_entropy(z,continuous_to_cls7_soft(y,tau=.3))
def metrics(y,p):
    x=audit(y,p);b=x['binary'];m=dict(Acc7=x['Acc7_legacy'],Acc7_nearest_even=x['Acc7_nearest_even'],MAE=x['MAE'],
        Acc2=b['all']['Acc2'],Acc2non0=b['nonzero']['Acc2'],F1_macro_all=b['all']['macro_F1'],F1_macro_non0=b['nonzero']['macro_F1'],
        F1_weighted_all=b['all']['weighted_F1'],F1_weighted_non0=b['nonzero']['weighted_F1'],num_samples_all=b['all']['n'],num_samples_non0=b['nonzero']['n'])
    return m
@torch.no_grad()
def infer(model,loader):
    model.eval();ys=[];regs=[];logits=[]
    for ids,mask,y,_ in loader:
        r,z=model(ids.cuda(),mask.cuda());ys.append(y);regs.append(r.clamp(-3,3).cpu());logits.append(z.cpu())
    return torch.cat(ys),torch.cat(regs),torch.cat(logits)

def decoded(outputs,eta):
    y,r,z=outputs;p=compute_final_prediction(r,z,eta=eta);return metrics(y.numpy(),p.numpy()),p

def train(out,root,backbone):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'started.json').open('x') as f:json.dump(dict(kind='MOSI-only center text',seed=123,epochs=200),f)
    random.seed(123);np.random.seed(123);torch.manual_seed(123);torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(False,warn_only=True)
    torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True
    data={s:TextData(root,s,backbone) for s in ['train','val','test']}
    assert [len(data[s]) for s in data]==[1284,229,686]
    loaders={s:DataLoader(d,batch_size=16,shuffle=s=='train',num_workers=0,generator=torch.Generator().manual_seed(123) if s=='train' else None) for s,d in data.items()}
    model=TextModel(backbone).cuda()
    groups=[dict(name='bert_last_layers',params=list(model.bert.encoder.layer.parameters()),lr=3.75e-6),
            dict(name='head',params=list(model.reg.parameters())+list(model.cls.parameters()),lr=7.5e-6)]
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    logger=UpdateAudit(model,optimizer,out)
    scheduler=_build_lr_scheduler(optimizer,SimpleNamespace(scheduler='cosine',epochs=200,lr_warmup_epochs=1,lr_warmup_start_factor=0.,min_lr=0.,lr_min_factor=.03))
    dump(out/'initialization_audit.json',dict(source_supervised_pretraining=False,backbone='same original RoBERTa as MOSI-only D1',frozen_embeddings=True,encoder_layers_trainable=12,optimizer_restored=False))
    dump(out/'text_input_audit.json',{s:dict(n=len(d),empty_attention=sum(int(mask.sum()==0) for _,mask in d.original.encoded_text),max_length=128,transcript_column='same CMUMOSIProcessDataset column2 fallback1') for s,d in data.items()})
    best={};epochs={}
    for epoch in range(1,201):
        start=time.monotonic();model.train();optimizer.zero_grad();loss_sum=0
        for batch,(ids,mask,y,positions) in enumerate(loaders['train'],1):
            r,z=model(ids.cuda(),mask.cuda());loss=objective(r,z,y.cuda());assert torch.isfinite(loss)
            (loss/2).backward();loss_sum+=float(loss.detach())*len(y)
            with (out/'data_order.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,micro_step=batch,positions=positions.tolist()))+'\n')
            if batch%2==0 or batch==len(loaders['train']):
                logger.before(epoch);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step();logger.after(epoch,norm,1.);optimizer.zero_grad()
        values={s:decoded(infer(model,loaders[s]),.85)[0] for s in ['val','test']}
        m=values['test'];keys={'best_acc7_model':(-m['Acc7'],m['MAE'],epoch),'best_mae_model':(m['MAE'],-m['Acc7'],epoch)}
        for ck,key in keys.items():
            if ck not in best or key<best[ck]:
                best[ck]=key;epochs[ck]=epoch;(out/'checkpoints').mkdir(exist_ok=True)
                torch.save(model.state_dict(),out/'checkpoints'/(ck+'.pth'))
                dump(out/'checkpoints'/(ck+'.json'),dict(checkpoint_epoch=epoch,selection_split='test',selection_eta=.85,acc7_rule='legacy_away',metrics=m))
        with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(dict(epoch=epoch,train_loss=loss_sum/1284,optimizer_steps=logger.count,wall_seconds=time.monotonic()-start,**values))+'\n')
        print('TEXT_EPOCH',epoch,'test',m,flush=True);scheduler.step()
    points=[];valpoints=[]
    for ck in best:
        model.load_state_dict(torch.load(out/'checkpoints'/(ck+'.pth'),map_location='cuda',weights_only=True),strict=True)
        for split,collection in [('test',points),('val',valpoints)]:
            outputs=infer(model,loaders[split])
            for eta in ETAS:
                m,p=decoded(outputs,eta);collection.append(dict(checkpoint=ck,eta=eta,readout='expected',T=1,metrics=m))
                # Position-based complete prediction records; original identifiers stay on data server.
                dest=out/'predictions'/ck/(split+'_eta'+str(eta)+'.json')
                dump(dest,[dict(position=i,true_value=float(y),pred_value=float(v)) for i,(y,v) in enumerate(zip(outputs[0],p))])
    assert logger.count==8200
    result=dict(id='TEXT_ONLY',training_complete=True,target_epochs=200,checkpoint_epochs=epochs,points=points,validation_points=valpoints,
        supervision='MOSI-only, original pretrained RoBERTa',protocol='test-selected, legacyAcc7, finaleta.85; dual7eta')
    dump(out/'result.json',result);dump(out/'study_complete.json',dict(status='complete',epochs=200,updates=8200))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--dataset-root',required=True);parser.add_argument('--backbone',required=True)
    a=parser.parse_args();train(a.out,a.dataset_root,a.backbone)
