"""Scientific invariants: loss fidelity, inference isolation and MI gradient routing."""
import tempfile
from pathlib import Path
import torch
from transformers import BertConfig, BertModel
from run import emoe_loss, metrics
from fine import FINE

torch.set_num_threads(4)
torch.manual_seed(19)
from trains.utils.functions import eva_imp, uni_distill, entropy_balance
y=torch.randn(4,1)
o={k:torch.randn(4,1) for k in ['logits_c','logits_l','logits_v','logits_a']}
o.update({k:torch.randn(4,12) for k in ['c_proj','l_proj','v_proj','a_proj']})
w=torch.softmax(torch.randn(4,3),-1);o['channel_weight']=w
dist=torch.zeros(4,3)
ld,ad,vd=[eva_imp(o['logits_'+m],y) for m in ['l','a','v']]
for i in range(4):
    denom=1/(ld[i]+.1)+1/(vd[i]+.1)+1/(ad[i]+.1)
    for j,d in enumerate([ld,vd,ad]):dist[i,j]=(1/(d[i]+.1))/denom
official=(torch.nn.functional.l1_loss(o['logits_c'],y)
    +sum(torch.nn.functional.l1_loss(o['logits_'+m],y) for m in ['l','v','a'])/3
    +.1*(entropy_balance(w)+.1*(dist-w).square().mean())
    +.1*uni_distill(o['c_proj'],sum(o[m+'_proj']*w[:,i:i+1] for i,m in enumerate(['l','v','a']))))
torch.testing.assert_close(emoe_loss(o,y),official)
assert metrics([-.5,.5],[-.5,.5])['acc7_project']==100
with tempfile.TemporaryDirectory() as tmp:
    BertModel(BertConfig(vocab_size=50,hidden_size=768,num_hidden_layers=1,
                        num_attention_heads=12,intermediate_size=128)).save_pretrained(tmp)
    model=FINE(tmp,[768,5,20],[-2,-1,0,1,2,3])
    text=torch.zeros(4,3,8);text[:,0]=torch.randint(0,50,(4,8));text[:,1]=1
    audio=torch.randn(4,11,5);vision=torch.randn(4,13,20)
    audio[0]=0;vision[1]=0  # all-padding rows must remain finite
    labels=torch.tensor([[-1.2],[-1.1],[1.2],[1.1]])
    output=model(text,audio,vision,labels)
    assert torch.isfinite(output['loss']+output['critic_loss'])
    output['loss'].backward(retain_graph=True)
    assert all(p.grad is None for c in model.critics() for p in c.parameters())
    assert model.bert.embeddings.word_embeddings.weight.grad is not None
    model.zero_grad();output['critic_loss'].backward()
    assert model.bert.embeddings.word_embeddings.weight.grad is None
    assert all(any(p.grad is not None for p in c.parameters()) for c in model.critics())
    model.eval()
    before=[len(q) for q in model.queues]
    with torch.no_grad():
        p1=model(text,audio,vision)['logits_c']
        for p in model.label_s.parameters():p.fill_(100)
        model.queues=[[] for _ in range(7)]
        p2=model(text,audio,vision)['logits_c']
    torch.testing.assert_close(p1,p2,rtol=0,atol=0)
    assert sum(map(len,model.queues))==0
    assert sum(before)==4
print('PASS: official EMOE loss parity; FINE finite gradients; isolated critic updates; label/queue-free inference')
