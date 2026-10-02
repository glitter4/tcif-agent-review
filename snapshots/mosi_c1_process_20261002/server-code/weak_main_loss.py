"""Fixed weak-nonzero reweighting. Only training labels set normalization."""
import torch
import torch.nn.functional as F

def normalization_from_training_labels(labels, *, split):
    if split != 'train':
        raise ValueError('Weight normalization must use training labels only')
    y=torch.as_tensor(labels,dtype=torch.float64)
    if y.numel()!=1284 or not torch.isfinite(y).all():
        raise ValueError('Expected all1284 finite MOSI training labels')
    weak=(y.abs()>0)&(y.abs()<.5)
    return 1.+.5*float(weak.sum())/y.numel()

def sample_weights(y,training_mean):
    if not 1.<=training_mean<=1.5:
        raise ValueError('Invalid frozen training normalization')
    weak=(y.abs()>0)&(y.abs()<.5)
    return torch.where(weak,torch.full_like(y,1.5),torch.ones_like(y))/training_mean

def weighted_main_losses(y_reg,logits,y,target_probs,training_mean):
    if y_reg.shape!=y.shape or logits.shape!=target_probs.shape:
        raise ValueError('Per-example loss shapes must match')
    w=sample_weights(y,training_mean).detach()
    reg=((y_reg-y).abs()*w).mean()
    cls=(-(target_probs.to(logits)*F.log_softmax(logits,dim=-1)).sum(-1)*w).mean()
    # Caller retains original .75 CE and .05 gate coefficients; gate is untouched.
    return reg,cls
