"""MOSI seven-class target interpolation; evaluation mappings are unchanged."""
import torch

def two_hot(y,centers=None):
    y=torch.as_tensor(y)
    if not y.is_floating_point():y=y.to(torch.float32)
    if centers is None:centers=torch.arange(-3,4,device=y.device,dtype=y.dtype)
    centers=centers.to(device=y.device,dtype=y.dtype)
    if centers.shape!=(7,) or not torch.equal(centers,torch.arange(-3,4,device=y.device,dtype=y.dtype)):
        raise ValueError('two_hot is restricted to MOSI signed7 centers')
    if not torch.isfinite(y).all() or (y < -3).any() or (y > 3).any():
        raise ValueError('two_hot requires finite labels within[-3,3]; no silent clipping')
    flat=y.reshape(-1);upper=torch.searchsorted(centers,flat.contiguous(),right=True).clamp(1,6);lower=upper-1
    fraction=(flat-centers[lower])/(centers[upper]-centers[lower])
    probs=y.new_zeros((flat.numel(),7))
    probs.scatter_(1,lower[:,None],(1-fraction)[:,None]);probs.scatter_add_(1,upper[:,None],fraction[:,None])
    return probs.reshape(*y.shape,7)
