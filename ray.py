import numpy as np
import torch
from collections import namedtuple
from parameterization import g,conical_frustum_to_gaussian,para_rays

Rays = namedtuple('Rays',('origins','directions','viewdirs','radii','near','far'))

def namedtuple_map(fn,tup):
    """Apply fn to each element of tup and cast to tup's namedtuple"""
    return type(tup)(*map(fn,tup))

def sorted_piecewise_constant_pdf(bins,weights,num_samples,randomized=True):
    """compute the PDF and CDF of each ray"""
    # do padding to avoid NaNs when the input is zero or small
    eps = 1e-5
    weight_sum = torch.sum(weights,dim=-1,keepdim=True)
    padding = torch.maximum(torch.zeros_like(weight_sum),eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    # Compute the PDF and CDF of each weight vector
    # CDF starts with exactly 0 and ends with exactly 1
    pdf = weights / weight_sum
    cdf = torch.cumsum(pdf[...,:-1],dim=-1)
    cdf = torch.minimum(torch.ones_like(cdf),cdf)
    cdf = torch.cat([torch.zeros(list(cdf.shape[:-1])+[1],device=cdf.device),cdf,torch.ones(list(cdf.shape[:-1])+[1],device=cdf.device)],
                    dim=-1)
    
    # Draw uniform samples
    if randomized:
        s = 1 / num_samples
        u = (torch.arange(num_samples, device=cdf.device) * s)[None, ...]
        u = u + u + torch.empty(list(cdf.shape[:-1]) + [num_samples], device=cdf.device).uniform_(to=(s - torch.finfo(torch.float32).eps))
        # `u` is in [0, 1) --- it can be zero, but it can never be 1.
        u = torch.minimum(u, torch.full_like(u, 1. - torch.finfo(torch.float32).eps, device=u.device))
    else:
        # Match the behavior of jax.random.uniform() by spanning [0, 1-eps].
        u = torch.linspace(0., 1. - torch.finfo(torch.float32).eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])
    
    # Identify the location in `cdf` that corresponds to a random sample.
    # The final `True` index in `mask` will be the start of the sampled interval.
    mask = u[..., None, :] >= cdf[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0, _ = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2)
        x1, _ = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2)
        return x0, x1
    
    bins_g0, bins_g1 = find_interval(bins)
    cdf_g0, cdf_g1 = find_interval(cdf)

    t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
    samples = bins_g0 + t * (bins_g1 - bins_g0)
    return samples

def convert_to_ndc(origins,directions,focal,w,h,near=1.0):
    """Convert a set of rays to NDC coordinates"""
    # Shift ray origins to near plane
    t = -(near + origins[...,2]) / (directions[...,2] + 1e-15)
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / (oz + 1e-15))
    o1 = -((2 * focal) / h) * (oy / (oz+ 1e-15) )
    o2 = 1 + 2 * near / (oz+ 1e-15)

    d0 = -((2 * focal) / w) * (dx / (dz+ 1e-15) - ox / (oz+ 1e-15))
    d1 = -((2 * focal) / h) * (dy / (dz+ 1e-15) - oy / (oz+ 1e-15))
    d2 = -2 * near / (oz+ 1e-15)

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions

def sample_along_rays(origins,directions,radii,num_samples,near,far,ray_shape):
    """[summary]

    Arguments:
        origins:torch.tensor(float32),[batch_size,3],ray origins.
        directions:torch.tensor(float32),[batch_size,3],ray directions.
        radii:torch.tensor(float32),[batch_size,1],ray radii.
        num_samples:int,number of samples.
        near:torch.tensor,[batch_size,1],near clip.
        far:torch.tensor,[batch_size,1],far clip.
        ray_shape:torch.Size,shape of rays
    
    Returns:
        t_vals:torch.tensor,[batch_size,num_samples],sampled z values.
        means:torch.tensor,[batch_size,num_samples,3],sampled means.
        covs:torch.tensor,[batch_size,num_samples,3,3],sampled covariances.
    """
    batch_size = origins.shape[0]

    # mipnerf360 sample strategy
    s_vals = torch.linspace(0.,1,num_samples + 1,device=origins.device)
    t_vals = g(s_vals * g(far) + (1-s_vals) * g(near))
    t_vals = torch.broadcast_to(t_vals,[batch_size,num_samples + 1])
    #TODO: do not use diag matrix?
    means,covs = para_rays(t_vals=t_vals,origins=origins,directions=directions,radii=radii,ray_shape=ray_shape,diag=False)
    
    return t_vals,(means,covs)

def resample_along_rays(origins,directions,radii,t_vals,weights,randomized,resample_padding,ray_shape):
    return 0

def volumetric_rendering(rgb,density,t_vals,dirs,white_bkgd):
    return 0




