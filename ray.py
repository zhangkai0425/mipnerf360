import numpy as np
import torch
from collections import namedtuple
from parameterization import g,para_rays

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

def sample_along_rays(origins,directions,radii,num_samples,near,far,randomized):
    """[summary]

    Arguments:
        origins:torch.tensor(float32),[batch_size,3],ray origins.
        directions:torch.tensor(float32),[batch_size,3],ray directions.
        radii:torch.tensor(float32),[batch_size,1],ray radii.
        num_samples:int,number of samples.
        near:torch.tensor,[batch_size,1],near clip.
        far:torch.tensor,[batch_size,1],far clip.
    
    Returns:
        t_vals:torch.tensor,[batch_size,num_samples],sampled z values.
        means:torch.tensor,[batch_size,num_samples,3],sampled means.
        covs:torch.tensor,[batch_size,num_samples,3,3],sampled covariances.
    """
    batch_size = origins.shape[0]

    # mipnerf360 sample strategy
    s_vals = torch.linspace(0.,1,num_samples + 1,device=origins.device)
    t_vals = g(s_vals * g(far) + (1-s_vals) * g(near))

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand(batch_size, num_samples + 1, device=origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        # Broadcast t_vals to make the returned shape consistent.
        t_vals = torch.broadcast_to(t_vals, [batch_size, num_samples + 1])

    #TODO: do not use diag matrix?
    means,covs = para_rays(t_vals=t_vals,origins=origins,directions=directions,radii=radii,diag=False)
    
    return t_vals,(means,covs)

def resample_along_rays(origins,directions,radii,t_vals,weights,randomized,resample_padding):
    """Resampling along rays.

    Arguments:
        origins:torch.tensor(float32), [batch_size, 3], ray origins.
        directions:torch.tensor(float32), [batch_size, 3], ray directions.
        radii:torch.tensor(float32), [batch_size, 1], ray radii.
        t_vals:torch.tensor(float32), [batch_size, num_samples+1].
        weights:torch.tensor(float32), weights for t_vals
        randomized:bool, use randomized samples.
        resample_padding:float, added to the weights before normalizing.

    Returns:
        t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
        means: torch.tensor, [batch_size, num_samples, 3], sampled means.
        covs:torch.tensor,[batch_size,num_samples,3,3],sampled covariances.
    """
    # stop grad,do not backprop during sampling
    with torch.no_grad():
        weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
        weights_max = torch.maximum(weights_pad[..., :-1], weights_pad[..., 1:])
        weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

         # Add in a constant (the sampling function will renormalize the PDF).
        weights = weights_blur + resample_padding

        new_t_vals = sorted_piecewise_constant_pdf(
                t_vals,
                weights,
                t_vals.shape[-1],
                randomized,
            )

    #TODO: do not use diag matrix?
    means,covs = para_rays(t_vals=new_t_vals,origins=origins,directions=directions,radii=radii,diag=False)
    return new_t_vals, (means, covs)

def volumetric_rendering(rgb,density,t_vals,dirs,white_bkgd):
    """Volumetric rendering function

    Arguments:
        rgb:torch.tensor(float32),color,[batch_size,num_samples,3]
        density: torch.tensor(float32), density, [batch_size, num_samples, 1].
        t_vals:torch.tensor(float32), [batch_size, num_samples].
        dirs:torch.tensor(float32), [batch_size, 3].
        white_bkgd:bool,whether use white background(nerf synthetic data) or not(nerf llff data)

    Returns:
        comp_rgb:torch.tensor(float32), [batch_size, 3],final color of batch pixels.
        disp:torch.tensor(float32),[batch_size],final disparity of batch pixels.
        acc:torch.tensor(float32),[batch_size],finla accumulation of batch rays.
        weights:torch.tensor(float32),[batch_size,num_samples],weights along the rays.
    """
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]
    delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density[..., 0] * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(-torch.cat([
        torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)
    ], dim=-1))
    weights = alpha * trans

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)
    distance = (weights * t_mids).sum(dim=-1) / acc
    distance = torch.clamp(torch.nan_to_num(distance), t_vals[:, 0], t_vals[:, -1])

    if white_bkgd:
        comp_rgb = comp_rgb + (1. - acc[..., None])
    return comp_rgb, distance, acc, weights