import numpy as np
import torch
from collections import namedtuple

Rays = namedtuple('Rays',('origins','directions','viewdirs','radii','near','far'))

def namedtuple_map(fn,tup):
    """Apply fn to each element of tup and cast to tup's namedtuple"""
    return type(tup)(*map(fn,tup))

def sorted_piecewise_constant_pdf(bins,weights,num_samples,randomized):
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
    
    