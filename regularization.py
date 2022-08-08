import torch
from para import g

def t_to_s(t_vals,near,far):
    """transform t to s:using the formula in the paper"""
    s_vals = (g(t_vals) - g(near)) / (g(far) - g(near))
    return s_vals

def loss_dist(s_vals,weights):
    """compute the loss_dist according to the paper 

    Arguments:
        s_vals:torch.tensor,[batch_size,num_samples+1],sampled disparity values.
        weights:torch.tensor(float32),[batch_size,num_samples], weights for t_vals

    Returns:
        loss_dist:torch.tensor(float32),loss_dist according to the paper  
    """
    loss_dist = 0
    for i in range(weights.shape[-1]):
        for j in range(weights.shape[-1]):
            loss_dist += weights[...,i] * weights[...,j] * torch.abs((s_vals[...,i] + s_vals[...,i+1])/2-(s_vals[...,j] + s_vals[...,j+1])/2)
    loss_dist += 1/3 * torch.sum(weights ** 2 * (s_vals[...,1:]-s_vals[...,:-1]))
    return loss_dist
