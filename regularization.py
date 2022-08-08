import torch

def t_to_s(t_vals,near,far):
    """transform t to s:using the formula in the paper"""
    s_vals = (g(t_vals) - g(near)) / (g(far) - g(near))
    return s_vals

def loss_dist(t_vals,near,far,weights):
    """compute the loss_dist according to the paper 

    Arguments:
        t_vals:torch.tensor,[batch_size,num_samples+1],sampled z values.
        near:torch.tensor,[batch_size,1],near clip.
        far:torch.tensor,[batch_size,1],far clip.
        weights:torch.tensor(float32), weights for t_vals

    Returns:
        loss_dist:torch.tensor(float32),loss_dist according to the paper  
    """
    loss_dist = 0
    s_vals = t_to_s(t_vals=t_vals,near=near,far=far)
    for i in range(weights.shape[-1]):
        for j in range(weights.shape[-1]):
            loss_dist += weights[i] * weights[j] * torch.abs((s_vals[i] + s_vals[i+1])/2-(s_vals[j] + s_vals[j+1])/2)
    loss_dist += 1/3 * torch.sum(weights ** 2 * (s_vals[1:]-s_vals[:-1]))
    return loss_dist