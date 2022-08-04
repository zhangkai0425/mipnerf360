import torch
import torch.nn as nn

def bounds(t_vals_fine,fine_weights,t_vals_coarse):
    """compute the bounds of prop net:opposite of the origin paper
    because I don't think the paper is right:the output of the proposal network should be the
    envelope of the nerf network,so we should compute the bound using nerf_net's output
    what do you think?Please raise an issue if you have a different idea!

    Arguments:
        t_vals_fine:torch.tensor(float32), [batch_size, num_samples+1],t_vals from the nerf_net.
        fine_weights:torch.tensor(float32), [batch_size, num_samples],fine weights from the nerf_net
        T:torch.tensor(float32), [batch_size, num_samples+1],t_vals from the prop_net.

    Returns:
        bounds:torch.tensor(float32), [batch_size, num_samples],bounds of the fine_weight,should be consist with the coarse weight,
        therefore the coarse_weight becomes the envelope of the fine_weight
    """
    t0 = t_vals_fine[...,:-1]
    t1 = t_vals_fine[...,1:]
    T0 = t_vals_coarse[...,:-1]
    T1 = t_vals_coarse[...,1:]
    fine_weights = fine_weights.detach()
    # fine_weights and coarse weights are all 128 samples,the same shape
    B = torch.zeros_like(fine_weights)
    # use for loop,only 128 times,so don't worry the cost of time
    for i in range(fine_weights.shape[-1]):
        L,R = T0[...,i],T1[...,i]
        B[...,i] = torch.sum(fine_weights[~((t0>R)|(t1<L))],dim=-1)

    B = B.detach()
    return B

def loss_prop(coarse_weights,bounds):
    """[summary]

    Arguments:
        coarse_weights {[type]} -- [description]
        bounds {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    return 0
def distillation(student,teacher):
    """implementation of distillation network"""

    return True
    