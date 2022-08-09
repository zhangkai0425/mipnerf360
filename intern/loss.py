import torch
import torch.nn as nn
from intern.distillation import bounds,loss_prop
from intern.regularization import loss_dist

def Loss_prop(t,w,t_hat,w_hat):
    """comput the proposal loss of prop_net

    Arguments:
        t:torch.tensor(float32), [batch_size, num_samples+1],t_vals from the nerf_net.
        w:torch.tensor(float32), [batch_size, num_samples],fine weights from the nerf_net
        t_hat:torch.tensor(float32), [batch_size, num_samples+1],t_vals from the prop_net.
        w_hat:torch.tensor(float32), [batch_size, num_samples],coarse weights from the prop_net

    Returns:
        loss:torch.tensor(float32),loss of the proposal net 
    """
    Bounds = bounds(t_vals_fine=t,fine_weights=w,t_vals_coarse=t_hat)
    loss = loss_prop(coarse_weights=w_hat,bounds=Bounds)

    return loss

def Loss_nerf(input,target):
    """compute the reconstruction loss of nerf_net

    Arguments:
        input:torch.tensor(float32), [batch_size, 3],input rgb color of batch rays.
        target:torch.tensor(float32), [batch_size, 3],target rgb color of batch rays.

    Returns:
        mse_loss:torch.tensor(float32),mse_loss
        psnr:torch.tensor(float32),psnr
    """
    batch_size = input.shape[0]
    mse_loss = ((input[...,:3] - target[..., :3]) ** 2).sum() / batch_size
    psnr = mse_to_psnr(mse_loss)
    # this way,we can change the loss to log(loss) and speed up the training
    mse_loss = -mse_to_psnr(mse_loss) + 30

    return mse_loss,psnr

def Loss_dist(s_vals,weights):
    """compute the distance loss of rays distances and weights

    Arguments:
        s_vals:torch.tensor,[batch_size,num_samples+1],sampled disparity values.
        weights:torch.tensor(float32), weights for t_vals

    Returns:
        loss:torch.tensor(float32),loss_dist according to the paper  
    """
    loss = loss_dist(s_vals=s_vals,weights=weights)
    return loss


def mse_to_psnr(mse):
    """transform mse to psnr"""
    return -10.0 * torch.log10(mse)