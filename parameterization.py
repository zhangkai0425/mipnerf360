import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

def contract(x):
    """contract function of x,used in parameterization
    Arguments:
        x {[torch.tensor]} -- the parameter of t along the ray

    Returns:
        [contracted x] -- x contracted to a zone with the radius of 2
    """
    x_norm = torch.norm(x)
    if x_norm <= 1:
        return x
    else:
        return (2-1/x_norm)*(x/x_norm)

def gaussian_to_xyz(d, t_mean, t_var, r_var, diag=False):
    """lift a gaussian of conical axis to xyz axis
    Arguments:
        d:torch.float32 3-vector,the axis of the cone of each rays
        t_mean:float,t_mean of each intervals
        t_var:float,t_var of each intervals
        r_var:float,r_var of each intervals
        diag:Bool,whether use diag to approximate the matrix
    """
    mean = d[...,None,:] * t_mean[...,None]

    d_mag_sq = torch.maximum(torch.sum(d ** 2,dim=-1,keepdim=True),torch.tensor([1e-10]))
    if diag:
        d_outer_diag = d ** 2
        null_outer_diag = 1 - d_outer_diag / d_mag_sq
        t_cov_diag = t_var[..., None] * d_outer_diag[..., None, :]
        xy_cov_diag = r_var[..., None] * null_outer_diag[..., None, :]
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = d[..., :, None] * d[..., None, :]
        eye = torch.eye(d.shape[-1],device=d.device)
        null_outer = eye - d[..., :, None] * (d / d_mag_sq)[..., None, :]
        t_cov = t_var[..., None, None] * d_outer[..., None, :, :]
        xy_cov = r_var[..., None, None] * null_outer[..., None, :, :]
        cov = t_cov + xy_cov
        return mean, cov

def gaussian_contract(mean,cov):
    """lift a gaussian of xyz axis to contracted axis

    Arguments:
        mean,float,mean of xyz axis
        cov,tensor.float32,cov of xyz axis

    Returns:
        mean,float,contracted mean
        cov,tensor.float32,contracted cov matrix
    """
    mean = contract(mean)
    Jf = jacobian(contract,mean)
    cov = torch.matmul(Jf,cov)
    cov = torch.matmul(cov,Jf.T)

    return mean, cov

def conical_frustum_to_gaussian(d, t0, t1, base_radius, diag, stable=True):
    """[summary]
    approximate a conical frustum as a Gaussion distribution (mean+cov)
    Arguments:
        d {[torch.tensor]} -- [description]
        t0 {[type]} -- [description]
        t1 {[type]} -- [description]
        base_radius {[type]} -- [description]
        diag {[type]} -- [description]

    Keyword Arguments:
        stable {bool} -- [description] (default: {True})
    """
    if stable: 
        # stable method used in mipnerf
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
        t_var = (hw**2) / 3 - (4 / 15) * ((hw**4 * (12 * mu**2 - hw**2)) /
                                          (3 * mu**2 + hw**2)**2)
        r_var = base_radius**2 * ((mu**2) / 4 + (5 / 12) * hw**2 - 4 / 15 *
                                  (hw**4) / (3 * mu**2 + hw**2))
    else:
        # unstable method of the origin formula
        t_mean = (3 * (t1**4 - t0**4)) / (4 * (t1**3 - t0**3))
        r_var = base_radius**2 * (3 / 20 * (t1**5 - t0**5) / (t1**3 - t0**3))
        t_mosq = 3 / 5 * (t1**5 - t0**5) / (t1**3 - t0**3)
        t_var = t_mosq - t_mean**2
    xyz_mean,xyz_cov = gaussian_to_xyz(d, t_mean, t_var, r_var, diag)
    mean,cov = gaussian_contract(mean=xyz_mean,cov=xyz_cov)
    
    return mean,cov



