import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

def g(x):
    """compute the disparity of x:g(x)=1/x"""
    # pad the tensor to avoid dividing zero
    eps = 1e-6
    x += eps
    s = 1/x
    return s

def t_to_s(t_vals,near,far):
    """transform t to s:using the formula in the paper"""
    s_vals = (g(t_vals) - g(near)) / (g(far) - g(near))
    return s_vals

def contract(x):
    """contract function of x,used in parameterization"""
    x_norm = torch.norm(x)
    if x_norm <= 1:
        return x
    else:
        return (2-1/x_norm)*(x/x_norm)

def gaussian_to_xyz(d, t_mean, t_var, r_var, diag=False):
    """lift a gaussian of conical axis to xyz axis

    Arguments:
        d:torch.tensor(float32),[batch_size,3],ray directions.
        t_mean:torch.tensor(float32),[batch_size,1],t_mean of each intervals
        t_var:torch.tensor(float32),[batch_size,1],t_var of each intervals
        r_var:torch.tensor(float32),[batch_size,1],r_var of each intervals
        diag:boolean, whether or the Gaussian will be diagonal or full-covariance.
    
    Returns:
        means:torch.tensor,[batch_size,num_samples,3],sampled means.
        covs:torch.tensor,[batch_size,num_samples,3,3],sampled covariances.
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
    """Approximate a conical frustum as a contracted Gaussian distribution (mean+cov).
    
    Arguments:
        d:torch.tensor(float32),[batch_size,3],ray directions.
        t0:torch.tensor(float32),[batch_size,1],the starting distance of the frustum.
        t1:torch.tensor(float32),[batch_size,1],the ending distance of the frustum.
        base_radius:torch.tensor(float32),[batch_size,1],the scale of the radius as a function of distance.
        diag:boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable:boolean, whether or not to use the stable computation described in the mipnerf paper (setting this to False will cause catastrophic failure).

    Returns:
        a contracted Gaussian (mean and covariance).
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

def para_rays(t_vals,origins,directions,radii,diag=False):
    """parameterize rays and return means and covs.

    Arguments:
        t_vals:torch.tensor,[batch_size,num_samples],sampled z values.
        origins:torch.tensor(float32),[batch_size,3],ray origins.
        directions:torch.tensor(float32),[batch_size,3],ray directions.
        radii:torch.tensor(float32),[batch_size,1],ray radii.
    
    Returns:
        means:torch.tensor,[batch_size,num_samples,3],sampled means.
        covs:torch.tensor,[batch_size,num_samples,3,3],sampled covariances.
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    means, covs = conical_frustum_to_gaussian(d=directions,t0=t0,t1=t1,base_radius=radii,diag=diag,stable=True)
    means = means + origins[...,None,:]
    return means,covs