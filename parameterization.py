import torch
import torch.nn as nn

def contract(x):
    """[summary]
    contract function of x,used in parameterization
    Arguments:
        x {[torch.tensor]} -- [the parameter of t along the ray]

    Returns:
        [contracted x] -- [x contracted to a zone with the radius of 2]
    """
    x_norm = torch.norm(x)
    if x_norm <= 1:
        return x
    else:
        return (2-1/x_norm)*(x/x_norm)
