import torch
from torch._C import device, dtype
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        # used for encoding contracted xyz
        self.P = torch.tensor([[0.8506508,0,0.5257311],
                  [0.809017,0.5,0.309017],
                  [0.5257311,0.8506508,0],
                  [1,0,0],
                  [0.809017,0.5,-0.309017],
                  [0.8506508,0,-0.5257311],
                  [0.309017,0.809017,-0.5],
                  [0,0.5257311,-0.8506508],
                  [0.5,0.309017,-0.809017],
                  [0,1,0],
                  [-0.5257311,0.8506508,0],
                  [-0.309017,0.809017,-0.5],
                  [0,0.5257311,0.8506508],
                  [-0.309017,0.809017,0.5],
                  [0.309017,0.809017,0.5],
                  [0.5,0.309017,0.809017],
                  [0.5,-0.309017,0.809017],
                  [0,0,1],
                  [-0.5,0.309017,0.809017],
                  [-0.809017,0.5,0.309017],
                  [-0.809017,0.5,-0.309017]
                    ],requires_grad=False) # shape:[21,3]
        

    def forward(self, mean, cov):
        """forward function of positional encoding

        Arguments:
            mean:torch.float32,[batch_size,num_samples,3],mean of each num_samples of xyz(contracted)
            cov:torch.float32,[batch_size,num_samples,3,3],cov of each num_samples of xyz(contracted)

        Returns:
            enc:torch.float32,[batch_size,num_samples,21*2],postional_encoding
        """
        self.P = self.P.to(device=mean.device)
        mean = mean[...,None]
        mean_gamma = torch.matmul(self.P,mean).squeeze(dim=-1)
        
        if cov is not None:
            # IPE
            A = torch.matmul(cov,self.P.T)
            # A:shape = [batch_size,num_samples,3,21]
            sigma = torch.sum(self.P.T * A,dim=2) 
            # sigma:shape = [batch_size,num_samples,21]
            enc_sin = torch.exp(-0.5*sigma) * torch.sin(mean_gamma)
            enc_cos = torch.exp(-0.5*sigma) * torch.cos(mean_gamma)
            enc = torch.cat((enc_sin,enc_cos),-1)
            return enc

        else:
            # PE
            enc = torch.cat((torch.sin(mean_gamma),torch.cos(mean_gamma)),-1)
            return enc

class ViewdirectionEncoding(nn.Module):
    def __init__(self,viewdir_min_deg,viewdir_max_deg):
        super().__init__()
        # used for encoding theta/phi
        self.scales = torch.tensor([2 ** i for i in range(viewdir_min_deg, viewdir_max_deg)],dtype=torch.float32,requires_grad=False)

    def forward(self,viewdirs):
        """forward function of viewdirection encoding

        Arguments:
            viewdirs:torch.tensor(float32),[batchsize,3],view direction of each samples
            
        Returns:
            enc:torch.float32,[batch_size,num_samples,(viewdir_max_deg-viewdir_min_deg)*2],viewdirection_encoding
        """
        # compute theta and phi using viewdirs unit normal vector:n=(x,y,z)
        self.scales = self.scales.to(device=viewdirs.device)
        x,y,z = viewdirs[...,0,None],viewdirs[...,1,None],viewdirs[...,2,None]
        theta = torch.arccos(z)
        phi = torch.arctan(y/(x+1e-6))
        print

        # encoding the theta and phi
        theta = self.scales * theta
        phi = self.scales * phi

        enc = torch.cat((torch.sin(theta),torch.cos(theta),torch.sin(phi),torch.cos(phi)),-1)
        return enc