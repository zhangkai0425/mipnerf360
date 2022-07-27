import torch
from torch._C import dtype
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self,min_deg,max_deg):
        super().__init__()
        # used for encoding theta/phi
        self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)],dtype=torch.float32,requires_grad=False)
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
                    ],requires_grad=False)

    def forward(self, mean, cov,view_dir=False,theta=None,phi=None):
        """forward function of positional encoding

        Arguments:
            mean,torch.float32,shape(batch_size,num_samples,3),mean of each num_samples of xyz(contracted)
            cov,torch.float32,shape(batch_size,num_samples,3,3),cov of each num_samples of xyz(contracted)
            view_dir,bool,encoding xyz or theta/phi

        Returns:
            enc,torch.float32,shape(batch_size,num_samples,21*2,3),postional_encoding
        """
        if view_dir:
            # encoding the theta and phi
            theta = self.scales * theta
            phi = self.scales * phi
            enc = torch.cat((torch.sin(theta),torch.cos(theta),torch.sin(phi),torch.cos(phi)),-1)
            return enc

        mean_gamma = torch.matmul(self.P,mean)
        if cov is not None:
            # IPE
            A = torch.matmul(cov,self.P.T)
            sigma = torch.sum(self.P.T * A,dim=3) #TODO: 这里维度需要再仔细注意！
            enc_sin = torch.exp(-0.5*sigma) * torch.sin(mean_gamma)
            enc_cos = torch.exp(-0.5*sigma) * torch.cos(mean_gamma)
            enc = torch.cat((enc_sin,enc_cos),-1)
            return enc

        else:
            # PE
            enc = torch.cat((torch.sin(mean_gamma),torch.cos(mean_gamma)),-1)
            return enc