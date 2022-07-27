import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg, max_deg):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
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
                    ])
        self.scales = nn.Parameter(torch.tensor([2 ** i for i in range(min_deg, max_deg)]), requires_grad=False)

    def forward(self, x, y=None):
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return x_ret