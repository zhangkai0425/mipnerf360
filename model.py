import torch
import torch.nn as nn
from ray import sample_along_rays,resample_along_rays,volumetric_rendering
from encoding import PositionalEncoding,ViewdirectionEncoding
from torch.nn.modules.activation import Sigmoid
from regularization import t_to_s

def _kaiming_init(model):
    """perform kaiming initialization to the model"""
    for module in model.modules():
        if isinstance(module,nn.Linear):
            nn.init.kaiming_uniform_(module.weight)

class prop_net(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 hidden_proposal=256,
                 density_bias=-1,
                 viewdir_min_deg=0,
                 viewdir_max_deg=4,
                 device=torch.device("cuda"),
                 ):
        super().__init__()

        # parameters initialize
        self.randomized = randomized
        self.num_samples = num_samples
        self.hidden_proposal = hidden_proposal
        self.density_bias = density_bias
        self.viewdir_min_deg = viewdir_min_deg
        self.viewdir_max_deg = viewdir_max_deg
        self.device = device

        # IPE module
        self.positional_encoding = PositionalEncoding()
        self.viewdirs_encoding = ViewdirectionEncoding(self.viewdirs_min_deg,self.viewdirs_max_deg)

        self.input_size = 21*3*2 + (self.viewdir_max_deg-self.viewdir_min_deg) * 2 * 2 #TODO: self.input
        self.density_activation = nn.Softplus()

        # proposal network: depth = 4 width = 256
        self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.Sigmoid(),
            nn.Linear(self.hidden_proposal,1) # output only density 
        )

        # initialize the model ang put the model to device
        _kaiming_init(self)
        self.to(device)
        
    def density_to_weight(self,t_vals,density,dirs):
        """Transform density to weights

        Arguments:
            t_vals:torch.tensor(float32), [batch_size, num_samples].
            density:torch.tensor(float32), density, [batch_size, num_samples, 1].
            dirs:torch.tensor(float32), [batch_size, 3].

        Returns:
            weights:torch.tensor(float32),[batch_size,num_samples],weights along the rays.
        """
        t_dists = t_vals[..., 1:] - t_vals[..., :-1]
        delta = t_dists * torch.linalg.norm(dirs[..., None, :], dim=-1)
        density_delta = density[..., 0] * delta

        alpha = 1 - torch.exp(-density_delta)
        trans = torch.exp(-torch.cat([torch.zeros_like(density_delta[..., :1]),
        torch.cumsum(density_delta[..., :-1], dim=-1)], dim=-1))
        weights = alpha * trans
        return weights

    def forward(self,rays):
        # sample
        t_vals, (mean, var) = sample_along_rays(origins=rays.origins,directions=rays.directions,radii=rays.radii,num_samples=self.num_samples,
                                                        near=rays.near,far=rays.far,randomized=self.randomized)
        # integrated postional encoding(IPE) of samples
        samples_enc = self.positional_encoding(mean,var)
        viewdirs_enc = self.viewdirs_encoding(rays.viewdirs.to(self.device))
        input_enc = torch.cat((samples_enc,viewdirs_enc),-1)

        # predict density and return weights
        raw_density = self.model(input_enc)
        density = self.density_activation(raw_density + self.density_bias)
        weights = self.density_to_weight(t_vals=t_vals,density=density,dirs=rays.directions.to(density.device))
        return t_vals,weights

class nerf_net(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 hidden_nerf=1024,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 viewdir_min_deg=0,
                 viewdir_max_deg=4,
                 device=torch.device("cuda"),
                 ):
        super().__init__()  

        # parameters initialize
        self.randomized = randomized
        self.num_samples = num_samples
        self.hidden_nerf = hidden_nerf
        self.density_bias = density_bias
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.viewdir_min_deg = viewdir_min_deg
        self.viewdir_max_deg = viewdir_max_deg
        self.device = device

        # IPE module
        self.positional_encoding = PositionalEncoding()
        self.viewdirs_encoding = ViewdirectionEncoding(self.viewdir_min_deg,self.viewdir_max_deg)

        self.input_size = 21*3*2 + (self.viewdir_max_deg-self.viewdir_min_deg) * 2 * 2 
        self.density_activation = nn.Softplus()

        # nerf network: depth = 8 width = 1024
        self.model = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.ReLU(True),
            nn.Linear(self.hidden_nerf,self.hidden_nerf),
            nn.Sigmoid()
        )
        # output the final density of nerf network
        self.final_density = nn.Sequential(
            nn.Linear(self.hidden_nerf,1),
            nn.Sigmoid()
        )   
        # output the final color of nerf network
        self.final_color = nn.Sequential(
            nn.Linear(self.hidden_nerf,3),
            nn.Sigmoid()
        )
        # initialize the model ang put the model to device
        _kaiming_init(self)
        self.to(device)
    
    def forward(self,rays,t_vals,coarse_weights):
        final_rgbs = []
        final_dist = []
        final_accs = []

        # resample
        t_vals,(mean,var) = resample_along_rays(origins=rays.origins,directions=rays.directions,radii=rays.radii,
                            t_vals=t_vals.to(rays.origins.device),weights=coarse_weights,randomized=self.randomized,resample_padding=self.resample_padding)
        
        # integrated postional encoding(IPE) of samples
        samples_enc = self.positional_encoding(mean=mean,var=var)
        viewdirs_enc = self.viewdirs_encoding(rays.viewdirs.to(self.device))
        input_enc = torch.cat((samples_enc,viewdirs_enc),-1)

        # predict density and color
        feature = self.model(input_enc)
        raw_density = self.final_density(feature)
        raw_rgb = self.final_color(feature)

        # volumetric rendering
        rgb = raw_rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
        density = self.density_activation(raw_density + self.density_bias)
        comp_rgb,distance,acc,weights = volumetric_rendering(rgb=rgb, density=density, t_vals=t_vals, dirs=rays.directions.to(rgb.device),white_bkgd=self.white_bkgd)
        
        final_rgbs = comp_rgb
        final_dist = distance
        final_accs = acc

        # save the weights and t_vals of nerf_net,used in the distillation section
        self.fine_weights = weights
        self.t_vals = t_vals
        # save the s_vals of nerf_net,used in the regularization section
        self.s_vals = t_to_s(t_vals=t_vals,near=rays.near,far=rays.far)

        # return everything 
        # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray,Fine weights,S vals
        return final_rgbs, final_dist, final_accs, self.t_vals, self.fine_weights, self.s_vals

class mipNeRF360(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 hidden_proposal=256,
                 hidden_nerf=1024,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 viewdirs_min_deg=0,
                 viewdirs_max_deg=4,
                 device=torch.device("cuda"),
                 ):
        super().__init__()

        # parameters initialize
        self.randomized = randomized
        self.num_samples = num_samples
        self.hidden_proposal = hidden_proposal
        self.hidden_nerf = hidden_nerf
        self.density_bias = density_bias
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding 
        self.viewdirs_min_deg = viewdirs_min_deg
        self.viewdirs_max_deg = viewdirs_max_deg
        self.device = device

        # proposal network: depth = 4 width = 256
        self.prop_net = prop_net(randomized=self.randomized,num_samples=self.num_samples,
                        hidden_proposal=self.hidden_proposal,density_bias=self.density_bias,
                        viewdir_min_deg=self.viewdirs_min_deg,viewdir_max_deg=self.viewdirs_max_deg,
                        device=self.device)

        # nerf network: depth = 8 width = 1024
        self.nerf_net = nerf_net(randomized=self.randomized,num_samples=self.num_samples,
                        hidden_nerf=self.hidden_nerf,density_bias=self.density_bias,rgb_padding=self.rgb_padding,
                        resample_padding=self.resample_padding,viewdir_min_deg=self.viewdirs_min_deg,viewdir_max_deg=self.viewdirs_max_deg,
                        device=self.device)

        self.to(device)

    def forward(self,rays):
        #TODO:forward to output final density and color
        return 0
    
    def render_image(self,rays,height,width,chunks=4096):
        #TODO:use the final density and color to render a complete image
        return 0