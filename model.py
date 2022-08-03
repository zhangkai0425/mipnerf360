import torch
import torch.nn as nn
from ray import sample_along_rays,resample_along_rays,volumetric_rendering
from encoding import PositionalEncoding,ViewdirectionEncoding
from torch.nn.modules.activation import Sigmoid

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
    
    def forward(self,rays):
        #TODO: sample_along_rays很难实现，真的
        # sample
        s_vals, (mean, var) = sample_along_rays(origins=rays.origins,directions=rays.directions,radii=rays.radii,num_samples=self.num_samples,
                                                        near=rays.near,far=rays.far)
        # integrated postional encoding(IPE) of samples
        samples_enc = self.positional_encoding(mean,var)
        viewdirs_enc = self.viewdirs_encoding(rays.viewdirs.to(self.device))
        input_enc = torch.cat((samples_enc,viewdirs_enc),-1)

        # predict density
        raw_density = self.model(input_enc)
        density = self.density_activation(raw_density + self.density_bias)
    
        return s_vals,density

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
    
    def forward(self,rays,t_vals,weights):
        final_rgbs = []
        final_dist = []
        final_accs = []
        # sample
        # 根据proposal net预测结果进行重采样 TODO:此处最难！最难！
        t_vals,(mean,var) = resample_along_rays(origins=rays.origins,directions=rays.directions,radii=rays.radii,
                            t_vals=t_vals.to(rays.origins.device),weights=weights,randomized=self.randomized,resample_padding=self.resample_padding)
        
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
        
        final_rgbs.append(comp_rgb)
        final_dist.append(distance)
        final_accs.append(acc)

        # save the weights of nerf_net,used in the distillation section
        self.weights = weights

        # return everything 
        if self.return_raw:
            raws = torch.cat((torch.clone(rgb).detach(), torch.clone(density).detach()), -1).cpu()
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(final_rgbs), torch.stack(final_dist), torch.stack(final_accs), raws
        else:
            # Predicted RGB values for rays, Disparity map (inverse of depth), Accumulated opacity (alpha) along a ray
            return torch.stack(final_rgbs), torch.stack(final_dist), torch.stack(final_accs)

class mipNeRF360(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 hidden_proposal=256,
                 hidden_nerf=1024,
                 density_bias=-1,
                 rgb_padding=0.001,
                 resample_padding=0.01,
                 min_deg=0,
                 max_deg=16,
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
        self.resample_padding = resample_padding #TODO:不知何用，予以保留
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.viewdirs_min_deg = viewdirs_min_deg
        self.viewdirs_max_deg = viewdirs_max_deg
        self.device = device

        self.input_size = 0 #TODO: self.input

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

    def render_image(self,rays,height,width,chunks=4096):
        return 0