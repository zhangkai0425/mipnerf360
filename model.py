import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding
from torch.nn.modules.activation import Sigmoid

def _kaiming_init(model):
    """perform kaiming initialization to the model
    Arguments:
        model,nn.module

    Returns:
        model,nn.module
    """
    for module in model.modules():
        if isinstance(module,nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
"""
    还是有一些基本的想法：
    1.IPE的部分需要完成，但是需要完成在哪里呢？是否需要重新建一个文件呢？
    我认为是必要的，需要重建一个单独的postional_encoding.py，单独存这个东西
    这个的好处是什么呢？是非常直观，之后的修改代码也很容易
    还是那句话，架构务必要清晰
    2.第二点，需要把两个类的forward函数均完成，这一点可以采样函数先不真正实现，
    实现之后再具体实现采样函数
    3.实现完输入输出之后，在mipnerf360类中封装一下两个net，力求简洁
    4.下午最后一个任务是实现train.py中的蒸馏学习策略单元，这一点也比较麻烦
"""
class prop_net(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 hidden_proposal=256,
                 density_bias=-1,
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
        self.density_bias = density_bias
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.viewdirs_min_deg = viewdirs_min_deg
        self.viewdirs_max_deg = viewdirs_max_deg
        self.device = device

        # IPE module
        self.positional_encoding = PositionalEncoding(self.min_deg,self.max_deg)
        self.


        self.input_size = 0 #TODO: self.input

        # proposal network: depth = 4 width = 256
        self.proposal = nn.Sequential(
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
        s_vals, (mean, var) = sample_along_rays(rays.origins,rays.directions,rays.radii,self.num_samples,
                                                        rays.near,rays.far,randomized=self.randomized,lindisp=False)
        # integrated postional encoding(IPE) of samples
        samples_enc = self.
        return 0

class nerf_net(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
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
        self.hidden_nerf = hidden_nerf
        self.density_bias = density_bias
        self.rgb_padding = rgb_padding
        self.resample_padding = resample_padding
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.viewdirs_min_deg = viewdirs_min_deg
        self.viewdirs_max_deg = viewdirs_max_deg
        self.device = device

        self.input_size = 0 #TODO: self.input

        # nerf network: depth = 8 width = 1024
        self.nerf = nn.Sequential(
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
    
    def forward(self,rays):
        
        
        
        return 0
    


"""[summary]
计划把render实现在mipnerf360中
"""
class mipNeRF360(nn.Module):
    def __init__(self,
                 randomized=False,
                 num_samples=128,
                 num_levels=2,
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
        super(mipNeRF360, self).__init__()

        # parameters initialize
        self.randomized = randomized
        self.num_samples = num_samples
        self.num_levels = num_levels
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
        self.proposal = nn.Sequential(
            nn.Linear(self.input_size,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,self.hidden_proposal),
            nn.ReLU(True),
            nn.Linear(self.hidden_proposal,1) # output only density 
        )

        # nerf network: depth = 8 width = 1024
        self.nerf = nn.Sequential(
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
    
    def forward(self,rays):
        final_rgbs = []
        final_dist = []
        final_accs = []
        # two stages: proposal network and nerf network
        for l in range(self.num_levels):
            if l == 0:
                # stage 1: proposal network output density
                s_vals, (mean, var) = sample_along_rays(rays.origins,rays.directions,rays.radii,self.num_samples,
                                                        rays.near,rays.far,randomized=self.randomized,lindisp=False)
            else: 
                # stage 2: nerf network output density and 
                s_vals, (mean, var) = resample_along_rays(rays.origins,rays.directions,rays.radii,
                                                          s_vals.to(rays.origins.device),weights.to(rays.origins.device),
                                                          randomized=self.randomized,stop_grad=True,resample_padding=self.resample_padding)
            # integrated positional encoding(IPE) of samples
            samples_enc = self.postional
                    


        return 0
    def render_image(self,rays,height,width,chunks=4096):
        return 0
    
        




# 拓展到200行，提交10个commit
