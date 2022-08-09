import torch
import shutil
import os.path
import numpy as np
from os import path
import torch.optim as optim
from model import mipNeRF360
from config import get_config
from scheduler import lr_decay
import torch.utils.tensorboard as tb
from dataset import get_dataloader, cycle
from loss import Loss_prop,Loss_nerf,Loss_dist,mse_to_psnr

def test_model(config):
    test_data = iter(cycle(get_dataloader(dataset_name=config.dataset_name, base_dir=config.base_dir, split="test", factor=config.factor, batch_size=config.batch_size, shuffle=True, device=config.device)))
    model = mipNeRF360(
        randomized=config.randomized,
        num_samples=config.num_samples,
        hidden_proposal=config.hidden_proposal,
        hidden_nerf=config.hidden_nerf,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        viewdir_min_deg=config.viewdir_min_deg,
        viewdir_max_deg=config.viewdir_max_deg,
        device=config.device
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()
    
