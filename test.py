import cv2
import torch
import shutil
import os.path
import numpy as np
from os import path
from tqdm import tqdm
from utils import to8b
from loss import Loss_nerf
import torch.optim as optim
from model import mipNeRF360
from config import get_config
from scheduler import lr_decay
import torch.utils.tensorboard as tb
from dataset import get_dataloader, cycle
from pose import visualize_depth, visualize_normals
from loss import Loss_prop,Loss_nerf,Loss_dist,mse_to_psnr

def test_model(config):
    test_data = get_dataloader(config.dataset_name, config.base_dir, split="test", factor=config.factor, shuffle=False)
    model = mipNeRF360(
        randomized=config.randomized,
        num_samples=config.num_samples,
        hidden_proposal=config.hidden_proposal,
        hidden_nerf=config.hidden_nerf,
        density_bias=config.density_bias,
        rgb_padding=config.rgb_padding,
        resample_padding=config.resample_padding,
        white_bkgd=config.white_bkgd,
        viewdir_min_deg=config.viewdir_min_deg,
        viewdir_max_deg=config.viewdir_max_deg,
        device=config.device
    )
    model.load_state_dict(torch.load(config.model_weight_path))
    model.eval()

    print("Evaluating model on", len(test_data), "different view directions")
    
    save_path = path.join(config.log_dir,"test")
    os.makedirs(save_path, exist_ok=True)

    for index,(rays,pixels) in enumerate(test_data):
        print("Evaluating the model:[{}/{}]:".format(index,len(test_data)))
        img, dist, acc = model.render_image(rays, test_data.h, test_data.w, chunks=config.chunks)
        target_img = to8b(torch.cat(pixels, dim=0).reshape(test_data.h, test_data.w, 3).numpy())
        _,psnr = Loss_nerf(input=img,target=target_img)
        print("PSNR={}".format(psnr))
        cv2.imwrite(path.join(save_path,"rgb_{:04d}.png".format(index)), img)
        if config.visualize_depth:
            dist = to8b(visualize_depth(dist, acc, test_data.near, test_data.far))
            cv2.imwrite(path.join(save_path,"dist_{:04d}.png".format(index)), dist)
        if config.visualize_normals:
            norm = to8b(visualize_normals(dist, acc))
            cv2.imwrite(path.join(save_path,"norm_{:04d}.png".format(index)), norm)
        index += 1
    print("Evaluating completed!")
