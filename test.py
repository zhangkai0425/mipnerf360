import cv2
import torch
import shutil
import os.path
import numpy as np
from os import path
from tqdm import tqdm
import torch.optim as optim
from model import mipNeRF360
from config import get_config
from intern.utils import to8b
from intern.loss import Loss_nerf
import torch.utils.tensorboard as tb
from intern.scheduler import lr_decay
from dataset import get_dataloader, cycle
from intern.pose import visualize_depth, visualize_normals
from intern.loss import Loss_prop,Loss_nerf,Loss_dist,mse_to_psnr

def test_model(config):
    test_data = get_dataloader(config.dataset_name, config.base_dir, split="visualize", factor=config.factor, shuffle=False)
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
        # print("debug here::",pixels.shape,pixels[0])
        print("Evaluating the model:[{}/{}]:".format(index,len(test_data)))
        img, dist, acc = model.render_image(rays, test_data.h, test_data.w, chunks=config.chunks)
        cv2.imwrite(path.join(save_path,"rgb_{:04d}.png".format(index)), img)
        target_img = pixels.reshape(test_data.h, test_data.w, 3).numpy()
        mse = np.sum(((img / 255.).astype(np.float32) - target_img) ** 2)
        psnr = -10.0 * np.log10(mse)
        print("PSNR={}".format(psnr))
        
        if config.visualize_depth:
            dist_img = to8b(visualize_depth(dist, acc, test_data.near, test_data.far))
            cv2.imwrite(path.join(save_path,"dist_{:04d}.png".format(index)), dist_img)
        if config.visualize_normals:
            norm_img = to8b(visualize_normals(dist, acc))
            cv2.imwrite(path.join(save_path,"norm_{:04d}.png".format(index)), norm_img)
        index += 1
    print("Evaluating completed!")

if __name__ == "__main__":
    config = get_config()
    test_model(config)