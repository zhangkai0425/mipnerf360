import torch
import imageio
from os import path
from tqdm import tqdm
from model import mipNeRF360
from intern.utils import to8b
from config import get_config
from dataset import get_dataloader
from intern.pose import visualize_depth, visualize_normals


def visualize(config):
    data = get_dataloader(config.dataset_name, config.base_dir, split="render", factor=config.factor, shuffle=False)

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

    print("Generating Video using", len(data), "different view points")
    rgb_frames = []
    if config.visualize_depth:
        depth_frames = []
    if config.visualize_normals:
        normal_frames = []
    for ray in tqdm(data):
        img, dist, acc = model.render_image(ray, data.h, data.w, chunks=config.chunks)
        rgb_frames.append(img)
        if config.visualize_depth:
            depth_frames.append(to8b(visualize_depth(dist, acc, data.near, data.far)))
        if config.visualize_normals:
            normal_frames.append(to8b(visualize_normals(dist, acc)))

    imageio.mimwrite(path.join(config.log_dir, "video.mp4"), rgb_frames, fps=30, quality=10)
    if config.visualize_depth:
        imageio.mimwrite(path.join(config.log_dir, "depth.mp4"), depth_frames, fps=30, quality=10)
    if config.visualize_normals:
        imageio.mimwrite(path.join(config.log_dir, "normals.mp4"), normal_frames, fps=30, quality=10)


if __name__ == "__main__":
    config = get_config()
    visualize(config)
