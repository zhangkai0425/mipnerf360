import os
import cv2
import json
import torch
import numpy as np
from os import path
from PIL import Image
from intern.utils import normalize,to_float
from torch.utils.data import Dataset, DataLoader
from intern.ray import Rays,convert_to_ndc,namedtuple_map
from intern.pose import generate_spherical_cam_to_world,generate_spiral_cam_to_world,recenter_poses,poses_avg,look_at


def get_dataset(dataset_name, base_dir, split, factor=4, device=torch.device("cuda")):
    d = dataset_dict[dataset_name](base_dir, split, factor=factor, device=device)
    return d

def get_dataloader(dataset_name, base_dir, split, factor=4, batch_size=None, shuffle=True, device=torch.device("cuda")):
    d = get_dataset(dataset_name, base_dir, split, factor, device)
    # make the batchsize height*width, so that one "batch" from the dataloader corresponds to one
    # image used to render a video, and don't shuffle dataset
    if split == "render" or split == "visualize":
        batch_size = d.w * d.h
        shuffle = False
    loader = DataLoader(d, batch_size=batch_size, shuffle=shuffle)
    loader.h = d.h
    loader.w = d.w
    loader.near = d.near
    loader.far = d.far
    return loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class NeRFDataset(Dataset):
    """NeRF dataset:a base class"""
    def __init__(self, base_dir, split, spherify=False, near=2, far=6, white_bkgd=False, factor=1, n_poses=120, radius=None, radii=None, h=None, w=None, device=torch.device("cuda")):
        super().__init__()
        self.base_dir = base_dir
        self.split = split
        self.spherify = spherify
        self.near = near
        self.far = far
        self.white_bkgd = white_bkgd
        self.factor = factor
        self.n_poses = n_poses
        self.n_poses_copy = n_poses
        self.radius = radius
        self.radii = radii
        self.h = h
        self.w = w
        self.device = device
        self.rays = None
        self.images = None
        self.load()

    def load(self):
        if self.split == "render":
            self.generate_render_rays()
        else:
            self.generate_training_rays()

        self.flatten_to_pytorch()
        print('Done')

    def generate_training_poses(self):
        """
        Generate training poses, datasets should implement this function to load the proper data from disk.
        Should initialize self.h, self.w, self.focal, self.cam_to_world, and self.images
        """
        raise ValueError('no generate_training_poses(self).')

    def generate_render_poses(self):
        """Generate arbitrary poses (views)"""
        self.focal = 1200
        self.n_poses = self.n_poses_copy
        if self.spherify:
            self.generate_spherical_poses(self.n_poses)
        else:
            self.generate_spiral_poses(self.n_poses)

    def generate_spherical_poses(self, n_poses=120):
        self.poses = generate_spherical_cam_to_world(self.radius, n_poses)
        self.cam_to_world = self.poses[:, :3, :4]

    def generate_spiral_poses(self, n_poses=120):
        self.cam_to_world = generate_spiral_cam_to_world(self.radii, self.focal, n_poses)

    def generate_training_rays(self):
        """
        Generates rays to train mip-NeRF
        """
        print("Loading Training Poses")
        self.generate_training_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_render_rays(self):
        """
        Generates rays used to render a video using a trained mip-NeRF
        """
        print("Generating Render Poses")
        self.generate_render_poses()
        print("Generating rays")
        self.generate_rays()

    def generate_rays(self):
        """Computes rays using a General Pinhole Camera Model
        Assumes self.h, self.w, self.focal, and self.cam_to_world exist
        """
        x, y = np.meshgrid(
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_directions = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal,
             -np.ones_like(x)],
            axis=-1)
        # Rotate ray directions from camera frame to the world frame
        directions = ((camera_directions[None, ..., None, :] * self.cam_to_world[:, None, None, :3, :3]).sum(axis=-1))  # Translate camera frame's origin to the world frame
        origins = np.broadcast_to(self.cam_to_world[:, None, None, :3, -1], directions.shape)
        # unit vector
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor
        dx = np.sqrt(np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])

        self.rays = Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            near=ones * self.near,
            far=ones * self.far
            )

    def flatten_to_pytorch(self):
        # flatten the data,making it easy to index
        if self.rays is not None:
            self.rays = namedtuple_map(lambda r: torch.tensor(r).float().reshape([-1, r.shape[-1]]), self.rays)
        if self.images is not None:
            self.images = torch.from_numpy(self.images.reshape([-1, 3]))

    def ray_to_device(self, rays):
        return namedtuple_map(lambda r: r.to(self.device), rays)

    def __getitem__(self, i):
        ray = namedtuple_map(lambda r: r[i], self.rays)
        if self.split == "render":
            # render rays
            return ray  # Don't put on device, will batch it using config.chunks in mipNeRF.render_image() function.
        else:
            # training rays
            pixel = self.images[i]  # Don't put pixel on device yet, waste of space.
            return self.ray_to_device(ray), pixel

    def __len__(self):
        if self.split == "render":
            return self.rays[0].shape[0]
        else:
            return len(self.images)


class Blender(NeRFDataset):
    """Blender Dataset."""
    def __init__(self, base_dir, split, factor=1, spherify=False, white_bkgd=True, near=2, far=6, radius=4, radii=1, h=800, w=800, device=torch.device("cuda")):
        super(Blender, self).__init__(base_dir, split, factor=factor, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, radius=radius, radii=radii, h=h, w=w, device=device)

    def generate_training_poses(self):
        """Load data from disk."""
        print("Loading Blender Dataset")
        split_dir = self.split
        with open(path.join(self.base_dir, 'transforms_{}.json'.format(split_dir)), 'r') as fp:
            meta = json.load(fp)
        images = []
        cams = []
        for i in range(len(meta['frames'])):
            frame = meta['frames'][i]
            fname = os.path.join(self.base_dir, frame['file_path'] + '.png')
            with open(fname, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                if self.factor >= 2:
                    [halfres_h, halfres_w] = [hw // 2 for hw in image.shape[:2]]
                    image = cv2.resize(
                        image, (halfres_w, halfres_h), interpolation=cv2.INTER_AREA)
            cams.append(np.array(frame['transform_matrix'], dtype=np.float32))
            images.append(image)
        self.images = np.stack(np.array(images), axis=0)
        if self.white_bkgd:
            self.images = (
                    self.images[..., :3] * self.images[..., -1:] +
                    (1. - self.images[..., -1:]))
        else:
            self.images = self.images[..., :3]
        self.h, self.w = self.images.shape[1:3]
        self.cam_to_world = np.stack(cams, axis=0)
        camera_angle_x = float(meta['camera_angle_x'])
        self.focal = .5 * self.w / np.tan(.5 * camera_angle_x)
        self.n_poses = self.images.shape[0]


class LLFF(NeRFDataset):
    def __init__(self, base_dir, split, factor=4, spherify=False, near=0, far=1, white_bkgd=False, device=torch.device("cpu")):
        super(LLFF, self).__init__(base_dir, split, spherify=spherify, near=near, far=far, white_bkgd=white_bkgd, factor=factor, device=device)
    def generate_training_poses(self):
        """Load data from disk."""
        img_dir = 'images'
        if self.factor != 1:
            img_dir = 'images_' + str(self.factor)
        img_dir = path.join(self.base_dir,img_dir)
        img_files = [
            path.join(img_dir, f)
            for f in sorted(os.listdir(img_dir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        # print("debug here::images_file = ",img_files)
        images = []
        for img_file in img_files:
            with open(img_file, 'rb') as img_in:
                image = to_float(np.array(Image.open(img_in)))
                images.append(image)
        images = np.stack(images, -1)

        # load poses
        with open(path.join(self.base_dir, 'poses_bounds.npy'), 'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        # Update poses according to downsampling.
        poses[:2, 4, :] = np.array(images.shape[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.factor
        # Correct rotation matrix ordering and move variable dim to axis 0.
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(images, -1, 0)
        self.images = images
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        # Rescale according to a default bd factor.
        scale = 1. / (bds.min() * .75)
        poses[:, :3, 3] *= scale
        bds *= scale
        self.bds = bds
        # Recenter poses.
        poses = recenter_poses(poses)
        self.poses = poses
        self.images = images
        self.h, self.w = images.shape[1:3]
        self.n_poses = images.shape[0]
    
    def generate_render_poses(self):
        print("Generating render poses")
        self.generate_training_poses()
        self.n_poses = self.n_poses_copy  # get overwritten in generate_training_poses, change back to original
        if self.spherify:
            self.generate_spherical_poses(self.n_poses)
        else:
            self.generate_spiral_poses(self.n_poses)
        self.cam_to_world = self.poses[:, :3, :4]
        self.focal = self.poses[0, -1, -1]
    
    def generate_training_rays(self):
        print("Loading training poses")
        self.generate_training_poses()
        if self.split == "train":
            indices = [i for i in np.arange(self.images.shape[0]) if i not in np.arange(self.images.shape[0])[::8]]
        else:
            indices = np.arange(self.images.shape[0])[::8]
        self.images = self.images[indices]
        self.poses = self.poses[indices]
        self.cam_to_world = self.poses[:, :3, :4]
        self.focal = self.poses[0, -1, -1]
        print("Generating rays")
        self.generate_rays()
    
    def generate_spherical_poses(self, n_poses=120):
        """Generate a 360 degree spherical path for rendering."""
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = self.poses[:, :3, 2:3]
        rays_o = self.poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (self.poses[:, :3, 3] - center).mean(0)
        vec0 = normalize(up)
        vec1 = normalize(np.cross([.1, .2, .3], vec0))
        vec2 = normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(self.poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        self.bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, n_poses):
            cam_origin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = normalize(cam_origin)
            vec0 = normalize(np.cross(vec2, up))
            vec1 = normalize(np.cross(vec2, vec0))
            pos = cam_origin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        self.poses = np.concatenate([
            new_poses,
            np.broadcast_to(self.poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
    
    def generate_spiral_poses(self, n_poses=120):
        """Generate a spiral path for rendering."""
        c2w = poses_avg(self.poses)
        # Get average pose.
        up = normalize(self.poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = self.bds.min() * .9, self.bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = self.poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_poses + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([look_at(z, up, c), hwf], 1))
        self.poses = np.array(render_poses).astype(np.float32)

    def generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        super().generate_rays()
        ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins, self.rays.directions, self.focal, self.w, self.h)
        mat = ndc_origins
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

        dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
        dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(ndc_origins[..., :1])
        # print("debug here::rays::",ndc_origins.shape,radii.shape)
        self.rays = Rays(
            origins=ndc_origins,
            directions=ndc_directions,
            viewdirs=self.rays.viewdirs,
            radii=radii,
            near=ones * self.near,
            far=ones * self.far)

dataset_dict = {
    'blender': Blender,
    'llff': LLFF,
}