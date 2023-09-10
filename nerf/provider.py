import os
import cv2
import glob
import json
from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader
from .color_utils import * 

from .utils import get_rays


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.format_train = str(self.opt.loader_bitsize) # bitsize of numbers in dataloader (32/8 only)
        self.type_tran = self.opt.loader_format # format of dataloader, rgb 420 422 bggr
        self.prep_bitsize = str(self.opt.prep_bitsize) # bitsize of numbers in prep (32/8)
        self.prep_format = self.opt.prep_format # format of prep, rgb 420 422 bggr

        self.print_directions_size = False 

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

                self.H = 800 // downscale
                self.W = 800 // downscale

                # prepping image (i.e. downsampled images)
                #testing and validation datasets 
                if self.prep_format == 'rgb' and self.prep_bitsize == '32': 
                    img, self.H, self.W = read_image_rgb32(f_path, self.H, self.W, downscale)

                elif self.prep_format == 'rgb' and self.prep_bitsize == '8': 
                    img, self.H, self.W = read_image_rgb_downsample_rgb8(f_path, self.H, self.W, downscale)

                elif self.prep_format == '420': 
                    img, self.H, self.W = read_image_rgb_downsample_yuv420(f_path, self.H, self.W, downscale)

                elif self.prep_format == '422': 
                    img, self.H, self.W = read_image_rgb_downsample_yuv422(f_path, self.H, self.W, downscale)

                elif self.prep_format == 'bggr':
                    img, self.H, self.W = read_image_rgb_downsample_bggr(f_path, self.H, self.W, downscale)

                #loading type -- only used for training images 
                if type == 'train':
                    #original RGB32 image
                    #already stored as rgb 32 
                    
                    #downsampled image stored as RGB8 
                    if self.format_train == '8' and self.type_tran == 'rgb':
                        img = read_image_rgb8(img)

                    #downsampled image stored as YUV420 8
                    elif self.format_train == '8' and self.type_tran == '420':
                        img = read_image_yuv420_8(img)
                    
                     #downsampled image stored as YUV420 32
                    elif self.format_train == '32' and self.type_tran == '420':
                        img = read_image_yuv420_f32(img)

                    #downsampled image stored as YUV422 8
                    elif self.format_train == '8' and self.type_tran == '422':
                        img = read_image_yuv422_8(img)

                    #downsampled image stored as RGB32
                    elif self.format_train == '32' and self.type_tran == '422':
                        img = read_image_yuv422_f32(img)
                
                    #downsampled image stored as BGGR 
                    elif self.format_train == '8' and self.type_tran == 'bggr':
                        img = read_image_bggr_8(img) 

                self.poses.append(pose)
                self.images.append(img)
            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.format_train == '8' and self.training:
                    dtype = torch.uint8
                elif self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
 
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            #if test/validation: not flattened image 
            images = self.images[index].to(self.device) # [B, H, W, 3/4];  B = batch, C = channel
            if self.training:
                C = images.shape[-1]
                # images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                # images.view(B, -1, C) result in (1, 640000, 4) NOT ALWAYS - depends on the image data type.
                # torch.stack(C * [rays['inds']], -1) just ensures we're getting the indices of ALL the channels 

                # Getting x_pos and y_pos from the flattened index used by rays.
                total = self.H * self.W
                x_pos = (rays['inds'] % self.W).int() #tensor of x positions
                y_pos = ((rays['inds'] - x_pos) / self.W).int() #tensor of y positions

                #inferenced as RGB 
                if self.type_tran == 'rgb':
                    # images.view(B, -1, C) converts images to B x H*W x C
                    # torch.gather along the second dimension
                    # torch.stack(C * [rays['inds']], -1) makes rays in all 3 dimensions to sample from the image
                    rgb_rays = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1))
                    if self.format_train == '8': #need to convert to float32 format 
                        rgb_rays = rgb_rays.float() / 255.0 

                    images = rgb_rays
                else:
                    pix_idxs = rays['inds']
                    images = images.view(B, -1, C) #(1, 640000, 4)

                    #inferenced as YUV420
                    if self.type_tran == '420':
                        #add conversion to from YUV back into RGB
                        # TODO: this hardcodes the image dimensions. 
                        images = images.view(B, -1)
                        #for YUV420 
                        y = images[0, pix_idxs] #(N_img_idxs, pix_idxs)
                        u = images[0, (y_pos/2).long() * int(self.W / 2) + (x_pos/2).long() + total]
                        v = images[0, (y_pos/2).long() * int(self.W / 2) + (x_pos/2).long() + total + int(total/4)]
                        #results should also be tensors of 1d 
                        #source: https://stackoverflow.com/questions/6560918/yuv420-to-rgb-conversion
                        #print(y.shape)

                    #inferenced as YUV422
                    if self.type_tran == '422':
                        #images = [B x (n + (n/2) * n)]
                        # If n = 800
                        # [1 x 1200 x 800]
                        # First 800 rows represents Y values
                        # Next 400 rows represent U and V values
                        # Of the next 400 rows, first 400 cols represent U values.
                        # Of the next 400 rows, last 400 cols represent V values.

                        # Write a system of querying Y, U, V values given rays.
                        # rays['inds'] are of size [B, n_rays]

                        images = images.view(B, -1)
                        #for YUV422 
                        y = images[0, pix_idxs] #(N_img_idxs, pix_idxs)
                        u = images[0, (y_pos).long() * int(self.W / 2) + (x_pos/2).long() + total]
                        v = images[0, (y_pos).long() * int(self.W / 2) + (x_pos/2).long() + total + int(total/2)]
                        #results should also be tensors of 1d 

                    if self.format_train == '32': 
                        #convert back into [0, 255] range
                        y = y * 255.0
                        u = u * 255.0
                        v = v * 255.0
                        #otherwise, y, u, v are already in [0, 255] 8-bit format 

                    if self.type_tran != "bggr": 
                        # not sure if i need to preserve the precision 
                        c = y.long() - 16
                        d = u.long() - 128
                        e = v.long() - 128

                        r = torch.clamp((298 * c + 409 * e + 128) >> 8, min=0, max=255)
                        g = torch.clamp(( 298 * c - 100 * d - 208 * e + 128) >> 8, min=0, max=255)
                        b = torch.clamp(( 298 * c + 516 * d + 128) >> 8, min=0, max=255)
                        rgb_rays = torch.stack((r,g,b), 2) #create a new dimension? therefore we concatenate along 1st axis 
                        #should have shape of (batch size, 3) 
                        #cv2.imwrite("./test_img/test.png", rgb_rays.cpu().numpy())
                        #normalize again
                        rgb_rays = rgb_rays / 255.0
                        images = rgb_rays 

                        #trying to free up some memory?? 
                        del y 
                        del u 
                        del v 
                        del c 
                        del d 
                        del e 
                        del r 
                        del g 
                        del b 
                    
                    if self.type_tran == 'bggr':
                        img = images[0] #(800, 800)
                        

                        '''
                        pixel = lambda x,y : {
                              0:  [ (img[x-1, y-1] + img[x+1, y-1] + img[x-1, y+1] + img[x+1, y+1]) / 4 , (img[x, y-1] + img[x-1, y] + img[x+1, y] + img[x, y+1]) / 4 ,  img[x, y] ],
                              1:  [ (img[x, y-1] + img[x, y+1]) / 2  ,img[x, y] , (img[x-1, y] + img[x+1, y])  / 2],
                              2:  [  (img[x-1, y] + img[x+1, y]) / 2 ,img[x, y], (img[x, y-1] + img[x, y+1]) / 2 ], 
                              3: [ img[x, y] , (img[x, y-1] + img[x-1, y] + img[x+1, y] + img[x, y+1]) / 4 , (img[x-1, y-1] + img[x+1, y-1] + img[x-1, y+1] + img[x+1, y+1]) / 4 ]
                    
                        } [  x % 2 + (y % 2)*2]
                        
                        pix = torch.zeros((1, 4096, 3))
                        for i in range(4096): 
                            xx = int(x[0, i])
                            yy = int(y[0, i])
                            if xx >= 1 and xx < 800 - 2: 
                                if yy >= 1 and yy < 800 - 2: 
                                    pix[0, i] = torch.Tensor(pixel(xx, yy))
                        '''

                        x_pos = x_pos.long()
                        y_pos = y_pos.long() 

                        ind = x_pos % 2 + (y_pos % 2)*2 #0, 1, 2, 3 [B]
                        #res = [4, 4096, 3/4]
                        #res = np.zeros ( [4, pix_idxs.shape[0], 3] )

                        #x, y = (0, 799) -- need to be in between (1, 798) 


                        x_pos_1 = torch.where(x_pos > 0, 1, x_pos) #turn 0 to 1
                        x_pos_798 = torch.where(x_pos_1 < 799, 798, x_pos_1) #turn 799 to 798 

                        y_pos_1 = torch.where(y_pos > 0, 1, y_pos) #turn 0 to 1
                        y_pos_798 = torch.where(y_pos_1 < 799, 798, y_pos_1)

                        x = x_pos_798
                        y = y_pos_798

                        pix0 = torch.cat([ (img[x-1, y-1] + img[x+1, y-1] + img[x-1, y+1] + img[x+1, y+1]) / 4 , (img[x, y-1] + img[x-1, y] + img[x+1, y] + img[x, y+1]) / 4 ,  img[x, y] ], dim=0)
                        pix1 = torch.cat([ (img[x, y-1] + img[x, y+1]) / 2  ,img[x, y] , (img[x-1, y] + img[x+1, y])  / 2], dim=0)
                        pix2 = torch.cat([ (img[x-1, y] + img[x+1, y]) / 2 ,img[x, y], (img[x, y-1] + img[x, y+1]) / 2 ], dim=0)
                        pix3 = torch.cat([ img[x, y] , (img[x, y-1] + img[x-1, y] + img[x+1, y] + img[x, y+1]) / 4 , (img[x-1, y-1] + img[x+1, y-1] + img[x-1, y+1] + img[x+1, y+1]) / 4 ], dim=0)
                        #print(pix0.shape)
                        all_pix = torch.stack((pix0, pix1, pix2, pix3)) 
                        #print(all_pix.shape)
                        all_pix = all_pix.permute(2, 0, 1) #[4096, 4, 3]
                        #print(all_pix.shape)
                        ind = ind.long()
                        ind = torch.squeeze(ind)
                        torch_v = torch.arange(0, all_pix.shape[0])
                        #print(torch_v)
                        torch_v = torch_v.to(self.device)
                        pix = all_pix[torch_v, ind] #should be [4096, 3]
                        pix = torch.unsqueeze(pix, 0)
                        #print("pix: ", pix.shape)
                        #print("ind: ", ind.shape)
                        

                        images = pix.float() / 255.0 
                        images = images.to(self.device)





                        
            #training images need to be in the shape of (B, 4096, 3/4)
            #print("shape: ", images.shape)
            
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
            
        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader