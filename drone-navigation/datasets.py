"""
Dataset classes.
"""

import os
import io
import numpy as np
from PIL import Image
import torch
import json

from utils.dataset_utils import normalize_v
from utils.zipreader import is_zip_path, ZipReader


def pil_loader(path: str) -> Image.Image:
    """ PIL image loader.
        Ref: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def zipped_pil_loader(path):
    """ PIL image loader that supports zipped images.
        Ref: https://github.com/SwinTransformer/Transformer-SSL/blob/main/data/cached_image_folder.py#L179
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    if isinstance(path, bytes):
        img = Image.open(io.BytesIO(path))
        return img.convert('RGB')
    elif is_zip_path(path):
        data = ZipReader.read(path)
        img = Image.open(io.BytesIO(data))
        return img.convert('RGB')
    else:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')  # Put the return in the with block to avoid error 'ValueError: seek of closed file'.


def zipped_numpy_loader(path):
    """ NumPy loader that supports zipped files. """
    if isinstance(path, bytes):
        x = np.load(io.BytesIO(path))
    elif is_zip_path(path):
        data = ZipReader.read(path)
        x = np.load(io.BytesIO(data))
    else:
        with open(path, 'rb') as f:
            x = np.load(f)
    return x


class ILDataset(torch.utils.data.Dataset):
    """ Imitation learning (IL) dataset in drone racing. """
    def __init__(self, dataset_dir, ann_file_name, transform):
        # Load image paths and velocities from annotation file.
        self.dataset_dir = dataset_dir
        ann_path = os.path.join(dataset_dir, ann_file_name)
        self.img_rel_paths = []
        self.vels = []

        _, ext = os.path.splitext(os.path.basename(ann_path))
        ext = ext.lower()
        if ext == '.txt':  # Image annotations.
            with open(ann_path, 'r') as f:
                for line in f:
                    values = line.strip().split(',')
                    img_rel_path = values[0]                   # 1st value is relative path of image file.
                    vel = [float(x) for x in values[1:]]       # 2nd-5th values are for velocity.
                    self.img_rel_paths.append(img_rel_path)
                    self.vels.append(vel)
        elif ext == '.json':  # Video annotations.
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
            assert self.ann['type'] == 'video'
            for video_name in self.ann['ann']:
                video = self.ann['ann'][video_name]
                for frame_ann in video:
                    self.img_rel_paths.append(frame_ann['img_rel_path'])
                    self.vels.append(frame_ann['vel'])
        else:
            raise ValueError()

        self.vels = np.array(self.vels, dtype=np.float32)

        # Normalize velocities.
        self.vels = normalize_v(self.vels)
        
        # Other settings.
        self.transform = transform
        
    def __len__(self):
        return len(self.img_rel_paths)

    def __getitem__(self, index):
        # Load image.
        img_path = os.path.join(self.dataset_dir, self.img_rel_paths[index])
        img_arr = pil_loader(img_path)
        img = self.transform(img_arr)

        # Load velocity.
        vel = self.vels[index]

        item = {'img': img, 'vel': vel}
        return item


class ILVideoDataset(torch.utils.data.Dataset):
    """ Imitation learning (IL) video dataset in drone racing. """
    def __init__(self, dataset_dir, ann_file_name, zip_file_name, transform, clip_len=8, use_flow=False, use_flow_vis_as_img=False,
            use_depth_vis_as_img=False):
        self.dataset_dir = dataset_dir
        self.clip_len = clip_len
        self.zip_file_name = zip_file_name
        self.use_flow = use_flow
        self.use_flow_vis_as_img = use_flow_vis_as_img
        self.use_depth_vis_as_img = use_depth_vis_as_img
        # Load annotation file.
        # Format: 
        # {
        #     'type': 'video', 
        #     'ann': {
        #         'video name 1': [
        #             {'timestamp': timestamp, 'img_rel_path': img_rel_path, 'flow_rel_path': flow_rel_path, 'vel': vel},
        #             ...
        #         ]
        #         ...
        #     }
        # }
        ann_path = os.path.join(dataset_dir, ann_file_name)
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        assert self.ann['type'] == 'video'

        # Generate clip indices. Format: (video name, start frame index).
        self.clip_indices = []
        for video_name in self.ann['ann']:
            video = self.ann['ann'][video_name]
            if len(video) >= clip_len:
                for start_frame_index in range(len(video) - clip_len + 1):
                    if self.use_flow or self.use_flow_vis_as_img:
                        # Only use the clips with flow in the last frame.
                        end_frame_index = start_frame_index + clip_len - 1
                        if 'flow_rel_path' in self.ann['ann'][video_name][end_frame_index]:
                            self.clip_indices.append((video_name, start_frame_index))
                    elif self.use_depth_vis_as_img:
                        # FIXME: Only use the clips with flow in the last frame. Need to be fixed.
                        end_frame_index = start_frame_index + clip_len - 1
                        if 'depth_rel_path' in self.ann['ann'][video_name][end_frame_index]:
                            self.clip_indices.append((video_name, start_frame_index))
                    else:
                        self.clip_indices.append((video_name, start_frame_index))
        
        # Other settings.
        self.transform = transform
        
    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, index):
        # Get annotation.
        video_name, start_frame_index = self.clip_indices[index]

        imgs = []
        vels = []
        for frame_index in range(start_frame_index, start_frame_index + self.clip_len):
            frame_ann = self.ann['ann'][video_name][frame_index]

            # Load image.
            if self.use_flow_vis_as_img: # Use flow visualization as input image.
                img_rel_path = frame_ann['flow_rel_path'].replace('_flow_', '_flow_vis_').replace('.npy', '.jpg')
            elif self.use_depth_vis_as_img: # Use depth visualization as input image.
                img_rel_path = frame_ann['depth_rel_path'].replace('_depth_', '_depth_vis_').replace('.npy', '.jpg')
            else:
                img_rel_path = frame_ann['img_rel_path']
            
            if self.zip_file_name:
                img_path = os.path.join(self.dataset_dir, self.zip_file_name + '@' + img_rel_path)
            else:
                img_path = os.path.join(self.dataset_dir, img_rel_path)
            img_arr = zipped_pil_loader(img_path)
            img = self.transform(img_arr)
            imgs.append(img)

            # Load velocities and normalize.
            vel = normalize_v(np.array(frame_ann['vel'], dtype=np.float32))  # Note: the np.array() is required since normalize_v() is an in-place function.
            vels.append(vel)

        imgs = torch.stack(imgs, dim=1)            # Shape: [C,H,W] -> [C,D,H,W].
        vel = vels[-1]                             # Only use last frame's velocities. Shape: [4,].

        item = {'img': imgs, 'vel': vel}

        if self.use_flow:
            # Load flow.
            flow_rel_path = frame_ann['flow_rel_path']
            if self.zip_file_name:
                flow_path = os.path.join(self.dataset_dir, self.zip_file_name + '@' + flow_rel_path)
            else:
                flow_path = os.path.join(self.dataset_dir, flow_rel_path)
            flow = zipped_numpy_loader(flow_path).astype(np.float32)
            item['flow'] = flow.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W].

        return item