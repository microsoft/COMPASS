import io
import os
import json
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data

from .zipreader import is_zip_path, ZipReader


def zipped_pil_loader(path):
    """ PIL image loader that supports zipped images.
        Ref: https://github.com/SwinTransformer/Transformer-SSL/blob/main/data/cached_image_folder.py#L179
    """
    # Open path as file to avoid ResourceWarning.
    # Ref: https://github.com/python-pillow/Pillow/issues/835
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
            # Put the return in the with block to avoid error
            # 'ValueError: seek of closed file'.
            return img.convert('RGB')  


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


def check_data_types(frame_ann, data_types):
    """ Check if the given frame has the required data types. """
    ret = True
    for data_type in data_types:
        if data_type in ['img']:
            if f'{data_type}_rel_path' not in frame_ann:
                ret = False
        elif data_type in ['steering']:
            if f'{data_type}' not in frame_ann:
                ret = False
        else:
            raise ValueError(f'Unknown data type {data_type}.')
    return ret


class CarFrameDataset(data.Dataset):
    """ Car frame dataset class.
        The annotation file format: 
        {
            'type': 'video', 
            'ann': {
                'video name 1': [
                    {
                        'timestamp': timestamp, 
                        'img_rel_path': img_rel_path,
                        'steering': steering
                    },
                    ...
                ]
                ...
            }
        }
    Args:
        dataset_dir (string): Path to dataset directory.
        zip_file_name (string): Zipped dataset file name.
        ann_file_name (string): Annotation file name.
        transform (callable): A function/transform that takes in
            an image and returns a transformed version.
    """
    
    def __init__(self, dataset_dir, zip_file_name, ann_file_name, data_types=['img', 'steering'], transform=None):
        # Settings.
        self.dataset_dir = dataset_dir
        self.ann_file_name = ann_file_name
        self.zip_file_name = zip_file_name
        self.data_types = data_types
        self.transform = transform

        # Load annotation file.
        ann_path = os.path.join(self.dataset_dir, self.ann_file_name)
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)
        assert self.ann['type'] == 'video'

        # Generate frame index mappings.
        self.global_index_to_pair = {}  # Global frame index to paired frame index.
        self.pair_to_global_index = {}  # Paired frame index to global frame index.

        for video_name in self.ann['ann']:
            video = self.ann['ann'][video_name]
            for frame_index, frame_ann in enumerate(video):
                if check_data_types(frame_ann, self.data_types):
                    global_frame_index = len(self.global_index_to_pair)
                    self.global_index_to_pair[global_frame_index] = (video_name, frame_index)
                    self.pair_to_global_index[(video_name, frame_index)] = global_frame_index

    def __getitem__(self, index):
        """
        Args:
            index (int): Global frame index in the whole dataset.
        Returns:
            item: An item dictionary containing all the annotations for the frame.
                  Format: {'img': image tensor, 'steering': steering tensor}
        """
        # Get frame annotation.
        video_name, frame_index = self.global_index_to_pair[index]
        frame_ann = self.ann['ann'][video_name][frame_index]

        # Load data.
        item = {}
        if 'img' in self.data_types:
            # Load image.
            img_rel_path = frame_ann['img_rel_path']
            if self.zip_file_name:
                img_path = os.path.join(self.dataset_dir, self.zip_file_name + '@' + img_rel_path)
            else:
                img_path = os.path.join(self.dataset_dir, img_rel_path)
            img_arr = zipped_pil_loader(img_path)
            img = self.transform(img_arr)
            item['img'] = img  # Shape: [C, H, W].

        if 'steering' in self.data_types:
            # Load steering.
            steering = torch.tensor(frame_ann['steering'], dtype=torch.float32)
            item['steering'] = steering  # Shape: [].

        return item

    def __len__(self):
        return len(self.global_index_to_pair)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str


class SimpleClipDataset(data.Dataset):
    """ Simple clip dataset class for car.
    Args:
        frame_dataset (data.Dataset): Car frame dataset.
        clip_len (int): Number of frames in each sampled clip.
        max_clip_pairs (int): Maximum number of clip pairs.
        transform (callable): A function/transform that takes in
            a sampled clip and returns a transformed version.
    Attributes:
        clip_pairs (list): List of (video_name, start_frame_index) tuples
    """
    def __init__(self, frame_dataset, clip_len=2, max_clip_pairs=-1, transform=None):
        self.frame_dataset = frame_dataset
        self.clip_len = clip_len
        self.max_clip_pairs = max_clip_pairs
        self.transform = transform

        # Create clip paired indices (video name, start frame index).
        ann = self.frame_dataset.ann['ann']
        data_types = self.frame_dataset.data_types
        self.clip_pairs = []
        for video_name in ann:
            video_ann = ann[video_name]
            if len(video_ann) >= clip_len:
                for start_frame_index in range(len(video_ann) - clip_len + 1):
                    # Only use the clips that required data types are available in all frames.
                    end_frame_index = start_frame_index + clip_len - 1
                    if all(check_data_types(video_ann[i], data_types) for i in range(start_frame_index, end_frame_index + 1)):
                        self.clip_pairs.append((video_name, start_frame_index))
        
        # Limit the number of clip pairs.
        if self.max_clip_pairs >= 0:
            self.clip_pairs = self.clip_pairs[:self.max_clip_pairs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            item: An item dictionary containing all the annotations for the clip
                  Format: 
                    Without transform: {'img': [img_1, ...], 'steering': [steering_1, ...], ...}
                    With transform: {'img': img_tensor, 'steering': steering_tensor}
        """
        # Create clip item.
        video_name, start_frame_index = self.clip_pairs[index]
        clip_item = defaultdict(list)
        for frame_index in range(start_frame_index, start_frame_index + self.clip_len):
            global_index = self.frame_dataset.pair_to_global_index[(video_name, frame_index)]
            frame_item = self.frame_dataset[global_index]
            for data_type in self.frame_dataset.data_types:
                clip_item[data_type].append(frame_item[data_type])
        
        # Apply transformation.
        if self.transform is not None:
            clip_item = self.transform(clip_item)

        return clip_item

    def __len__(self):
        return len(self.clip_pairs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        return fmt_str


class IntervalClipDataset(data.Dataset):
    """ Clip w/ frame interval class for car.
    Args:
        frame_dataset (data.Dataset): Car frame dataset.
        clip_len (int): Number of frames in each sampled clip.
        frame_interval (int): Interval between consecutive frames.
        max_clip_pairs (int): Maximum number of clip pairs.
        transform (callable): A function/transform that takes in
            a sampled clip and returns a transformed version.
    Attributes:
        clip_pairs (list): List of (video_name, [frame index 1, ..., frame index n]) tuples
    """
    def __init__(self, frame_dataset, clip_len=2, frame_interval=1, max_clip_pairs=-1, transform=None):
        self.frame_dataset = frame_dataset
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.max_clip_pairs = max_clip_pairs
        self.transform = transform

        # Create clip paired indices (video name, [frame index 1, ..., frame index n]).
        ann = self.frame_dataset.ann['ann']
        data_types = self.frame_dataset.data_types
        self.clip_pairs = []
        frame_range_len = self.frame_interval * (self.clip_len - 1) + 1
        for video_name in ann:
            video_ann = ann[video_name]
            if len(video_ann) >= frame_range_len:
                for start_frame_index in range(len(video_ann) - frame_range_len + 1):
                    # Only use the clips that required data types are available in all frames.
                    end_frame_index = start_frame_index + frame_range_len - 1
                    frame_range = list(range(start_frame_index, end_frame_index + 1, self.frame_interval))
                    if all(check_data_types(video_ann[i], data_types) for i in frame_range):
                        self.clip_pairs.append((video_name, frame_range))
        
        # Limit the number of clip pairs.
        if self.max_clip_pairs >= 0:
            self.clip_pairs = self.clip_pairs[:self.max_clip_pairs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            item: An item dictionary containing all the annotations for the clip
                  Format: 
                    Without transform: {'img': [img_1, ...], 'steering': [steering_1, ...], ...}
                    With transform: {'img': img_tensor, 'steering': steering_tensor}
        """
        # Create clip item.
        video_name, frame_range = self.clip_pairs[index]
        clip_item = defaultdict(list)
        for frame_index in frame_range:
            global_index = self.frame_dataset.pair_to_global_index[(video_name, frame_index)]
            frame_item = self.frame_dataset[global_index]
            for data_type in self.frame_dataset.data_types:
                clip_item[data_type].append(frame_item[data_type])
        
        # Apply transformation.
        if self.transform is not None:
            clip_item = self.transform(clip_item)

        return clip_item

    def __len__(self):
        return len(self.clip_pairs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        return fmt_str


class MultiIntervalsToLastClipDataset(data.Dataset):
    """ Clip w/ multiple intervals to last frame class for car.
    Args:
        frame_dataset (data.Dataset): Car frame dataset.
        frame_intervals (list): Multiple intervals to the last frame.
            e.g. [1,] -> frame t-1, t
                 [8,4,2,1] -> frame t-8, t-4, t-2, t-1, t
        max_clip_pairs (int): Maximum number of clip pairs.
        transform (callable): A function/transform that takes in
            a sampled clip and returns a transformed version.
    Attributes:
        clip_pairs (list): List of (video_name, [frame index 1, ..., frame index n]) tuples
    """
    def __init__(self, frame_dataset, frame_intervals=[1], max_clip_pairs=-1, transform=None):
        self.frame_dataset = frame_dataset
        self.frame_intervals = frame_intervals
        self.max_clip_pairs = max_clip_pairs
        self.transform = transform

        # Create clip paired indices (video name, [frame index 1, ..., frame index n]).
        ann = self.frame_dataset.ann['ann']
        data_types = self.frame_dataset.data_types
        self.clip_pairs = []
        frame_range_len = max(frame_intervals) + 1
        for video_name in ann:
            video_ann = ann[video_name]
            if len(video_ann) >= frame_range_len:
                for start_frame_index in range(len(video_ann) - frame_range_len + 1):
                    # Only use the clips that required data types are available in all frames.
                    end_frame_index = start_frame_index + frame_range_len - 1
                    frame_range = list(end_frame_index - interval for interval in self.frame_intervals)
                    frame_range.append(end_frame_index)
                    if all(check_data_types(video_ann[i], data_types) for i in frame_range):
                        self.clip_pairs.append((video_name, frame_range))
        
        # Limit the number of clip pairs.
        if self.max_clip_pairs >= 0:
            self.clip_pairs = self.clip_pairs[:self.max_clip_pairs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            item: An item dictionary containing all the annotations for the clip
                  Format: 
                    Without transform: {'img': [img_1, ...], 'steering': [steering_1, ...], ...}
                    With transform: {'img': img_tensor, 'steering': steering_tensor}
        """
        # Create clip item.
        video_name, frame_range = self.clip_pairs[index]
        clip_item = defaultdict(list)
        for frame_index in frame_range:
            global_index = self.frame_dataset.pair_to_global_index[(video_name, frame_index)]
            frame_item = self.frame_dataset[global_index]
            for data_type in self.frame_dataset.data_types:
                clip_item[data_type].append(frame_item[data_type])
        
        # Apply transformation.
        if self.transform is not None:
            clip_item = self.transform(clip_item)

        return clip_item

    def __len__(self):
        return len(self.clip_pairs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of data points: {}\n'.format(self.__len__())
        return fmt_str