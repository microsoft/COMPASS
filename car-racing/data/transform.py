## 
import torch
from torchvision import transforms


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# Training transform: Only resize.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),             
    transforms.ToTensor(),
    normalize
])

# Validation transform: Only resize.
# Note: It is different from the ImageNet example, which resizes image to
#       256x256, and then center crops it to 224x224.
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),   
    normalize
])


class SimpleClipTransform(object):
    """ Simple clip transform. """
    def __call__(self, item):
        for data_type in item:
            data = item[data_type]
            if data_type in ['img']:
                item[data_type] = torch.stack(data, dim=1)  # Shape: [C, H, W] * D -> [C, D, H, W].
            elif data_type in ['steering']:
                item[data_type] = torch.stack(data, dim=0)  # Shape: [] * D -> [D,].
            else:
                raise ValueError()
        return item


class TwoFrameClipTransform(object):
    """ Clip transform to generate two-frame pairs. 
        Currently only support previous frames to last frame:
            [frame 1, ..., n-1, n] -> [[frame 1, n], ..., [n-1, n]]. 
        and it only applies to images. """
    def __call__(self, item):
        for data_type in item:
            data = item[data_type]
            if data_type in ['img']:
                item[data_type] = torch.stack(
                    [torch.stack([x, data[-1]], dim=1) for x in data[:-1]]
                )  # Shape: [C, H, W] * D -> [C, 2, H, W] * (D-1) -> [D-1, C, 2, H, W].
            elif data_type in ['steering']:
                item[data_type] = torch.stack(data, dim=0)  # Shape: [] * D -> [D,].
            else:
                raise ValueError()
        return item