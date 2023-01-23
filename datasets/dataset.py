from abc import ABC

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision.transforms.functional as t_func
import torch

imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                   'std': [1/0.229, 1/0.224, 1/0.225]}


def transform_image(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm)
    ])


def augment(im, mask):
    size = im.shape[-2:]
    i, j, h, w = transforms.RandomResizedCrop.get_params(im, [0.95, 0.95], [1.0, 1.0])
    im = t_func.resized_crop(im, i, j, h, w, size, interpolation=Image.BILINEAR)
    mask = t_func.resized_crop(mask, i, j, h, w, size, interpolation=Image.NEAREST)

    ang = transforms.RandomRotation.get_params([-5, 5])
    im = t_func.rotate(im, ang, interpolation=Image.BILINEAR)
    mask = t_func.rotate(mask, ang, interpolation=Image.NEAREST, fill=[7, 11, 13])

    if torch.rand(1) < 0.5:
        im = t_func.hflip(im)
        mask = t_func.hflip(mask)

    return im, mask


class GenericDataset(Dataset, ABC):
    def classes(self):
        raise NotImplementedError

    def classnames(self):
        raise NotImplementedError

    def colors(self):
        raise NotImplementedError

    def pred_to_color_mask(self, true, pred):
        raise NotImplementedError

    def get_folds(self):
        raise NotImplementedError
