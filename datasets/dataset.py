from abc import ABC
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch

imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                   'std': [1/0.229, 1/0.224, 1/0.225]}


def label_to_tensor(label, keys):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(keys * v, axis=1)
    one_ch = np.sum(label * v[None, None], axis=2)
    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse).movedim(2, 0)


def prepare_image(transformation):
    def f(im_path):
        im_orig = Image.open(im_path)
        im_tensor = transformation(im_orig)
        im_orig.close()
        return im_tensor

    return f


def adapt_label(new_size):
    def f(label):
        lab_res = label.resize(new_size[::-1], Image.NEAREST)
        return lab_res

    return f


def adapt_image(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm)
    ])


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
