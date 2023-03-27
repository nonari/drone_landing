from abc import ABC
import importlib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torch

from config import TrainConfig

imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                   'std': [1 / 0.229, 1 / 0.224, 1 / 0.225]}


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


class DummyDataset(GenericDataset):
    def __init__(self, config):
        self.config = config
        self.color_key = np.array([[0, 0, 0], [255, 255, 255]])
        self.batch_size = config.batch_size
        # self.input_size = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG['input_size']
        self.input_size = 64, 64

    def classes(self):
        return 2

    def classnames(self):
        return np.array(['class1', 'class2'])

    def colors(self):
        return self.color_key

    def pred_to_color_mask(self, true, pred):
        return self.color_key[true], self.color_key[pred]

    def get_folds(self):
        folds = []
        if isinstance(self.config, TrainConfig):
            folds.append(([0, 1], [2, 3]))
        else:
            folds.append([2, 3])
        return folds

    def __getitem__(self, index):
        image = torch.rand((3, self.input_size[0], self.input_size[1]))
        sparse_label = torch.zeros((self.classes(), self.input_size[0], self.input_size[1]))
        dense_label = np.random.randint(0, self.classes(), (self.input_size[0], self.input_size[1]))
        idxy = np.arange(self.input_size[0])[:, None].repeat(self.input_size[1], axis=1)
        idxx = np.arange(self.input_size[1])[None].repeat(self.input_size[0], axis=0)
        sparse_label[dense_label, idxy, idxx] = 1

        return image, sparse_label
