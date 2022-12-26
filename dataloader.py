import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import Config
from torchvision import transforms
import torch
from os import path
from glob import glob
from PIL import Image
import importlib


imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                   'std': [1/0.229, 1/0.224, 1/0.225]}


def transform_tugraz(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm)
    ])


class_names = [
    'unlabeled',
]

tugraz_color_keys = np.asarray([
    [0, 0, 0],
    [128, 64, 128],
    [130, 76, 0],
    [0, 102, 0],
    [112, 103, 87],
    [28, 42, 168],
    [48, 41, 30],
    [0, 50, 89],
    [107, 142, 35],
    [70, 70, 70],
    [102, 102, 156],
    [254, 228, 12],
    [254, 148, 12],
    [190, 153, 153],
    [153, 153, 153],
    [255, 22, 96],
    [102, 51, 0],
    [9, 143, 150],
    [119, 11, 32],
    [51, 51, 0],
    [190, 250, 190],
    [112, 150, 146],
    [2, 135, 115],
    [255, 0, 0],
])


tugraz_classnames = [
    'nolabel',
    'paved',
    'dirt',
    'grass',
    'gravel',
    'water',
    'rocks',
    'pool',
    'veget',
    'roof',
    'wall',
    'window',
    'door',
    'fence',
    'pole',
    'person',
    'dog',
    'car',
    'bicycle',
    'tree',
    'bald',
    'marker',
    'obstacle',
    'conflict'
]



def label_to_tensor_v2(label, keys):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(keys * v, axis=1)
    one_ch = np.sum(label * v[None, None], axis=2)
    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse).movedim(2, 0)


def label_to_tensor_v3(label, keys, device='cuda'):
    v = torch.tensor([256 * 256, 256, 1]).unsqueeze(dim=0).to(device)
    keys = torch.tensor(keys).to(device)
    one_key = (keys * v).sum(dim=1).reshape(1, 1, -1)
    label = torch.tensor(label).to(device)
    one_ch = (label * v.unsqueeze(dim=1)).sum(dim=2, keepdims=True)
    sparse = (one_key == one_ch).float()
    return sparse.movedim(2, 0)


def label_to_tensor(label, color_mask):
    sparse_mask = np.all(np.equal(label, color_mask), axis=3).astype(np.float32)
    return torch.tensor(sparse_mask)


def prepare_image(transformation):
    def f(im_path):
        im_orig = Image.open(im_path)
        im_tensor = transformation(im_orig)
        im_orig.close()
        return im_tensor
    return f


def label_transformation(color_keys, new_size, device):
    def f(label):
        lab_res = label.resize(new_size[::-1], Image.NEAREST)
        return label_to_tensor_v2(np.asarray(lab_res), color_keys)
    return f


class TUGrazDataset(Dataset):
    def __init__(self, options):
        subset = 'training_set' if options.train else 'testing_set'
        images_root = path.join(options.tugraz_root, subset, options.tugraz_images_loc)
        labels_root = path.join(options.tugraz_root, subset, f'gt/semantic/{options.tugraz_labels_loc}')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x)[:3]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))

        self._image_paths = np.asarray(image_paths)
        self._label_paths = np.asarray(label_paths)

        net_config = importlib.import_module(f'net_configurations.{options.model_config}').CONFIG
        t_tugraz = transform_tugraz(net_config['input_size'])

        self._prepare_im = prepare_image(t_tugraz)
        device = torch.device('cuda' if torch.cuda.is_available() and options.gpu else 'cpu')
        self._prepare_lab = prepare_image(label_transformation(
            tugraz_color_keys, net_config['input_size'], device))

    def classes(self):
        return 24

    def classnames(self):
        return tugraz_classnames

    def pred_to_color_mask(self, true, pred):
        pred_mask = tugraz_color_keys[pred]
        true_mask = tugraz_color_keys[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        return self._prepare_im(self._image_paths[item]),  self._prepare_lab(self._label_paths[item])


if __name__ == '__main__':
    conf = Config()
    tg = TUGrazDataset(conf)
    print(tg.__len__())
