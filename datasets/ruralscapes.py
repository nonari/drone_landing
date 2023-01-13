import numpy as np
import torch
from os import path
from glob import glob
from PIL import Image
import importlib

from sklearn.model_selection import KFold

from datasets.dataset import transform_image, GenericDataset

ruralscapes_color_keys = np.asarray([
    [255, 255, 0],
    [0, 255, 0],
    [0, 127, 0],
    [0, 255, 255],
    [127, 127, 0],
    [255, 255, 255],
    [127, 127, 63],
    [255, 0, 255],
    [127, 127, 127],
    [255, 0, 0],
    [255, 127, 0],
    [0, 0, 255],
])


ruralscapes_classnames = [
    'building',
    'land',
    'forest',
    'sky',
    'fence',
    'road',
    'hill',
    'church',
    'car',
    'person',
    'haystack',
    'water'
]

test_ids = ['0051', '0093', '0047', '0056' '0086']
folds = [['0101', '0053', '0089', '0116', '0043'],
         ['0044', '0085', '0061', '0046', '0118'],
         ['0045', '0088', '0114', '0050', '0097']]


def get_all(dirname, ids):
    all_frames = []
    for i in ids:
        id_frames_exp = path.join(dirname, f'*{i:04}*')
        id_frames_paths = glob(id_frames_exp)
        all_frames += id_frames_paths

    return all_frames


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


class TUGrazDataset(GenericDataset):
    def __init__(self, options):
        self.options = options

        images_root = path.join(options.rural_root, 'frames')
        labels_root = path.join(options.rural_root, 'labels/resized_labels')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x)[:3]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))

        self._image_paths = np.asarray(image_paths)
        self._label_paths = np.asarray(label_paths)

        net_config = importlib.import_module(f'net_configurations.{options.model_config}').CONFIG
        t_rural = transform_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_rural)
        device = torch.device('cuda' if torch.cuda.is_available() and options.gpu else 'cpu')
        self._prepare_lab = prepare_image(label_transformation(
            ruralscapes_color_keys, net_config['input_size'], device))

    def get_folds(self):

        return folds_part

    def classes(self):
        return 12

    def classnames(self):
        return ruralscapes_classnames

    def colors(self):
        return ruralscapes_color_keys

    def pred_to_color_mask(self, true, pred):
        pred_mask = ruralscapes_color_keys[pred]
        true_mask = ruralscapes_color_keys[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        return self._prepare_im(self._image_paths[item]),  self._prepare_lab(self._label_paths[item])
