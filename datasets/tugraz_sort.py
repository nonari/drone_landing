import numpy as np
import torch
from os import path
from glob import glob
from PIL import Image
import importlib

from datasets.data_refactor import tugraz_new_labels
from datasets.dataset import adapt_image, GenericDataset


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

old_keys = []
for v in tugraz_new_labels.values():
    old_keys += v
new_tugraz_keys = np.delete(tugraz_color_keys, old_keys, axis=0)

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
new_tugraz_classnames = np.delete(np.asarray(tugraz_classnames), old_keys, axis=0)

train_folds = [['12', '6', '11', '7', '10'],
               ['8', '4', '2', '0'],
               ['1', '5', '9', '3']]


def get_all(dirname, ids):
    all_frames = []
    for i in ids:
        id_frames_exp = path.join(dirname, f'{i}_*')
        id_frames_paths = glob(id_frames_exp)
        all_frames += id_frames_paths

    return all_frames


def label_to_tensor_v2(label, keys):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(keys * v, axis=1)
    one_ch = np.sum(label * v[None, None], axis=2)
    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse).movedim(2, 0)


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


class TUGrazSortedDataset(GenericDataset):
    def __init__(self, config):
        self.config = config
        if not path.exists(config.tugraz_root):
            raise Exception('Incorrect path for Tugraz sorted dataset')
        subset = 'training_set'
        images_root = path.join(config.tugraz_root, subset, 'classif')
        labels_root = path.join(config.tugraz_root, subset, f'gt/semantic/label_collapsed')
        self.images_root = images_root

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')

        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x).split('_')[1].split('.')[0]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))

        self._image_paths = np.asarray(image_paths)
        self._label_paths = np.asarray(label_paths)

        self.inv_idx = {}
        for idx, k in enumerate(self._image_paths):
            self.inv_idx[k] = idx

        net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
        t_tugraz = adapt_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_tugraz)
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self._prepare_lab = prepare_image(label_transformation(
            new_tugraz_keys, net_config['input_size'], device))

    def p_to_i(self, p):
        return [self.inv_idx[k] for k in p]

    def get_folds(self):
        folds = []
        fold0_val = get_all(self.images_root, train_folds[2])
        fold1_val = get_all(self.images_root, train_folds[0])
        fold2_val = get_all(self.images_root, train_folds[1])
        if self.config.train:
            fold0_train = get_all(self.images_root, train_folds[0] + train_folds[1])
            fold1_train = get_all(self.images_root, train_folds[1] + train_folds[2])
            fold2_train = get_all(self.images_root, train_folds[0] + train_folds[2])
            folds.append((self.p_to_i(fold0_train), self.p_to_i(fold0_val)))
            folds.append((self.p_to_i(fold1_train), self.p_to_i(fold1_val)))
            folds.append((self.p_to_i(fold2_train), self.p_to_i(fold2_val)))
        else:
            folds = [self.p_to_i(fold0_val), self.p_to_i(fold1_val), self.p_to_i(fold2_val)]
        return folds

    def classes(self):
        return 14

    def classnames(self):
        return new_tugraz_classnames

    def colors(self):
        return new_tugraz_keys

    def pred_to_color_mask(self, true, pred):
        pred_mask = new_tugraz_keys[pred]
        true_mask = new_tugraz_keys[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        return self._prepare_im(self._image_paths[item]),  self._prepare_lab(self._label_paths[item])
