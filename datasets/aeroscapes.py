import numpy as np
from torch.utils.data import Dataset
import torch
from os import path
from glob import glob
from PIL import Image
import importlib

from datasets.dataset import transform_image

aeroscapes_color = np.asarray([
    [0, 0, 0],
    [255, 22, 96],
    [119, 11, 32],
    [9, 143, 150],
    [102, 51, 0],
    [2, 135, 115],
    [70, 70, 70],
    [51, 51, 0],
    [128, 64, 128]
])


aeroscapes_classnames = [
    'back',
    'person',
    'bike',
    'car',
    'animal',
    'obstacle',
    'building',
    'vegetation',
    'road',
]


def aero_label_to_tensor(label):
    # Compatibility issue? READ ONLY label
    label = label.copy()
    label[label == 4] = 0
    label[label == 5] = 0
    label[label == 11] = 0
    label[label == 6] = 4
    label[label == 7] = 5
    label[label == 8] = 6
    label[label == 9] = 7
    label[label == 10] = 8

    r, c = label.shape
    sparse_mask = np.zeros((r, c, 9), np.int32)
    y, x = np.unravel_index(np.arange(r*c), (r, c))
    sparse_mask[y, x, label.flatten().astype(np.int32)] = 1
    sparse_mask = sparse_mask.astype(np.float32)
    return torch.tensor(sparse_mask).movedim(2, 0)


def prepare_image(transformation):
    def f(im_path):
        im_orig = Image.open(im_path)
        im_tensor = transformation(im_orig)
        im_orig.close()
        return im_tensor
    return f


def label_transformation(new_size):
    def f(label):
        lab_res = label.resize(new_size[::-1], Image.NEAREST)
        return aero_label_to_tensor(np.asarray(lab_res))
    return f


class AeroscapesDataset(Dataset):
    def __init__(self, options):
        images_root = path.join(options.aeroscapes_root, 'JPEGImages')
        labels_root = path.join(options.aeroscapes_root, 'SegmentationClass')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: path.basename(x)[:-4])
        label_paths = sorted(label_paths, key=lambda x: path.basename(x)[:-4])

        self._image_paths = np.asarray(image_paths)
        self._label_paths = np.asarray(label_paths)

        net_config = importlib.import_module(f'net_configurations.{options.model_config}').CONFIG
        t_tugraz = transform_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_tugraz)
        self._prepare_lab = prepare_image(label_transformation(net_config['input_size']))

    def classes(self):
        return 9

    def classnames(self):
        return aeroscapes_classnames

    def colors(self):
        return aeroscapes_color

    def pred_to_color_mask(self, true, pred):
        pred[pred == 2] = 0
        pred[pred == 4] = 0
        pred[pred == 5] = 0
        pred[pred == 6] = 0
        pred[pred == 7] = 0
        pred[pred == 8] = 0
        pred[pred == 19] = 0
        pred[pred == 23] = 0
        pred[pred == 15] = 1
        pred[pred == 18] = 2
        pred[pred == 17] = 3
        pred[pred == 16] = 6
        pred[pred == 22] = 5
        pred[pred == 13] = 5
        pred[pred == 14] = 5
        pred[pred == 9] = 6
        pred[pred == 10] = 6
        pred[pred == 11] = 6
        pred[pred == 12] = 6
        pred[pred == 3] = 7
        pred[pred == 9] = 7
        pred[pred == 20] = 7
        pred[pred == 21] = 7
        pred[pred == 1] = 8
        pred_mask = aeroscapes_color[pred]
        true_mask = aeroscapes_color[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        return self._prepare_im(self._image_paths[item]),  self._prepare_lab(self._label_paths[item])


