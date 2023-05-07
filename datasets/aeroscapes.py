import numpy as np
from torch.nn import functional
import torch
from os import path
from glob import glob
from config import TrainConfig

from datasets.dataset import GenericDataset

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


def label_to_tensor(label, _):
    label[label == 4] = 0
    label[label == 5] = 0
    label[label == 11] = 0
    label[label == 6] = 4
    label[label == 7] = 5
    label[label == 8] = 6
    label[label == 9] = 7
    label[label == 10] = 8
    label[label == -1] = 0
    sparse = functional.one_hot(label.squeeze(dim=0).long(), num_classes=9)
    return sparse.movedim(2, 0).float()


def load_txt(file):
    names = []
    with open(file, 'r') as f:
        for line in f:
            names.append(line[:-1])
    return names


def get_all(paths, ids):
    all_frames = list(filter(lambda p: p[-14:-4] in ids, paths))

    return all_frames


class Aeroscapes(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        if not path.exists(config.aeroscapes_root):
            raise Exception('Incorrect path for Aeroscapes dataset')

        self._color_keys = aeroscapes_color
        self._class_names = aeroscapes_classnames

        images_root = path.join(config.aeroscapes_root, 'JPEGImages')
        labels_root = path.join(config.aeroscapes_root, 'SegmentationClass')
        train_set_path = path.join(config.aeroscapes_root, 'ImageSets', 'trn.txt')
        val_set_path = path.join(config.aeroscapes_root, 'ImageSets', 'val.txt')

        self._train_ids = load_txt(train_set_path)
        self._val_ids = load_txt(val_set_path)

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        self._image_paths = sorted(image_paths, key=lambda x: path.basename(x)[:-4])
        self._label_paths = sorted(label_paths, key=lambda x: path.basename(x)[:-4])

        self._label_paths = list(map(lambda x: x[1], filter(lambda x: x[0] % 2 == 0, enumerate(self._label_paths))))
        self._image_paths = list(map(lambda x: x[1], filter(lambda x: x[0] % 2 == 0, enumerate(self._image_paths))))

        self.inv_idx = {}
        self.index()

        self._label_to_tensor = label_to_tensor

    def index(self):
        self.inv_idx.clear()
        for idx, k in enumerate(self._image_paths):
            self.inv_idx[k] = idx

    def p_to_i(self, p):
        return [self.inv_idx[k] for k in p]

    def get_folds(self):
        train_imgs = get_all(self._image_paths, self._train_ids)
        val_imgs = get_all(self._image_paths, self._val_ids)
        if isinstance(self.config, TrainConfig):
            return [(self.p_to_i(train_imgs), self.p_to_i(val_imgs))]
        else:
            return [self.p_to_i(val_imgs)]
