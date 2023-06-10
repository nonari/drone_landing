import numpy as np
import torch
from os import path
from glob import glob

from datasets.dataset import GenericDataset


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


def label_to_tensor(label, keys):
    v = torch.tensor([256 * 256, 256, 1])
    one_key = (torch.tensor(keys) * v).sum(dim=1)
    one_ch = (label * v[:, None, None]).sum(dim=0)

    sparse = (one_key[:, None, None] == one_ch).float()
    return sparse


class TUGraz(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self._color_keys = tugraz_color_keys
        self._class_names = tugraz_classnames
        if not path.exists(config.tugraz_root):
            raise Exception('Incorrect path for TUGraz dataset')

        image_paths = glob(path.join(config.tugraz_root, 'training_set/low_res_images/*.jpg'))
        label_paths = glob(path.join(config.tugraz_root, 'training_set/gt/semantic/low_res_label_images/*.png'))

        self._image_paths = sorted(image_paths, key=lambda x: int(path.basename(x).split('.')[0]))
        self._label_paths = sorted(label_paths, key=lambda x: int(path.basename(x).split('.')[0]))

        if config.data_factor > 1:
            self._image_paths = list(
                map(lambda x: x[1], filter(lambda x: x[0] % config.data_factor == 0, enumerate(self._image_paths))))
            self._label_paths = list(
                map(lambda x: x[1], filter(lambda x: x[0] % config.data_factor == 0, enumerate(self._label_paths))))

        self._label_to_tensor = label_to_tensor

    def get_folds(self):
        return [list(range(len(self._image_paths)))]
