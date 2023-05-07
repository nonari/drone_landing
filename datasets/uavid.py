import numpy as np
import torch
from torch.nn.functional import one_hot

from config import TrainConfig
from datasets.dataset import GenericDataset
from glob import glob
from os import path


def label_to_tensor(label, keys):
    v = torch.tensor([256 * 256, 256, 1])
    one_key = (torch.tensor(keys) * v).sum(dim=1)
    one_ch = (label * v[:, None, None]).sum(dim=0)

    # Moving car to static car
    one_ch[one_ch == one_key[6]] = one_key[5]

    sparse = (one_key[:, None, None] == one_ch).float()
    return sparse


# SLOW METHOD
def label_to_tensor_reg(label, keys):
    one_key = np.sum(np.left_shift(keys, [0, 8, 16]), axis=1)
    one_ch = np.sum(np.left_shift(label, [0, 8, 16]), axis=2).flatten()

    key_idx = np.argsort(one_key)
    _, inv_map = np.unique(np.concatenate([one_ch, one_key]), return_inverse=True)
    class_label = key_idx[inv_map[:-one_key.shape[0]].astype(int)].reshape(label.shape[:-1])

    return torch.movedim(one_hot(torch.tensor(class_label)), 2, 0)


color_keys = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [128, 64, 128],
    [0, 128, 0],
    [128, 128, 0],
    [64, 0, 128],
    [192, 0, 192],
    [64, 64, 0]
])

class_names = [
    "background",
    "building",
    "road",
    "tree",
    "low veg",
    "moving car",
    "static car",
    "human",
]


class UAVid(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        if not path.exists(config.uavid_root):
            raise Exception('Incorrect path for UAVid dataset')

        train_image_paths = glob(path.join(config.uavid_root, 'uavid_train', '*', 'Images', '*'))
        train_label_paths = glob(path.join(config.uavid_root, 'uavid_train', '*', 'Labels', '*'))
        val_image_paths = glob(path.join(config.uavid_root, 'uavid_val', '*', 'Images', '*'))
        val_label_paths = glob(path.join(config.uavid_root, 'uavid_val', '*', 'Labels', '*'))
        self._image_paths = train_image_paths + val_image_paths
        self._label_paths = train_label_paths + val_label_paths
        val_len = len(val_label_paths)
        train_len = len(train_label_paths)
        self._train_idx = [i for i in range(train_len)]
        self._val_idx = [i for i in range(train_len, val_len + train_len)]

        self._class_names = class_names
        self._color_keys = color_keys
        self._label_to_tensor = label_to_tensor

    def get_folds(self):
        if isinstance(self.config, TrainConfig):
            return [(self._train_idx, self._val_idx)]
        else:
            return [self._val_idx]
