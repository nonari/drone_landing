import numpy as np
import torch
from torch.nn.functional import one_hot

from config import TrainConfig
from datasets.dataset import GenericDataset, prepare_image, adapt_label, adapt_image, augment
from glob import glob
from os import path


def label_to_tensor(label, keys):
    one_key = np.sum(np.left_shift(keys, [0, 8, 16]), axis=1)
    one_ch = np.sum(np.left_shift(label, [0, 8, 16]), axis=2)

    # Moving car to static car
    one_ch[one_ch == one_key[6]] = one_key[5]

    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse).movedim(2, 0)


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

        net_config = config.net_config
        t_rural = adapt_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_rural)
        self._prepare_lab = prepare_image(adapt_label(net_config['input_size']))
        self._class_names = class_names
        self._label_to_tensor = label_to_tensor
        self._color_keys = color_keys

    def classes(self):
        return len(self._class_names)

    def classnames(self):
        return self._class_names

    def colors(self):
        return self._color_keys

    def pred_to_color_mask(self, true, pred):
        return self._color_keys[true], self._color_keys[pred]

    def get_folds(self):
        if isinstance(self.config, TrainConfig):
            return [(self._train_idx, self._val_idx)]
        else:
            return [self._val_idx]

    def __getitem__(self, index):
        tensor_im = self._prepare_im(self._image_paths[index])
        pil_lab = self._prepare_lab(self._label_paths[index])
        if isinstance(self.config, TrainConfig) and self.config._training and self.config.augment:
            tensor_im, pil_lab = augment(tensor_im, pil_lab)
        tensor_lab = self._label_to_tensor(np.asarray(pil_lab), self._color_keys)
        return tensor_im, tensor_lab
