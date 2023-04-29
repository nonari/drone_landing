import numpy as np
from os import path
from glob import glob

import torchvision.transforms
from PIL import Image
import importlib
from datasets.dataset import augment
from config import TrainConfig

from datasets.dataset import prepare_image, adapt_label, adapt_image, label_to_tensor, GenericDataset, \
    label_to_tensor_collapse, get_dataset_transform

color_keys = np.asarray([
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

ruralscapes_classnames = np.array([
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
])

test_ids = ['0051', '0093', '0047', '0056', '0086']
train_folds = [['0101', '0053', '0089', '0116', '0043'],
               ['0044', '0085', '0061', '0046', '0118'],
               ['0045', '0088', '0114', '0050', '0097']]

split_train_ids = ['0044', '0043', '0045', '0046', '0047', '0050',
                   '0053', '0085', '0093', '0097', '0101', '0114', '0118']

split_test_ids = ['0051', '0056', '0061', '0086', '0088', '0089', '0116']


def get_all(paths, ids):
    all_frames = list(filter(lambda p: p[-15:-11] in ids, paths))

    return all_frames


class RuralscapesDataset(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.color_keys = color_keys
        self.class_names = ruralscapes_classnames
        if not path.exists(config.rural_root):
            raise Exception('Incorrect path for Ruralscapes dataset')
        images_root = path.join(config.rural_root, 'frames')
        labels_root = path.join(config.rural_root, 'labels/resized_labels')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        self._image_paths = sorted(image_paths, key=lambda x: (int(path.basename(x).split('_')[1]),
                                                               int(path.basename(x).split('_')[2].split('.')[0])))
        self._label_paths = sorted(label_paths, key=lambda x: (int(path.basename(x).split('_')[2]),
                                                               int(path.basename(x).split('_')[3].split('.')[0])))

        self.inv_idx = {}
        self.index()

        net_config = config.net_config
        t_rural = adapt_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_rural)
        self._prepare_lab = prepare_image(adapt_label(net_config['input_size']))
        self._label_to_tensor = label_to_tensor
        self._no_classes = len(self.class_names)

    def index(self):
        self.inv_idx.clear()
        for idx, k in enumerate(self._image_paths):
            self.inv_idx[k] = idx

    def p_to_i(self, p):
        return [self.inv_idx[k] for k in p]

    def get_folds(self):
        folds = []
        if isinstance(self.config, TrainConfig):
            fold0_train = get_all(self._image_paths, train_folds[0] + train_folds[1])
            fold0_val = get_all(self._image_paths, train_folds[2])
            fold1_train = get_all(self._image_paths, train_folds[1] + train_folds[2])
            fold1_val = get_all(self._image_paths, train_folds[0])
            fold2_train = get_all(self._image_paths, train_folds[0] + train_folds[2])
            fold2_val = get_all(self._image_paths, train_folds[1])
            folds.append((self.p_to_i(fold0_train), self.p_to_i(fold0_val)))
            folds.append((self.p_to_i(fold1_train), self.p_to_i(fold1_val)))
            folds.append((self.p_to_i(fold2_train), self.p_to_i(fold2_val)))
        else:
            fold_test = get_all(self._image_paths, test_ids)
            folds.append((self.p_to_i(fold_test)))
            folds.append((self.p_to_i(fold_test)))
            folds.append((self.p_to_i(fold_test)))
        return folds

    def classes(self):
        return self._no_classes

    def classnames(self):
        return self.class_names

    def colors(self):
        return self.color_keys

    def pred_to_color_mask(self, true, pred):
        pred_mask = self.color_keys[pred]
        true_mask = self.color_keys[true]
        return true_mask, pred_mask

    def __getitem__(self, item):
        tensor_im = self._prepare_im(self._image_paths[item])
        pil_lab = self._prepare_lab(self._label_paths[item])
        if isinstance(self.config, TrainConfig) and self.config._training and self.config.augment:
            tensor_im, pil_lab = augment(tensor_im, pil_lab)
        tensor_lab = self._label_to_tensor(np.asarray(pil_lab), self.color_keys)
        return tensor_im, tensor_lab


class RuralscapesOrigSplit(RuralscapesDataset):
    def __init__(self, config):
        super().__init__(config)

    def get_folds(self):
        folds = []
        if isinstance(self.config, TrainConfig):
            fold0_train = get_all(self._image_paths, split_train_ids)
            fold0_val = get_all(self._image_paths, split_test_ids)
            folds.append((self.p_to_i(fold0_train), self.p_to_i(fold0_val)))
        else:
            fold_test = get_all(self._image_paths, split_test_ids)
            folds.append(self.p_to_i(fold_test))
        return folds


class RuralscapesOrigSegprop(RuralscapesOrigSplit):
    def __init__(self, *args):
        super().__init__(*args)
        labels_root = path.join(self.config.rural_root, 'labels/segprop_labels')
        images_root = path.join(self.config.rural_root, 'frames2')
        image_paths = glob(path.join(images_root, '*.jpg'))
        label_paths = glob(path.join(labels_root, '*.png'))
        image_paths = sorted(image_paths, key=lambda x: (int(path.basename(x).split('_')[1]),
                                                         int(path.basename(x).split('_')[2].split('.')[0])))
        label_paths = sorted(label_paths, key=lambda x: (int(path.basename(x).split('_')[2]),
                                                         int(path.basename(x).split('_')[3].split('.')[0])))
        self._image_paths += image_paths
        self._label_paths += label_paths
        self.index()


