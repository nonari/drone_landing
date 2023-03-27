import numpy as np
from os import path
from glob import glob

import torchvision.transforms
from PIL import Image
import importlib
import torchvision.transforms.functional as t_func
from config import TrainConfig

from datasets.dataset import prepare_image, adapt_label, adapt_image, label_to_tensor, GenericDataset

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


def jitter_hsv(image, hue_shift_limit=0.5,
               sat_shift_limit=1.,
               val_shift_limit=1., u=0.5):
    if np.random.random() < u:
        trans = torchvision.transforms.ColorJitter(
            brightness=val_shift_limit,
            saturation=sat_shift_limit,
            hue=hue_shift_limit)
        image = trans(image)

    return image


def rand_shift_scale_rotate(image, mask, u=0.5):
    if np.random.random() < u:
        size = image.shape[-2:]
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(image, [0.95, 0.95], [1.0, 1.0])
        image = t_func.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
        mask = t_func.resized_crop(mask, i, j, h, w, size, interpolation=Image.NEAREST)

    return image, mask


def random_hflip(image, mask, u=0.5):
    if np.random.random() < u:
        image = t_func.hflip(image)
        mask = t_func.hflip(mask)

    return image, mask


def augment_rural(img, label):
    img = jitter_hsv(
        img,
        hue_shift_limit=.2,
        sat_shift_limit=.02,
        val_shift_limit=.06,
    )

    img, label = rand_shift_scale_rotate(img, label)

    img, label = random_hflip(img, label)

    return img, label


class RuralscapesDataset(GenericDataset):
    def __init__(self, config):
        self.config = config
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
        return 12

    def classnames(self):
        return ruralscapes_classnames

    def colors(self):
        return color_keys

    def pred_to_color_mask(self, true, pred):
        pred_mask = color_keys[pred]
        true_mask = color_keys[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        tensor_im = self._prepare_im(self._image_paths[item])
        pil_lab = self._prepare_lab(self._label_paths[item])
        if isinstance(self.config, TrainConfig) and self.config._training:
            tensor_im, pil_lab = augment_rural(tensor_im, pil_lab)
        tensor_lab = label_to_tensor(np.asarray(pil_lab), color_keys)
        return tensor_im, tensor_lab


class RuralscapesOrigSplit(RuralscapesDataset):
    def __init__(self, *args):
        super().__init__(*args)

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
