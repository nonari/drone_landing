import numpy as np
import torch
from os import path
from glob import glob

import torchvision.transforms
from PIL import Image
import importlib
import torchvision.transforms.functional as t_func

from datasets.dataset import transform_image, GenericDataset, augment

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


def label_to_tensor(label, keys):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(keys * v, axis=1)
    one_ch = np.sum(label * v[None, None], axis=2)
    # Church to building
    one_ch[one_ch == one_key[7]] = one_key[0]
    one_ch[one_ch == one_key[6]] = one_key[2]

    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse).movedim(2, 0)


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
        return lab_res

    return f


def randomHueSaturationValue(image, hue_shift_limit=(-0.5, 0.5),
                             sat_shift_limit=(0, 1),
                             val_shift_limit=(0, 1), u=0.5):
    if np.random.random() < u:
        trans = torchvision.transforms.ColorJitter(
            brightness=val_shift_limit,
            saturation=sat_shift_limit,
            hue=hue_shift_limit)
        image = trans(image)

    return image


def randomShiftScaleRotate(image, mask, u=0.5):
    if np.random.random() < u:
        size = image.shape[-2:]
        i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(image, [0.95, 0.95], [1.0, 1.0])
        image = t_func.resized_crop(image, i, j, h, w, size, interpolation=Image.BILINEAR)
        mask = t_func.resized_crop(mask, i, j, h, w, size, interpolation=Image.NEAREST)

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flip(image, 1)
        mask = np.flip(mask, 1)

    return image, mask


def augment_rural(img, label):
    img = randomHueSaturationValue(
        img,
        hue_shift_limit=(0, .2),
        sat_shift_limit=(0, .02),
        val_shift_limit=(-0.06, .06)
    )

    img, label = randomShiftScaleRotate(img, label)

    img, label = randomHorizontalFlip(img, label)

    return img, label


class RuralscapesDataset(GenericDataset):
    def __init__(self, config):
        self.config = config
        images_root = path.join(config.rural_root, 'frames')
        labels_root = path.join(config.rural_root, 'labels/resized_labels')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        self.image_paths = sorted(image_paths, key=lambda x: (int(path.basename(x).split('_')[1]),
                                                              int(path.basename(x).split('_')[2].split('.')[0])))
        self.label_paths = sorted(label_paths, key=lambda x: (int(path.basename(x).split('_')[2]),
                                                              int(path.basename(x).split('_')[3].split('.')[0])))

        self.inv_idx = {}

        self.index()

        net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
        t_rural = transform_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_rural)
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self._prepare_lab = prepare_image(label_transformation(
            ruralscapes_color_keys, net_config['input_size'], device))

    def index(self):
        for idx, k in enumerate(self._image_paths):
            self.inv_idx[k] = idx

    def p_to_i(self, p):
        return [self.inv_idx[k] for k in p]

    def get_folds(self):
        folds = []
        if self.config.train:
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
        return ruralscapes_color_keys

    def pred_to_color_mask(self, true, pred):
        pred_mask = ruralscapes_color_keys[pred]
        true_mask = ruralscapes_color_keys[true]
        return true_mask, pred_mask

    def __len__(self):
        return self._image_paths.__len__()

    def __getitem__(self, item):
        # TODO move augment to Dataloader
        if self.config.train and self.config._training:
            tensor_im = self._prepare_im(self._image_paths[item])
            pil_lab = self._prepare_lab(self._label_paths[item])
            aug_im, aug_lab = augment(tensor_im, pil_lab)
            t_aug_lab = label_to_tensor(np.asarray(aug_lab), ruralscapes_color_keys)
            return aug_im, t_aug_lab
        else:
            tensor_im = self._prepare_im(self._image_paths[item])
            pil_lab = self._prepare_lab(self._label_paths[item])
            t_lab = label_to_tensor(np.asarray(pil_lab), ruralscapes_color_keys)
            return tensor_im, t_lab


class RuralscapesOrigSplit(RuralscapesDataset):
    def __init__(self, *args):
        super().__init__(*args)

    def get_folds(self):
        folds = []
        if self.config.train:
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

