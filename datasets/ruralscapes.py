import numpy as np
import torch
from os import path
from glob import glob
from PIL import Image
import importlib

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


def get_all(dirname, ids):
    all_frames = []
    for i in ids:
        id_frames_exp = path.join(dirname, f'*_{i}_*')
        id_frames_paths = glob(id_frames_exp)
        all_frames += id_frames_paths

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


class RuralscapesDataset(GenericDataset):
    def __init__(self, config):
        self.config = config
        self.images_root = path.join(config.rural_root, 'frames')
        labels_root = path.join(config.rural_root, 'labels/resized_labels')

        image_paths = glob(self.images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: (int(path.basename(x).split('_')[1]),
                                                         int(path.basename(x).split('_')[2].split('.')[0])))
        label_paths = sorted(label_paths, key=lambda x: (int(path.basename(x).split('_')[2]),
                                                         int(path.basename(x).split('_')[3].split('.')[0])))

        self.inv_idx = {}
        for idx, k in enumerate(image_paths):
            self.inv_idx[k] = idx

        self._image_paths = np.asarray(image_paths)
        self._label_paths = np.asarray(label_paths)

        net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
        t_rural = transform_image(net_config['input_size'])

        self._prepare_im = prepare_image(t_rural)
        device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
        self._prepare_lab = prepare_image(label_transformation(
            ruralscapes_color_keys, net_config['input_size'], device))

    def p_to_i(self, p):
        return [self.inv_idx[k] for k in p]

    def get_folds(self):
        folds = []
        if self.config.train:
            fold0_train = get_all(self.images_root, train_folds[0] + train_folds[1])
            fold0_val = get_all(self.images_root, train_folds[2])
            fold1_train = get_all(self.images_root, train_folds[1] + train_folds[2])
            fold1_val = get_all(self.images_root, train_folds[0])
            fold2_train = get_all(self.images_root, train_folds[0] + train_folds[2])
            fold2_val = get_all(self.images_root, train_folds[1])
            folds.append((self.p_to_i(fold0_train), self.p_to_i(fold0_val)))
            folds.append((self.p_to_i(fold1_train), self.p_to_i(fold1_val)))
            folds.append((self.p_to_i(fold2_train), self.p_to_i(fold2_val)))
        else:
            fold_test = get_all(self.images_root, test_ids)
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
