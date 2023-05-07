from abc import ABC
import importlib
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torchvision.transforms import transforms, InterpolationMode
import torch
import torchvision.transforms.functional as t_func
import torchvision
from torchvision.transforms import InterpolationMode as Interpolation

from config import TrainConfig

imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                   'std': [1 / 0.229, 1 / 0.224, 1 / 0.225]}


def jitter_hsv(image, hue_shift_limit=0.5,
               sat_shift_limit=1.,
               val_shift_limit=1., u=0.5):
    trans = torchvision.transforms.ColorJitter(
        brightness=val_shift_limit,
        saturation=sat_shift_limit,
        hue=hue_shift_limit)
    image = trans(image)

    return image


def rand_shift_scale_rotate(image, mask):
    size = image.shape[-2:]
    ratio = image.shape[2] / image.shape[1]
    i, j, h, w = torchvision.transforms.RandomResizedCrop.get_params(image, [0.9, 1.], [ratio-0.1, ratio+0.1])
    image = t_func.resized_crop(image, i, j, h, w, size, interpolation=Interpolation.BILINEAR)
    mask = t_func.resized_crop(mask, i, j, h, w, size, interpolation=Interpolation.NEAREST)

    angle = ((torch.rand(1) - 0.5) * 10).item()
    image = t_func.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=[0])
    mask = t_func.rotate(mask, angle, interpolation=InterpolationMode.BILINEAR, fill=[-1])

    return image, mask


def random_hflip(image, mask, u=0.5):
    if np.random.random() < u:
        image = t_func.hflip(image)
        mask = t_func.hflip(mask)

    return image, mask


def augment(img, label):
    # img = jitter_hsv(
    #     img,
    #     hue_shift_limit=.9,
    #     sat_shift_limit=.0,
    #     val_shift_limit=.9,
    # )

    img, label = rand_shift_scale_rotate(img, label)
    img, label = random_hflip(img, label)

    return img, label


def label_to_tensor(label, keys):
    v = torch.tensor([256 * 256, 256, 1])
    one_key = (torch.tensor(keys) * v).sum(dim=1)
    one_ch = (label * v[:, None, None]).sum(dim=0)

    # Church to building
    one_ch[one_ch == one_key[7]] = one_key[0]

    sparse = (one_key[:, None, None] == one_ch).float()
    return sparse


def label_to_tensor_collapse(orig_color_keys, collapse_colors):
    def f(label, dest_color_keys):
        v = torch.tensor([256 * 256, 256, 1])
        orig_one_key = (torch.tensor(orig_color_keys) * v).sum(dim=1)
        dest_one_key = (torch.tensor(dest_color_keys) * v).sum(dim=1)

        one_ch = (label * v[:, None, None]).sum(dim=0)

        for key, value in collapse_colors.items():
            for i in value:
                one_ch[one_ch == orig_one_key[i]] = orig_one_key[key]

        sparse = (dest_one_key[:, None, None] == one_ch).float()
        return sparse
    return f


def sc_to_tensor_collapse(collapse_colors):
    def f(label, dest_class_keys):
        for key, value in collapse_colors.items():
            for i in value:
                label[label == i] = key

        sparse = (label == torch.tensor(dest_class_keys)[:, None, None]).float()
        return sparse
    return f


def prepare_image(transformation):
    def f(im_path):
        im_orig = Image.open(im_path)
        im_tensor = transformation(im_orig)
        im_orig.close()
        return im_tensor

    return f


def adapt_label(new_size):
    def f(label):
        lab_res = label.resize(new_size[::-1], Image.NEAREST)
        tensor_lab = (torchvision.transforms.ToTensor()(lab_res) * 255).int()
        return tensor_lab

    return f


def adapt_image(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm)
    ])


def class_position(classname, class_key):
    for idx, c in enumerate(class_key):
        if c == classname:
            return idx
    raise Exception(f'{classname} not found')


def find_duplicates(classname, classnames):
    positions = []
    for idx, c in enumerate(classnames):
        if c == classname:
            positions.append(idx)

    return positions


def get_dataset_transform(key_dest_classnames, class_assoc):
    dest_classnames = [e[1] for e in class_assoc]

    transform_color_key = [len(class_assoc)] * len(key_dest_classnames)
    color_collapse = {}
    seen_dest_classnames = set()
    for idx, c in enumerate(dest_classnames):
        if c in seen_dest_classnames or c is None:
            continue
        else:
            seen_dest_classnames.add(c)
        pos = class_position(c, key_dest_classnames)
        duplicates = find_duplicates(c, dest_classnames)
        if len(duplicates) > 1:
            color_collapse[duplicates[0]] = duplicates[1:]
        transform_color_key[pos] = idx

    return transform_color_key, color_collapse


class GenericDataset(Dataset, ABC):
    def __init__(self, config):
        self.config = config
        self._prepare_image = prepare_image(adapt_image(config.net_config['input_size']))
        self._prepare_label = prepare_image(adapt_label(config.net_config['input_size']))
        self._augment = augment

    def classes(self):
        return len(self._class_names)

    def classnames(self):
        return self._class_names

    def colors(self):
        return self._color_keys

    def pred_to_color_mask(self, true, pred):
        return self._color_keys[true], self._color_keys[pred]

    def get_folds(self):
        raise NotImplementedError

    def __getitem__(self, item):
        tensor_im = self._prepare_image(self._image_paths[item])
        rgb_lab = self._prepare_label(self._label_paths[item])
        if isinstance(self.config, TrainConfig) and self.config._training and self.config.augment:
            tensor_im, rgb_lab = self._augment(tensor_im, rgb_lab)
        tensor_lab = self._label_to_tensor(rgb_lab, self._color_keys)
        return tensor_im, tensor_lab


class DummyDataset(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.color_key = np.array([[0, 0, 0], [255, 0, 0], [255, 255, 255]])
        # self.input_size = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG['input_size']
        self.input_size = 64, 64

    def classes(self):
        return 3

    def classnames(self):
        return np.array(['class1', 'class2', 'class3'])

    def colors(self):
        return self.color_key

    def pred_to_color_mask(self, true, pred):
        return self.color_key[true], self.color_key[pred]

    def get_folds(self):
        folds = []
        if isinstance(self.config, TrainConfig):
            folds.append(([0, 1], [2, 3]))
        else:
            folds.append([2, 3])
        return folds

    def __getitem__(self, index):
        image = torch.rand((3, self.input_size[0], self.input_size[1]))
        dense_label = torch.argmax(image, dim=0)
        sparse_label = one_hot(dense_label, num_classes=3).movedim(2, 0).float()

        return image, sparse_label
