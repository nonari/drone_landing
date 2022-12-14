import numpy
import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import Config
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import torch
from os import path
from glob import glob
from PIL import Image
import importlib


def transform_tugraz(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class_names = [
    'unlabeled',
]

tugraz_color_keys = np.asarray([
    [0, 0, 0],
    [128, 64, 128],
    [130, 76, 0],
    [0, 102, 0],
    [112, 103, 87],
    [28, 42, 468],
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


def label_to_tensor_v2(label, keys):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(keys[:, 0, 0, :]*v, axis=1)
    one_ch = np.sum(label * np.asarray(v)[None, None], axis=2)
    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.float32)
    return torch.tensor(sparse)


def label_to_tensor(label, color_mask):
    sparse_mask = np.all(np.equal(label, color_mask), axis=3).astype(np.float32)
    return torch.tensor(sparse_mask)


class TUGrazDataset(Dataset):
    def __init__(self, options, folds=1, shuffle=False, idx_ord=None):
        subset = 'training_set' if options.train else 'testing_set'
        images_root = path.join(options.tugraz_root, subset, options.tugraz_images_loc)
        labels_root = path.join(options.tugraz_root, subset, f'gt/semantic/{options.tugraz_labels_loc}')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x)[:3]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))

        image_paths = np.asarray(image_paths)
        label_paths = np.asarray(label_paths)

        net_config = importlib.import_module(f'net_configurations.{options.model_config}').CONFIG
        t_tugraz = transform_tugraz(net_config['input_size'])

        self._idx = np.arange(image_paths.shape[0])
        if idx_ord is not None:
            image_paths = image_paths[idx_ord]
            label_paths = label_paths[idx_ord]
            self._idx = idx_ord
        elif shuffle:
            idx = np.arange(image_paths.shape[0])
            np.random.shuffle(idx)
            image_paths = image_paths[idx]
            label_paths = label_paths[idx]
            self._idx = idx

        print('Loading images...')
        self.images = []
        for im_path in image_paths[:-380]:
            im = Image.open(im_path)
            self.images.append(t_tugraz(im))
            im.close()

        print('Loading labels...')
        self.labels = []
        for lab_path in label_paths[:-360]:
            lab = Image.open(lab_path)
            lab_res = lab.resize(net_config['input_size'], Image.NEAREST)
            lab.close()
            self.labels.append(lab_res)

        self._part_len = self.images.__len__() // folds
        self._folds = folds
        if folds > 1:
            self._len = self._part_len * (folds - 1)
        else:
            self._len = self.image.__len__()
        self._fold = 0

    def change_fold(self, k):
        self._fold = k

    def get_index(self):
        return self._idx

    def classes(self):
        return 24

    def __len__(self):
        return self._len

    def __getitem__(self, item):
        shifted = (item + self._fold * self._part_len) % (self._part_len * self._folds)
        print(item, '->', shifted)
        return self.images[shifted], label_to_tensor_v2(self.labels[shifted],
                                                        tugraz_color_keys[:, None, None]).moveaxis(2, 0)


if __name__ == '__main__':
    conf = Config()
    tg = TUGrazDataset(conf)
    print(tg.__len__())
