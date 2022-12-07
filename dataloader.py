import numpy as np
from torch.utils.data import DataLoader, Dataset
from config import Config
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import torch
from os import path
from glob import glob
from PIL import Image


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
])[:, None, None]


def label_to_tensor(label, color_mask):
    sparse_mask = np.all(np.equal(label, color_mask), axis=3).astype(np.float32)
    return torch.tensor(sparse_mask)


class TUGrazDataset(Dataset):
    def __init__(self, options):
        subset = 'training_set' if options.train else 'testing_set'
        images_root = path.join(options.tugraz_root, subset, 'low_res_im')
        labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_im_low_res')

        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x)[:3]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))

        t_tugraz = transform_tugraz(options.new_size)

        print('Loading images...')
        self.images = [t_tugraz(Image.open(i)) for i in image_paths]
        print('Loading labels...')
        self.labels = [label_to_tensor(Image.open(i).resize(options.new_size, Image.NEAREST), tugraz_color_keys) for i in label_paths]

    def __len__(self):
        return self.images.__len__()

    def __getitem__(self, item):
        return self.images[item], self.labels[item]


if __name__ == '__main__':
    conf = Config()
    tg = TUGrazDataset(conf)
    print(tg.__len__())
