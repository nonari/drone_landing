import numpy as np
from config import Config
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torchvision.transforms import InterpolationMode
import torch
from os import path
from glob import glob
from PIL import Image


transform_tugraz = transforms.Compose([
    transforms.Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

tugraz_color_keys = np.asarray([
        [128, 64, 128],
        [130, 76, 0],
        [0, 102, 0]
    ])[:, None, None]


# def expand_color_keys(color_keys, w, h):
#     color_pixel_mask = np.asarray(color_keys)[None, None]
#     return np.tile(color_pixel_mask, (1, 1, 1, h*w)).reshape((-1, h, w, 3))


def label_to_tensor(label, color_mask):
    sparse_mask = np.all(np.equals(label, color_mask), axis=3).astype(np.float32)
    return torch.tensor(sparse_mask)


class TUGraz(Dataset):
    def __init__(self, options):
        subset = 'training_set' if options.train else 'testing_set'
        images_root = path.join(options.tugraz_root, subset, 'images')
        labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_images')
        image_paths = glob(images_root + '/*.jpg')
        label_paths = glob(labels_root + '/*.png')
        image_paths = sorted(image_paths, key=lambda x: int(path.basename(x)[:3]))
        label_paths = sorted(label_paths, key=lambda x: int(path.basename(x)[:3]))
        self.images = [Image.open(i) for i in image_paths]
        self.labels = [Image.open(i) for i in label_paths]

    def __len__(self):
        return self.images.__len__()

    def __getitem__(self, item):
        return self.images[item], self.labels[item]
