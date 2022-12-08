from os import path

import numpy as np
from PIL import Image

from config import Config
from glob import glob


tugraz_classes = [
    "unlabeled",
    "paved-area",
    "dirt",
    "grass",
    "gravel",
    "water",
    "rocks",
    "pool",
    "vegetation",
    "root",
    "wall",
    "window",
    "door",
    "fence",
    "fence-pole",
    "person",
    "dog",
    "car",
    "bicycle",
    "tree",
    "bald-tree",
    "ar-marker",
    "obstacle",
    "conflicting"
]


tugraz_new_labels = {
    6: [13, 14, 22],
    5: [7],
    8: [19, 20],
    9: [10, 11, 12],
    17: [18]
}


def downsize_tugraz():
    new_size = 1500, 1000
    options = Config()
    subset = 'training_set'
    images_root = path.join(options.tugraz_root, subset, 'low_res_im')
    labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_im_low_res')

    image_paths = glob(images_root + '/*.jpg')
    label_paths = glob(labels_root + '/*.png')

    print('Loading images...')
    images = [Image.open(i) for i in image_paths]
    print('Loading labels...')
    labels = [Image.open(i) for i in label_paths]

    im_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            'low_res_im', path.basename(x)), image_paths))
    lb_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            'label_im_low_res', path.basename(x)), label_paths))
    for p, imo in zip(im_paths, images):
        imr = imo.resize(new_size, Image.BILINEAR)
        imr.save(p)
        imr.close()
        imo.close()

    for p, imo in zip(lb_paths, labels):
        imr = imo.resize(new_size, Image.NEAREST)
        imr.save(p)
        imr.close()
        imo.close()


def collapse_labels(label_im, equivalence, color_map):
    v = [256 * 256, 256, 1]
    one_ch = np.sum(label_im * np.asarray([v]), axis=2)
    for k in equivalence.keys():
        key_color = np.sum(color_map[k] * v)
        other_colors = np.sum(color_map[equivalence[k]] * [v], axis=1)
        one_ch[np.isin(one_ch, other_colors)] = key_color

    r = one_ch // 256*256
    p1 = one_ch - 256*256*r
    g = p1 // 256
    b = p1 - 256*g

    return np.dstack([r, g, b])

