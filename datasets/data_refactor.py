from os import path, makedirs

import numpy as np
from PIL import Image

from datasets.tugraz import tugraz_color_keys
from config import Config
from glob import glob
import cv2
import pprint

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
    subset = 'training_set'
    new_images_folder = 'low_res_im'
    new_labels_folder = 'label_im_low_res'

    options = Config()
    images_root = path.join(options.tugraz_root, subset, 'images')
    labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_images')

    new_images_dir = path.join(path.dirname(images_root), new_images_folder)
    new_labels_dir = path.join(path.dirname(labels_root), new_labels_folder)
    makedirs(new_images_dir, exist_ok=True)
    makedirs(new_labels_dir, exist_ok=True)

    image_paths = glob(images_root + '/*.jpg')
    label_paths = glob(labels_root + '/*.png')

    print('Loading images...')
    images = [Image.open(i) for i in image_paths]
    print('Loading labels...')
    labels = [Image.open(i) for i in label_paths]

    im_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            new_images_folder, path.basename(x)), image_paths))
    lb_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            new_labels_folder, path.basename(x)), label_paths))
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


def collapse_label(label_im, equivalence, color_map):
    v = [256 * 256, 256, 1]
    one_ch = np.sum(label_im * np.asarray(v)[None, None], axis=2)
    for k in equivalence.keys():
        key_color = np.sum(color_map[k] * v)
        other_colors = np.sum(color_map[equivalence[k]] * [v], axis=1)
        one_ch[np.isin(one_ch, other_colors)] = key_color

    r = one_ch // (256*256)
    p1 = one_ch - 256*256*r
    g = p1 // 256
    b = p1 - 256*g

    return np.dstack([r, g, b])


def refactor_tugraz_labels():
    subset = 'training_set'
    new_labels_folder = 'label_collapsed'

    options = Config()
    labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_im_low_res')

    new_labels_dir = path.join(path.dirname(labels_root), new_labels_folder)
    makedirs(new_labels_dir, exist_ok=True)

    label_paths = glob(labels_root + '/*.png')

    print('Loading labels...')
    labels = [Image.open(i) for i in label_paths]

    lb_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            new_labels_folder, path.basename(x)), label_paths))
    for p, imo in zip(lb_paths, labels):
        npim = collapse_label(np.asarray(imo), tugraz_new_labels, tugraz_color_keys)
        imr = Image.fromarray(npim.astype(np.uint8))
        imr.save(p)
        imr.close()
        imo.close()


def extract_ruralscapes():
    root = '/home/nonari/Documentos/ruralscapes/videos'
    videos = glob(root+'/*')
    for v in videos:
        name = path.basename(v).split('.')[0]
        prefix = path.join(path.dirname(root), 'frames', name)
        extract_frames(v, prefix)


def extract_frames(video_file, prefix):
    cap = cv2.VideoCapture(video_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS: {fps}')
    count = 0
    last_frame = None
    while True:
        is_read, frame = cap.read()
        if not is_read:
            resized = cv2.resize(last_frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(prefix + f"_{(count-1):06}.jpg", resized)
            cap.release()
            break
        elif count % 50 == 0:
            resized = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(prefix + f"_{count:06}.jpg", resized)
        last_frame = frame
        count += 1


def group():
    frame_paths = glob('/home/nonari/Documentos/ruralscapes/frames/*')
    ids = {}
    for p in frame_paths:
        idv = path.basename(p).split('_')[1]
        if idv not in ids:
            ids[idv] = 1
        else:
            ids[idv] += 1
    keys = sorted(ids.keys(), key=lambda x: ids[x])
    for k in keys:
        print(k, ids[k])


def downsize_rural():
    frame_paths = glob('/home/nonari/Documentos/ruralscapes/labels/manual_labels/*/*')
    for p in frame_paths:
        label = Image.open(p)
        res_label = label.resize((1280, 720), Image.NEAREST)
        new_path = path.join(path.dirname(path.dirname(path.dirname(p))), 'resized_labels', path.basename(p))
        res_label.save(new_path)
        res_label.close()
        label.close()


group()
