import numpy as np
from os import path, makedirs
from glob import glob
from PIL import Image


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

ROOT = '/home/nonari/windows/ruralscapes/labels/'


def compute_sparse_label(label):
    v = np.asarray([256 * 256, 256, 1])
    one_key = np.sum(ruralscapes_color_keys * v, axis=1)
    one_ch = np.sum(label * v[None, None], axis=2)

    sparse = np.equal(one_key[None, None], one_ch[..., None]).astype(np.uint8)
    return sparse


def gt():
    orig_lab_dir = path.join(ROOT, f'manual_labels')
    dest_lab_dir = path.join(ROOT, f'sparse_labels')
    makedirs(dest_lab_dir, exist_ok=True)
    video_paths = glob(path.join(orig_lab_dir, '*'))
    for f_path in video_paths:
        name = path.basename(f_path)
        sparse_frames_dir = path.join(dest_lab_dir, name)
        makedirs(sparse_frames_dir, exist_ok=True)
        label_paths = glob(path.join(orig_lab_dir, f'{name}/*'))
        for label_path in label_paths:
            frame_no = label_path.split('.')[0][-6:]
            label = Image.open(label_path).resize((1280, 720))
            label = np.asarray(label)
            sparse = np.moveaxis(compute_sparse_label(label), 0, 1)
            frame_name = path.join(dest_lab_dir, name, f'frame_{frame_no}.npz')
            np.savez_compressed(frame_name, sparse)


if __name__ == '__main__':
    gt()
