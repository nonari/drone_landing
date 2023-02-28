from os import path, listdir
from glob import glob
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


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


def main():
    orig = path.expanduser("~/tfm/ruralscapes")
    dest = path.join(orig, "labels/segprop_labels")

    results_dirs = glob(path.join(orig, "output_2k/i03/*"))

    for d in results_dirs:
        name = path.basename(d)
        if len(name) > 8:
            frames_paths = glob(path.join(d, '*'))
            for frame_path in frames_paths:
                image = generate_label(frame_path)
                num = frame_path[-10:-4]
                dest_name = path.join(dest, f'segfull_{name[0:8]}_{num}.png')
                image.save(dest_name)


def generate_label(sparse_path):
    sparse_label = np.load(sparse_path)
    sparse_label = sparse_label['map']
    label = np.argmax(sparse_label, axis=2).T
    color_label = ruralscapes_color_keys[label].astype(np.uint8)
    im = Image.fromarray(color_label)
    return im


if __name__ == "__main__":
    main()
