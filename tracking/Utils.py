import collections.abc
from typing import Iterator
from glob import glob
from os import path
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
import numpy as np


class FramesSequenceUAV123(collections.abc.Iterable):
    def __iter__(self) -> Iterator:
        return self

    def __init__(self, root, video_name, rois=None):
        frames_paths = glob(path.join(root, 'UAV123_10fps', 'data_seq', 'UAV123_10fps', video_name, '*'))
        self._root = root
        self._video_name = video_name
        self._frames_paths = sorted(frames_paths, key=lambda x: int(path.basename(x)[:-4]))

        self._rois = []
        annotations_path = path.join(root, 'UAV123_10fps', 'anno', 'UAV123_10fps', f'{video_name}.txt')
        if rois is None:
            with open(annotations_path, 'r') as annotations_file:
                for line in annotations_file:
                    roi_parts = line.split(',')
                    roi = int(roi_parts[1]), int(roi_parts[0]), int(roi_parts[3]), int(roi_parts[2])
                    self._rois.append(roi)
        else:
            self._rois = rois
        self._curr = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            obj = FramesSequenceUAV123(self._root, self._video_name, self._rois)
            obj._frames_paths = self._frames_paths.__getitem__(slice(*key.indices(len(self._frames_paths))))
            obj._rois = self._rois.__getitem__(slice(*key.indices(len(self._frames_paths))))
            return obj
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self._frames_paths)
            if key < 0 or key >= len(self._frames_paths):
                raise IndexError(f"The index {key} is out of range.")
            return self.get_idx(key)  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def get_idx(self, idx):
        image = Image.open(self._frames_paths[idx])
        tup = np.asarray(image), self._rois[idx]
        return tup

    def __next__(self):
        if self._curr >= len(self._frames_paths):
            raise StopIteration
        tup = self.get_idx(self._curr)
        self._curr += 1
        return tup


class AnySeq(FramesSequenceUAV123):
    def __iter__(self) -> Iterator:
        return self

    def __init__(self, root, video_name):
        frames_paths = glob(path.join(root, 'JPEGImages', video_name + '*'))
        self._root = root
        self._video_name = video_name
        self._frames_paths = sorted(frames_paths, key=lambda x: int(path.basename(x)[:-4]))
        self._rois = [[0, 0, 0, 0]] * len(frames_paths)
        self._curr = 0


class BlockingBuffer:
    def __init__(self):
        self.buff = []
        self.finish = False

    def close(self):
        self.finish = True

    def push(self, elem):
        if self.finish:
            exit(0)
        self.buff.append(elem)
        sleep(0.001)

    def pop(self):
        while not self.finish:
            if len(self.buff) > 0:
                return self.buff.pop(0)
            else:
                sleep(0.01)
        exit(0)


def anim(buff):
    ax1 = plt.subplot(1, 1, 1)
    pl1 = ax1.imshow(buff.pop())

    def update(i):
        im1 = buff.pop()
        pl1.set_data(im1)

    ani = FuncAnimation(plt.gcf(), update, interval=100)

    def close(event):
        if event.key == 'q':
            plt.close(event.canvas.figure)
            buff.close()
    cid = plt.gcf().canvas.mpl_connect("key_press_event", close)

    plt.show()
