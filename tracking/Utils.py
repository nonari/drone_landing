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

    def __init__(self, root, video_name):
        frames_paths = glob(path.join(root, 'UAV123_10fps', 'data_seq', 'UAV123_10fps', video_name, '*'))
        self.frames_paths = sorted(frames_paths, key=lambda x: int(path.basename(x)[:-4]))

        self.rois = []
        annotations_path = path.join(root, 'UAV123_10fps', 'anno', 'UAV123_10fps', f'{video_name}.txt')
        with open(annotations_path, 'r') as annotations_file:
            for line in annotations_file:
                roi_parts = line.split(',')
                roi = int(roi_parts[1]), int(roi_parts[0]), int(roi_parts[3]), int(roi_parts[2])
                self.rois.append(roi)

        self.curr = 0

    def __next__(self):
        image = Image.open(self.frames_paths[self.curr])
        tup = np.asarray(image), self.rois[self.curr]
        self.curr += 1
        if self.curr >= len(self.frames_paths):
            raise StopIteration
        return tup


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
