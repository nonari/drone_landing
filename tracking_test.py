import collections.abc
from typing import Iterator
from glob import glob
from os import path
import asyncio
import threading

import matplotlib.pyplot as plt
from skimage import morphology
import cv2 as cv
from PIL import Image
import numpy as np
from tracking.Utils import FramesSequenceUAV123, BlockingBuffer, anim


dataset_root = path.expanduser('~/windows/Dataset_UAV123_10fps')
seq = FramesSequenceUAV123(dataset_root, 'bike1')
buff = BlockingBuffer()

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7
                      )
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

SIZE = 720, 1280


def points_to_mask(points, size):
    mask = np.zeros(size)
    mask[points[:, 0, 0].astype(int), points[:, 0, 1].astype(int)] = 1
    mask = morphology.dilation(mask, selem=morphology.selem.disk(radius=4))
    return mask


def draw_mask(im, mask, color=(0, 255, 0)):
    im = im.copy()
    im[mask.astype(bool)] = color
    return im


def main():
    old_frame, r = seq.__next__()
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    mask = points_to_mask(p0[..., ::-1], SIZE)
    mask = draw_mask(old_frame, mask)
    plt.imshow(mask)
    plt.show()
    for im, r in seq:
        buff.push(im)

    buff.close()


main()
# x = threading.Thread(target=main)
# x.start()
# anim(buff)
# x.join()

