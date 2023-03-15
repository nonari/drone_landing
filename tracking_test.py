import collections.abc
from typing import Iterator
from glob import glob
from os import path
import asyncio
import threading

import matplotlib.pyplot as plt
from skimage import morphology, color, feature, registration
from skimage import draw as skdraw
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


def points_to_mask(points, size, old_mask=None):
    mask = np.zeros(size)
    if len(points.shape) > 2:
        points = np.squeeze(points, axis=1)
    points[:, 0] = np.clip(points[:, 0], 0, size[0] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, size[1] - 1)
    mask[points[:, 0].astype(int), points[:, 1].astype(int)] = 1
    mask = morphology.dilation(mask, selem=morphology.selem.disk(radius=4))
    if old_mask is not None:
        mask = np.clip(old_mask + mask, 0, 1)
    return mask


def draw_mask(im, mask, color=(0, 255, 0)):
    im = im.copy()
    im[mask.astype(bool)] = color
    return im


def find_corners(im, num_peaks=100):
    response = feature.corner_shi_tomasi(im, sigma=1)
    corner_points = feature.corner_peaks(response, min_distance=7, num_peaks=num_peaks)
    return corner_points.astype(np.float32)


def to_uint8(im):
    return (im * 255).astype(np.uint8)


def main():
    seq.__next__()
    old_frame, r = seq.__next__()
    old_gray = color.rgb2gray(old_frame)
    p0 = find_corners(old_gray)
    # p0 = cv.goodFeaturesToTrack(old_gray.astype(np.float32), mask=None, **feature_params)
    mask = points_to_mask(p0[..., ::], SIZE)
    drawn_im = draw_mask(old_frame, mask)
    buff.push(drawn_im)
    # plt.imshow(mask)
    # plt.show()
    p0 = p0[:, None, ::-1]
    for curr_frame, curr_roi in seq:
        curr_gray = color.rgb2gray(curr_frame)
        p1, st, err = cv.calcOpticalFlowPyrLK(
            to_uint8(old_gray),
            to_uint8(curr_gray), p0, None, **lk_params)
        mask = points_to_mask(p1[..., ::-1], SIZE, mask)
        drawn_im = draw_mask(curr_frame, mask)
        buff.push(drawn_im)
        p0 = p1
        old_gray = curr_gray
    buff.close()


# main()
x = threading.Thread(target=main)
x.start()
anim(buff)
x.join()

