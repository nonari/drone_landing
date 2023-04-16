import collections.abc
from typing import Iterator
from glob import glob
from os import path
import asyncio
import threading

import matplotlib.pyplot as plt
from skimage import morphology, color, feature, registration, measure, transform
from skimage import draw as skdraw
import cv2 as cv
from PIL import Image
import numpy as np
from tracking.Utils import FramesSequenceUAV123, BlockingBuffer, anim

dataset_root = path.expanduser('~/Documentos/Dataset_UAV123_10fps')
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

SIZE = 1280, 720


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
    corner_points = feature.corner_peaks(response, min_distance=7, num_peaks=num_peaks,
                                         threshold_rel=0.1)
    return corner_points.astype(np.float32)


def to_uint8(im):
    return (im * 255).astype(np.uint8)


def roi_to_vertices(x, y, w, h):
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y
    x3 = x + w
    y3 = y + h
    x4 = x
    y4 = y + h

    vertices = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    return vertices


def clean_points_in_rois(points, rois):
    for roi in rois:
        vertices = roi_to_vertices(*roi)
        result = measure.points_in_poly(points, vertices)
        points = points[np.logical_not(result)]

    return points


def clean_points_outside_frame(target, points2, size):
    frame_vertices = roi_to_vertices(0, 0, *size)
    result = measure.points_in_poly(target[:, ::-1], frame_vertices)
    target = target[result]
    points2 = points2[result]
    return target, points2


def cv_optical_flow_lk(fst_frame, snd_frame, points):
    p0 = points[:, None, ::-1]
    p1, st, err = cv.calcOpticalFlowPyrLK(
        to_uint8(fst_frame),
        to_uint8(snd_frame), p0, None, **lk_params)
    p1 = np.squeeze(p1, axis=1)[:, ::-1]
    return p1


def cv_find_homography(p_ori, p_dst):
    h_mat, err = cv.findHomography(p_ori[:, ::-1].astype(np.float32),
                                   p_dst[:, ::-1].astype(np.float32), method=cv.RANSAC)
    return h_mat


def pad_to(im, size):
    sh, sw = size
    oh, ow = im.shape[:2]
    dh = max(sh - oh, 0)
    dw = max(sw - ow, 0)
    padded_im = np.pad(im, ((0, dh), (0, dw), (0, 0)), constant_values=0)
    return padded_im


def pad_imgs(im1, im2):
    size1 = im1.shape[:2]
    size2 = im2.shape[:2]
    if size1[0] > size2[0]:
        im2 = pad_to(im2, (size1[0], 0))
    elif size1[0] < size2[0]:
        im1 = pad_to(im1, (size2[0], 0))

    if size1[1] > size2[1]:
        im2 = pad_to(im2, (0, size1[1]))
    elif size1[1] < size2[1]:
        im1 = pad_to(im1, (0, size2[1]))

    return im1, im2


def cv_merge(im1, im2):
    im1, im2 = pad_imgs(im1, im2)
    merged = cv.addWeighted(im1, 0.5, im2, 0.5, 0)
    return merged


def cv_warp(ori_im, dst_im, h_mat):
    width = ori_im.shape[1] + dst_im.shape[1]
    heigh = ori_im.shape[0] + dst_im.shape[0]
    warped_im = cv.warpPerspective(ori_im, h_mat, (width, heigh))
    # warped_im[:dst_im.shape[0], :dst_im.shape[1]] = dst_im

    return warped_im


plt.figure(dpi=1200)


def main():
    seq.__next__()
    old_frame, r = seq.__next__()
    old_gray = color.rgb2gray(old_frame)
    p0 = find_corners(old_gray)
    mask = points_to_mask(p0, SIZE[::-1])
    drawn_im = draw_mask(old_frame, mask)
    buff.push(drawn_im)
    # plt.imshow(mask)
    # plt.show()
    p0 = clean_points_in_rois(p0, [r])
    for curr_frame, curr_roi in seq[5:15]:
        curr_gray = color.rgb2gray(curr_frame)
        p1 = cv_optical_flow_lk(old_gray, curr_gray, p0)
        key_p0 = [cv.KeyPoint(i[1], i[0], 1) for i in p0]
        key_p1 = [cv.KeyPoint(i[1], i[0], 1) for i in p1]
        matches = [cv.DMatch(i, i, 1) for i in range(len(key_p0))]
        matched = cv.drawMatches(old_frame, key_p0, curr_frame, key_p1,
                                 matches,
                                 None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        p0, p1 = clean_points_outside_frame(p0, p1, SIZE)
        h_mat = cv_find_homography(p0, p1)
        warped_old = cv_warp(old_frame, curr_frame, h_mat)
        merged_im = cv_merge(warped_old, curr_frame)
        mask = points_to_mask(p1, SIZE[::-1], mask)
        drawn_im = draw_mask(curr_frame, mask)
        plt.imshow(merged_im)
        plt.show()
        plt.imshow(matched)
        plt.show()
        buff.push(drawn_im)
        old_frame = curr_frame
        old_gray = curr_gray
        p0 = find_corners(old_gray)
        p0 = clean_points_in_rois(p0, [curr_roi])


    buff.close()


main()
# thread = threading.Thread(target=main)
# thread.start()
# anim(buff)
# thread.join()
