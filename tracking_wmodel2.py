import collections.abc
from time import time
from typing import Iterator
from glob import glob
from os import path
import asyncio
import threading
import segmentation_models_pytorch as smp
from datasets.synthetic import synthetic_risks as risks
from matplotlib import colors as colorm
import matplotlib.pyplot as plt
from torchvision.transforms import transforms, InterpolationMode
import skimage.filters
import torch
from skimage import morphology, color, feature, registration, measure, transform, draw
from skimage import draw as skdraw
import cv2 as cv
from PIL import Image
import numpy as np
from skimage.morphology import dilation, disk
from skimage.filters import threshold_otsu
from datasets.synthetic import synthetic_color_keys as colors

from custom_models.safeuav import UNet_MDCB
from datasets.dataset import adapt_image
from tracking.Utils import FramesSequenceUAV123, BlockingBuffer, anim, AnySeq

dataset_root = path.expanduser('~/Documentos/aeroscapes')
seq = AnySeq(dataset_root, '000004')
buff = BlockingBuffer()

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7
                      )
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(25, 25),
                 maxLevel=4,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

SIZE = 1280, 720


def mask_with_ones(im, labels):
    mask = None
    for l in labels:
        if mask is None:
            mask = (im == l)
        else:
            mask = mask or (im == l)
    mask = True ^ mask
    return mask.astype(np.uint8)


def area(binary_im, max_dist, decay='linear', risk_lv=1, convert=True):
    if convert:
        binary_im = binary_im.astype(np.uint8)
    distance_map = cv.distanceTransform(binary_im, cv.DIST_L2, 3)
    zero = (distance_map < max_dist).astype(float)
    distance_map[distance_map > max_dist] = 0
    if decay == 'solid':
        return zero * risk_lv
    risk = zero - distance_map / max_dist
    if decay == 'log':
        risk = np.log2(risk + 1)

    return risk * risk_lv


def measure_max_width(binary_mask, angle):
    binary_array = np.array(binary_mask)
    rotated_array = np.rot90(binary_array, int(angle / 90))

    iterator = rotated_array

    max_width = 0
    for row_or_col in iterator:
        x = np.where(row_or_col == 1)[0]
        max_width = max(max_width, x.max() - x.min())

    return max_width


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


def find_corners(im, num_peaks=100, sigma=1, min_distance=7, threshold_rel=0.1):
    response = feature.corner_shi_tomasi(im, sigma=sigma)
    corner_points = feature.corner_peaks(response, min_distance=min_distance, num_peaks=num_peaks,
                                         threshold_rel=threshold_rel)
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
    if len(im.shape) == 3:
        padded_im = np.pad(im, ((0, dh), (0, dw), (0, 0)), constant_values=0)
    else:
        padded_im = np.pad(im, ((0, dh), (0, dw)), constant_values=0)

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


def cv_merge(im1, im2, w1=0.5):
    im1, im2 = pad_imgs(im1, im2)
    merged = cv.addWeighted(im1, w1, im2, 1-w1, 0)
    return merged


def cv_warp(ori_im, h_mat):
    width = ori_im.shape[1]
    heigh = ori_im.shape[0]
    warped_im = cv.warpPerspective(ori_im, h_mat, (width, heigh))
    # warped_im[:dst_im.shape[0], :dst_im.shape[1]] = dst_im

    return warped_im


def flow_to_color(flow):
    # magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
    x, y = flow[:, :, 0], flow[:, :, 1]
    magnitude = np.sqrt(x ** 2 + y ** 2)
    angle = np.arctan2(y, x)
    mask = np.zeros((*flow.shape[:2], 3))
    mask[..., 1] = 255
    mask[..., 0] = angle * 180 / np.pi / 2
    mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)

    rgb = cv.cvtColor(mask.astype(np.uint8), cv.COLOR_HSV2BGR)

    return rgb


def optical_flowb(im1, im2):
    im1_gray = to_uint8(color.rgb2gray(im1))
    im2_gray = to_uint8(color.rgb2gray(im2))
    im1_gray, im2_gray = pad_imgs(im1_gray, im2_gray)
    h, w = im1_gray.shape
    p0 = np.array([(a, b) for a in range(w) for b in range(h)]).astype(np.float32)
    p0int = p0.astype(int)

    p1, _, err = cv.calcOpticalFlowPyrLK(im1_gray, im2_gray, p0, None)
    p1[:, 0] = np.clip(p1[:, 0], 0, w - 1)
    p1[:, 1] = np.clip(p1[:, 1], 0, h - 1)
    p1 = p1.astype(int)

    moved = np.sum(np.abs(p0 - p1), axis=1) > 2
    shift = p0 - p1
    directions = np.arctan2(shift[:, 1], shift[:, 0])
    directions = ((directions + np.pi) / (2 * np.pi))
    valid = directions[moved]
    frame = np.zeros((im2_gray.shape[0], im2_gray.shape[1], 3))
    frame[p1[moved][:, 1], p1[moved][:, 0], 0] = valid
    frame[p1[moved][:, 1], p1[moved][:, 0], 1] = 0.5
    frame[p1[moved][:, 1], p1[moved][:, 0], 2] = 0.5
    # frame[p0int[moved][:, 1], p0int[moved][:, 0]] = 0, 1, 0
    frame = (color.hsv2rgb(frame) * 255).astype(np.uint8)
    return frame


def optical_flow(im1, im2):
    im1_gray = color.rgb2gray(im1)
    im2_gray = color.rgb2gray(im2)
    im1_gray, im2_gray = pad_imgs(im1_gray, im2_gray)
    flow = cv.calcOpticalFlowFarneback(
        im1_gray, im2_gray, None, 0.5, 5, 25, 3, 5, 1.2, 0)

    return flow


def crop_warped(matrix, size, im1, im2):
    h, w = size
    coords = np.array([[0, 0, 1], [0, h, 1], [w, h, 1], [w, 0, 1]]).T
    coords = np.dot(matrix, coords).astype(int)
    max_x_left = np.clip(max(coords[0, 0], coords[0, 1]), 0, w)
    min_x_right = np.clip(min(coords[0, 2], coords[0, 3]), 0, w)
    min_y_bot = np.clip(min(coords[1, 1], coords[1, 2]), 0, h)
    max_y_top = np.clip(max(coords[1, 3], coords[1, 0]), 0, h)
    crop1 = im1[max_y_top:min_y_bot, max_x_left:min_x_right]
    crop2 = im2[max_y_top:min_y_bot, max_x_left:min_x_right]
    return crop1, crop2, (max_y_top, min_y_bot, max_x_left, min_x_right)

def cv_corners(gray):
    gray = to_uint8(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)
    ret, dst = cv.threshold(dst, 0.01 * dst.max(), 1, 0)
    x, y = np.where(dst == 1)
    return np.hstack([x[:, None], y[:, None]])

def detect_movement(im1, im2):
    im1_gray = color.rgb2gray(im1)
    im2_gray = color.rgb2gray(im2)
    t0=time()
    im1_fil = cv.GaussianBlur(im1_gray, (5, 5), 2)
    im2_fil = cv.GaussianBlur(im2_gray, (5, 5), 2)
    # im1_fil = skimage.filters.gaussian(im1_gray, sigma=2)
    # im2_fil = skimage.filters.gaussian(im2_gray, sigma=2)
    print(f'tttt{time()-t0}')
    im1_gray, im2_gray = pad_imgs(im1_fil, im2_fil)

    merged_movement = np.abs(im1_gray - im2_gray)
    merged_movement = downscale(merged_movement, 2)
    print(f'abs{time() - t0}')
    border = np.zeros(im1_gray.shape)
    print(f'zeros{time() - t0}')
    border[im1_gray == 0] = 1
    border = downscale(border, 2)
    print(f'shift{time() - t0}')
    # border = dilation(border, selem=disk(25))
    # border = cv.dilate(border, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25)))
    print(f'dilate{time() - t0}')
    merged_movement[border == 1] = 0
    t = threshold_otsu(merged_movement)
    merged_movement[merged_movement < t] = 0
    merged_movement[merged_movement > 0] = 1
    print(f'merged{time() - t0}')
    merged_movement = cv.dilate(merged_movement, cv.getStructuringElement(cv.MORPH_ELLIPSE, (6, 6)))
    # merged_movement = dilation(merged_movement, selem=disk(6))
    merged_movement = upscale(merged_movement, 2)
    print(f'dil2- {time() - t0}')
    # plt.imshow(merged_movement)
    # plt.show()
    return merged_movement


def detect_objects(old_frame, curr_frame, movement, old_label, curr_label):
    person = (old_label == 7)
    curr_person = (curr_label == 7)
    person_moving_curr = np.logical_and(movement, curr_person).astype(np.uint8)
    person_moving_old = np.logical_and(movement, person).astype(np.uint8)

    cc_mov_curr = measure.label(person_moving_curr)
    cc_mov_old = measure.label(person_moving_old)
    prop_mov_old = measure.regionprops(cc_mov_old)
    prop_mov_curr = measure.regionprops(cc_mov_curr)
    for prop in prop_mov_old:
        if prop.area < 1000:
            person_moving_old[cc_mov_old == prop.label] = 0
    centroids_curr = {}
    all_dest = set()
    for prop in prop_mov_curr:
        if prop.area < 1000:
            person_moving_curr[cc_mov_curr == prop.label] = 0
        else:
            all_dest.add(prop.label)
            centroids_curr[prop.label] = np.array(prop.centroid)
    plt.imshow(person_moving_old)
    plt.show()
    plt.imshow(person_moving_curr)
    plt.show()

    y, x = np.where(person_moving_old == 1)
    mov_coords = np.hstack([y[:, None], x[:, None]]).astype(np.float32)
    idx = np.random.choice(np.arange(len(y)), int(len(y) * 0.05), replace=False)
    p0 = mov_coords[idx]
    p1 = cv_optical_flow_lk(old_frame, curr_frame, p0)
    p1, p0 = clean_points_outside_frame(p1, p0, curr_frame.shape[:2][::-1])
    p1int = p1.astype(int)
    dest_cc = cc_mov_curr[p1int[:, 0], p1int[:, 1]]
    p0int = p0.astype(int)
    orig_cc_all = set(np.unique(cc_mov_old[p0int[:, 0], p0int[:, 1]]))
    inside = dest_cc.astype(bool)
    p1 = p1[inside]
    p0 = p0[inside]
    dest_cc = dest_cc[inside]
    p0int = p0.astype(int)
    orig_cc = cc_mov_old[p0int[:, 0], p0int[:, 1]]
    not_found = orig_cc_all - set(orig_cc)
    r = [[]] * max(orig_cc_all)
    if len(set(orig_cc)) > 0:
        for ori, dest in zip(orig_cc, dest_cc):
            if len(r[ori-1]) == 0:
                r[ori-1] = [0] * max(all_dest)
            r[ori-1][dest-1] += 1
    for i in not_found:
        old_centroid = np.array(prop_mov_old[i-1].centroid)
        distances = {}
        for i in centroids_curr:
            dist = np.sqrt(np.sum(np.power(centroids_curr[i] - old_centroid, 2)))
            distances[i] = dist
        maxd = 9999
        lb = 0
        for ddd in distances:
            if distances[ddd] < maxd:
                lb = ddd
                maxd = distances[ddd]
        if maxd < 200:
            r[i-1] = [0]*max(all_dest)
            r[i-1][lb-1] = 99

    extensions = []
    for idx, ld in enumerate(r):
        if len(ld) > 0:
            dest = np.argmax(ld)
            c0 = np.array(prop_mov_old[idx].centroid)
            c1 = np.array(prop_mov_curr[dest].centroid)
            yt, xl, yb, xr = prop_mov_curr[dest].bbox
            vec = c1 - c0
            angle = np.rad2deg(np.arctan2(*vec))
            mag = np.sqrt(np.sum(np.power(vec, 2)))
            crop = person_moving_curr[yt:yb, xl:xr]
            width = measure_max_width(crop, angle)
            py, px = c1
            y_poly = np.array([py, (mag * 40)+py, (mag * 20)+py, py]).astype(int)
            x_poly = np.array([px-width/2, px-width/2, px+width/2, px+width/2]).astype(int)
            rectangle = draw.polygon(y_poly, x_poly, shape=person_moving_old.shape)
            rectangle = np.vstack(rectangle).T
            rot_rect = rotate_points(rectangle, (py, px), 90-np.median(angle), shape=person_moving_old.shape)
            extensions.append(rot_rect)

    return extensions


def rotate_points(points, center, angle, shape=None):
    angle_rad = np.radians(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Translate points relative to the center
    translated_points = points - center

    # Perform the rotation
    rotated_y = translated_points[:, 0] * cos_theta - translated_points[:, 1] * sin_theta
    rotated_x = translated_points[:, 0] * sin_theta + translated_points[:, 1] * cos_theta

    # Translate the rotated points back relative to the original center
    rotated_x += center[1]
    rotated_y += center[0]

    if shape is not None:
        rotated_x = np.clip(rotated_x, 0, shape[1] - 1)
        rotated_y = np.clip(rotated_y, 0, shape[0] - 1)

    rotated_points = np.stack((rotated_y, rotated_x), axis=1)

    return rotated_points


def init_model():
    unet = smp.Unet(classes=8, encoder_name='resnet18')
    model = torch.load(
        '/home/nonari/PycharmProjects/drone_landing/executions/multiple/aerobyuavid/models/0',
        map_location='cpu')
    unet.load_state_dict(model['model_state_dict'])
    totensor = adapt_image((704, 1024))

    def inn(npim):
        orig_size = npim.shape[:2]
        impil = Image.fromarray(npim)
        tensor = totensor(impil)
        pred = unet(tensor.unsqueeze(dim=0))
        label = torch.argmax(pred, dim=1)
        label = transforms.Resize(orig_size, interpolation=InterpolationMode.NEAREST)(label)
        label = label.squeeze(dim=0).numpy().astype(np.uint8)
        return label, colors[label].astype(np.uint8)
    return inn


def downscale(a, factor):
    return a[::factor, ::factor]


def upscale(a, factor):
    fy = np.repeat(a, factor, axis=0)
    fx = np.repeat(fy, factor, axis=1)
    return fx


def clean_points(p0, label):
    label = label.copy()
    label[label == 6] = 2
    label[label == 7] = 2
    binary_im = mask_with_ones(label, [2])
    distance_map = cv.distanceTransform(binary_im, cv.DIST_L2, 3)
    zero = distance_map < 20

    label[zero] = 2
    poslab = label[p0[:, 0], p0[:, 1]]
    pfilt = p0[poslab != 2].astype(np.float32)
    return pfilt


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colorm.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)[::-1]))
    return new_cmap


cmap = plt.get_cmap('hsv')
new_cmap = truncate_colormap(cmap, 0.0, 0.333)

plt.figure(dpi=1200)
torch.set_grad_enabled(False)

ite = [0]


def main():
    old_frame, r = seq[56]
    net = init_model()
    old_label, old_im_label = net(old_frame)
    old_gray = color.rgb2gray(old_frame)
    p0 = cv_corners(old_gray)
    mask = points_to_mask(p0, SIZE[::-1])
    drawn_im = draw_mask(old_frame, mask)
    buff.push(drawn_im)
    p0 = clean_points(p0, old_label)
    for curr_frame, curr_roi in seq[57:]:
        curr_label, curr_im_label = net(curr_frame)
        curr_gray = color.rgb2gray(curr_frame)
        p1 = cv_optical_flow_lk(old_gray, curr_gray, p0)
        p0, p1 = clean_points_outside_frame(p0, p1, SIZE)
        h_mat = cv_find_homography(p0, p1)
        warped_old = cv_warp(old_frame, h_mat)
        w_label_old = cv_warp(old_label, h_mat)
        warp_old_gray = color.rgb2gray(warped_old)
        merged_im = cv_merge(warped_old, curr_frame)
        plt.imshow(merged_im)
        plt.show()
        mask = points_to_mask(p1, SIZE[::-1], mask)
        movement = detect_movement(warp_old_gray, curr_gray)
        ext = detect_objects(warped_old, curr_frame, movement, w_label_old, curr_label)
        drawn_im = draw_mask(curr_frame, mask)
        curr_label_copy = curr_label.copy()
        for i in ext:
            curr_label_copy[(i[:, 0]).astype(int), (i[:, 1]).astype(int)] = 7
        person_bin = mask_with_ones(curr_label_copy, [7])
        risk = risks[curr_label_copy]
        person_risk = area(person_bin, 50, decay='solid', convert=True)
        risk += person_risk
        risk = np.clip(risk, 0, 1)
        ite[0] += 1
        plt.imshow(risk)
        plt.show()
        # plt.imshow(merged_im)
        # plt.show()
        # plt.imshow(matched)
        # plt.show()
        print('dddddddddddddd')
        buff.push(drawn_im)
        old_frame = curr_frame
        old_gray = curr_gray
        old_label = curr_label
        p0 = cv_corners(old_gray)
        p0 = clean_points(p0, old_label)

    buff.close()


def match_points(curr_frame, old_frame, p0, p1):
    key_p0 = [cv.KeyPoint(i[1], i[0], 1) for i in p0]
    key_p1 = [cv.KeyPoint(i[1], i[0], 1) for i in p1]
    matches = [cv.DMatch(i, i, 1) for i in range(len(key_p0))]
    matched = cv.drawMatches(old_frame, key_p0, curr_frame, key_p1,
                             matches,
                             None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return matched


main()
# thread = threading.Thread(target=main)
# thread.start()
# anim(buff)
# thread.join()
