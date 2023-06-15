import collections.abc
from time import time
from typing import Iterator
from glob import glob
from os import path
import asyncio
import threading
import segmentation_models_pytorch as smp
from datasets.synthetic import synthetic_risks as risks

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
from datasets.synthetic import TUGrazToSynthetic

SIZE = 1280, 720

def init_model():
    torch.set_grad_enabled(False)
    unet = smp.Unet(classes=8, encoder_name='resnet18')
    model = torch.load(
        '/home/nonari/PycharmProjects/drone_landing/executions/person/aeroandrural_tu/models/0',
        map_location=torch.device('cpu')
)
    unet.load_state_dict(model['model_state_dict'])
    unet.to(device=torch.device('cpu'))
    unet.train(mode=False)
    def inn(npim):
        pred = unet(npim.unsqueeze(dim=0))
        label = torch.argmax(pred, dim=1)
        label = label.squeeze(dim=0).numpy().astype(np.uint8)
        return label, colors[label].astype(np.uint8)
    return inn


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

from config import TestConfig
conf=TestConfig('aaa')
conf.net_config = {'input_size': (704, 1024)}
conf.tugraz_root="/home/nonari/Documentos/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/"
data = TUGrazToSynthetic(conf)
# impil = Image.open('/home/nonari/Documentos/semantic_drone_dataset_semantics_v1.1/semantic_drone_dataset/training_set/low_res_images/596.jpg')
# impil = impil.resize(SIZE, resample=Image.BILINEAR)
# im = np.asarray(impil)
im, lab = data[0]
unet = init_model()
label, colorim = unet(im)
person_bin = mask_with_ones(label, [7])
plab = measure.label(1 - person_bin)
porps = measure.regionprops(plab)
for i in porps:
    if i.area < 200:
        person_bin[plab==i.label] = 1
risk = risks[label]
person_risk = area(person_bin, 50, decay='solid', convert=True)
risk += person_risk
risk = np.clip(risk, 0, 1)
ppt_r = (1 - risk) / 3
ppt_r[0, 0] = 1
plt.imshow(ppt_r, cmap='hsv')
plt.show()
