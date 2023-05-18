import torch
from PIL import Image
import numpy as np
from custom_models.safeuav import UNet_MDCB
from datasets.synthetic import AeroscapesToSynthetic, synthetic_color_keys
from datasets.dataset import imagenet_norm, prepare_image, adapt_image
from glob import glob
from matplotlib import pyplot as plt

f = glob('/home/nonari/Documentos/Dataset_UAV123_10fps/UAV123_10fps/data_seq/UAV123_10fps/bike1/*')
f = sorted(f)

unet = UNet_MDCB(classes=9)
# model = torch.load('/home/nonari/PycharmProjects/drone_landing/executions/rural_by_uavid/executions/rural_with_uavid/models/0', map_location='cpu')
model = torch.load('/home/nonari/PycharmProjects/drone_landing/executions/aero_w_all/executions/aero_with_both/models/0', map_location='cpu')

unet.load_state_dict(model['model_state_dict'])

totensor = prepare_image(adapt_image((704, 1024)))

for i in f:
    tensor = totensor(i).unsqueeze(dim=0)
    pred = unet(tensor).squeeze().argmax(dim=0).numpy()

    plt.imshow(synthetic_color_keys[pred])
    plt.show()


