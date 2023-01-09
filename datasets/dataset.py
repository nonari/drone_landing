from PIL import Image
from torchvision.transforms import transforms

imagenet_norm = {'mean': [0.485, 0.456, 0.406],
                 'std': [0.229, 0.224, 0.225]}
imagenet_denorm = {'mean': [-0.485/0.229, -0.456/0.224, -0.406/0.225],
                   'std': [1/0.229, 1/0.224, 1/0.225]}


def transform_image(size):
    return transforms.Compose([
        transforms.Resize(size, interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(**imagenet_norm)
    ])
