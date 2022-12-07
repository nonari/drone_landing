from os import path

from PIL import Image

from config import Config
from glob import glob


tugraz_new_labels = {
    1: [0,0,0]
}


def downsize_tugraz():
    new_size = 1500, 1000
    options = Config()
    subset = 'training_set'
    images_root = path.join(options.tugraz_root, subset, 'low_res_im')
    labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_im_low_res')

    image_paths = glob(images_root + '/*.jpg')
    label_paths = glob(labels_root + '/*.png')

    print('Loading images...')
    images = [Image.open(i) for i in image_paths]
    print('Loading labels...')
    labels = [Image.open(i) for i in label_paths]

    im_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            'low_res_im', path.basename(x)), image_paths))
    lb_paths = list(map(lambda x: path.join(path.dirname(path.dirname(x)),
                                            'label_im_low_res', path.basename(x)), label_paths))
    for p, imo in zip(im_paths, images):
        imr = imo.resize(new_size, Image.BILINEAR)
        imr.save(p)
        imr.close()
        imo.close()

    for p, imo in zip(lb_paths, labels):
        imr = imo.resize(new_size, Image.NEAREST)
        imr.save(p)
        imr.close()
        imo.close()




def collapse_labels():
    pass