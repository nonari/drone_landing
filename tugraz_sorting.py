from config import Config
from os import path, makedirs
from glob import glob
from PIL import Image
import numpy as np
from skimage import io, morphology, filters
from skimage.morphology import square
from sklearn import cluster
import pickle
from shutil import copyfile


# Similarity comparison for CCV according to:
# G. Pass, R. Zabih, and J. Miller, “Comparing images using color coherence vectors,”
# Proceedings of the fourth ACM international conference on Multimedia - MULTIMEDIA ’96.
# ACM Press, 1996. doi: 10.1145/244130.244148.
def delta_g(alphabeta1, alphabeta2):
    a1, b1 = alphabeta1
    a2, b2 = alphabeta2
    return np.sum(np.abs(a1 - a2) + np.abs(b1 - b2))


# Compute similarity of a vector against a list of vectors.
def sim(val1, values, sim_fun):
    res = []
    for idx, val2 in enumerate(values):
        res.append((idx, sim_fun(val1, val2)))
    return res


def compute_ccv(img0, bins=(16, 16, 8), min_area=5):
    factor = 256 / np.array(bins)
    r = (img0[..., 0] // factor[0]).astype(np.uint8)
    g = (img0[..., 1] // factor[1]).astype(np.uint8)
    b = (img0[..., 2] // factor[2]).astype(np.uint8)

    bin_r, bin_g, bin_b = bins
    rep = r + bin_r * g + bin_r * bin_g * b
    rep = filters.median(rep, selem=square(3))
    labeled = morphology.label(rep, connectivity=2)

    labels, indexes, areas = np.unique(labeled, return_counts=True, return_index=True)
    colors = rep.ravel()[indexes]

    coherent_pos = np.where(areas >= min_area)
    incoherent_pos = np.where(areas < min_area)
    coherent_areas = areas[coherent_pos]
    incoherent_areas = areas[incoherent_pos]
    coherent_colors = colors[coherent_pos]
    incoherent_colors = colors[incoherent_pos]

    all_coherent = np.repeat(coherent_colors, coherent_areas)
    all_incoherent = np.repeat(incoherent_colors, incoherent_areas)

    coherent = np.zeros((bin_r * bin_g * bin_b))
    elem, count = np.unique(all_coherent, return_counts=True)
    coherent[elem] = count

    incoherent = np.zeros((bin_r * bin_g * bin_b))
    elem, count = np.unique(all_incoherent, return_counts=True)
    incoherent[elem] = count

    return coherent, incoherent

def generate_samples():
    subset = 'training_set'
    new_images_folder = 'low_res_im'
    new_labels_folder = 'label_im_low_res'

    options = Config(name='')
    images_root = path.join(options.tugraz_root, subset, 'low_res_images')
    labels_root = path.join(options.tugraz_root, subset, 'gt/semantic/label_images')

    new_images_dir = path.join(path.dirname(images_root), new_images_folder)
    new_labels_dir = path.join(path.dirname(labels_root), new_labels_folder)
    makedirs(new_images_dir, exist_ok=True)
    makedirs(new_labels_dir, exist_ok=True)

    image_paths = glob(images_root + '/*.jpg')
    label_paths = glob(labels_root + '/*.png')

    image_paths = sorted(image_paths, key=lambda x: int(x[-7:-4]))

    # samples = []
    # for image_path in image_paths:
    #     image = Image.open(image_path)
    #     alpha, beta = compute_ccv(np.asarray(image), bins=(4, 4, 4), min_area=5)
    #     samples.append(np.concatenate((alpha, beta)))
    #     print(image_path)
    #
    # samples = np.vstack(samples)
    # np.save('samples.npy', samples)
    return image_paths


def kmeans():
    subset = 'training_set'
    new_images_folder = 'classif'
    new_labels_folder = 'label_im_low_res'

    options = Config(name='')
    images_root = path.join(options.tugraz_root, subset, 'low_res_images')
    image_paths = generate_samples()
    image_paths = np.asarray(image_paths)
    samples = np.load('samples.npy')
    km = cluster.KMeans(n_clusters=12)
    km.fit(samples)
    for i in range(12):
        idx = np.where(km.labels_ == i)
        idx_paths = image_paths[idx]
        for idx_path in idx_paths:
            new_p = path.join(path.dirname(path.dirname(idx_path)), new_images_folder,f'{i}_'+path.basename(idx_path))
            copyfile(idx_path, new_p)

kmeans()