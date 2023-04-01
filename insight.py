import torch

from datasets.ruralscapes import ruralscapes_classnames
from datasets import ruralscapes
import numpy as np
from matplotlib import pyplot as plt
from utils import init_config
import config


def class_hist(dataset):
    labels_hist = np.zeros(dataset.classes())
    for im, label in dataset:
        label_hist = np.sum(label.numpy(), axis=(1, 2))
        labels_hist += label_hist
    print(labels_hist)
    print(labels_hist / labels_hist.sum())



params={'name': 'mock', 'dataset_name':'datasets.dataset.DummyDataset', 'model_config':'safeuav_base'}
train_conf = init_config(params, config.TrainConfig)
train_dataset = ruralscapes.RuralscapesOrigSplit(train_conf)
# test_dataset = ruralscapes.RuralscapesOrigSplit(test_conf)
class_hist(train_dataset)
