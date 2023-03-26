import torch

from datasets.ruralscapes import ruralscapes_classnames
from datasets import ruralscapes
import numpy as np
from matplotlib import pyplot as plt
import config


def class_hist(dataset):
    labels_hist = np.zeros(dataset.classes())
    for im, label in dataset:
        label_hist = np.sum(label.numpy(), axis=(1, 2))
        labels_hist += label_hist
    print(labels_hist)
    print(labels_hist / labels_hist.sum())


train_conf = config.TrainConfig(name='mock')
test_conf = config.TestConfig(name='mock')
train_dataset = ruralscapes.RuralscapesOrigSplit(train_conf)
test_dataset = ruralscapes.RuralscapesOrigSplit(test_conf)
class_hist(train_dataset)
