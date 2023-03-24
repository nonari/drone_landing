from datasets.ruralscapes import ruralscapes_classnames
from datasets import ruralscapes
import numpy as np
from matplotlib import pyplot as plt
import config


def class_hist(dataset):
    for i in dataset:
        pass


train_conf = config.Config(name='mock')
test_conf = config.TestConfig(name='mock')
train_dataset = ruralscapes.RuralscapesOrigSplit(train_conf)
test_dataset = ruralscapes.RuralscapesOrigSplit(test_conf)
class_hist(train_dataset)