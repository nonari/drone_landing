import torch
from torch.utils.data.dataset import T_co
from config import TrainConfig, TestConfig
from datasets.ruralscapes import RuralscapesOrigSplit
from datasets.ruralscapes import color_keys as rural_color_keys
from datasets.uavid import color_keys as uavid_color_keys
from datasets.aeroscapes import aeroscapes_color as aeroscapes_color_keys, Aeroscapes
from datasets.aeroscapes import aeroscapes_classnames
from datasets.aeroscapes import label_to_tensor as aero_label_to_tensor
from datasets.ruralscapes import ruralscapes_classnames
from datasets.uavid import UAVid
from datasets.uavid import class_names as uavid_classnames
from datasets.dataset import get_dataset_transform, label_to_tensor_collapse, sc_to_tensor_collapse, GenericDataset
import numpy as np

synthetic_classnames = np.array([
    'building',
    'land',
    'tree',
    'back',
    'obstacle',
    'road',
    'car',
    'person',
    'water'
])

synthetic_color_keys = np.asarray([
    [255, 255, 0],
    [0, 255, 0],
    [0, 255, 255],
    [127, 127, 0],
    [255, 255, 255],
    [255, 0, 255],
    [255, 0, 0],
    [255, 127, 0],
    [0, 0, 255]
])


class RuralscapesOrigToSynthetic(RuralscapesOrigSplit):
    def __init__(self, config):
        super().__init__(config)
        assoc = [
            ('building', 'building'),
            ('land', 'land'),
            ('forest', 'tree'),
            ('sky', 'back'),
            ('fence', 'obstacle'),
            ('road', 'road'),
            ('hill', 'back'),
            ('church', 'building'),
            ('car', 'car'),
            ('person', 'person'),
            ('haystack', 'obstacle'),
            ('water', 'water')
        ]

        transform_color_key, color_collapse = get_dataset_transform(synthetic_classnames, assoc)
        extended_colors = np.vstack([rural_color_keys, [-1, -1, -1]])
        dest_colors = extended_colors[transform_color_key]
        self._color_keys = dest_colors
        self._class_names = synthetic_classnames
        self._label_to_tensor = label_to_tensor_collapse(rural_color_keys, color_collapse)

    def colors(self):
        return synthetic_color_keys

    def classnames(self):
        return synthetic_classnames

    def classes(self):
        return len(synthetic_classnames)

    def pred_to_color_mask(self, true, pred):
        pred_mask = synthetic_color_keys[pred]
        true_mask = synthetic_color_keys[true]
        return true_mask, pred_mask


class UAVidToSynthetic(UAVid):
    def __init__(self, config):
        super().__init__(config)
        assoc = [
            ("background", "back"),
            ("building", "building"),
            ("road", "road"),
            ("tree", "tree"),
            ("low veg", "land"),
            ("moving car", "car"),
            ("static car", "car"),
            ("human", "person"),
        ]

        transform_color_key, color_collapse = get_dataset_transform(synthetic_classnames, assoc)
        extended_colors = np.vstack([uavid_color_keys, [-1, -1, -1]])
        dest_colors = extended_colors[transform_color_key]
        self._color_keys = dest_colors
        self._class_names = synthetic_classnames
        self._label_to_tensor = label_to_tensor_collapse(uavid_color_keys, color_collapse)

    def colors(self):
        return synthetic_color_keys

    def classnames(self):
        return synthetic_classnames

    def classes(self):
        return len(synthetic_classnames)

    def pred_to_color_mask(self, true, pred):
        pred_mask = synthetic_color_keys[pred]
        true_mask = synthetic_color_keys[true]
        return true_mask, pred_mask


def aero_adapter(color_collapse):
    def f(label, dest_class_key):
        sparse_label = aero_label_to_tensor(label, None)
        reduced_label = torch.argmax(sparse_label, dim=0).squeeze(dim=0)
        f2 = sc_to_tensor_collapse(color_collapse)
        return f2(reduced_label, dest_class_key)

    return f


class AeroscapesToSynthetic(Aeroscapes):
    def __init__(self, config):
        super().__init__(config)
        assoc = [
            ('back', 'back'),
            ('person', 'person'),
            ('bike', 'car'),
            ('car', 'car'),
            ('animal', 'person'),
            ('obstacle', 'obstacle'),
            ('building', 'building'),
            ('vegetation', 'tree'),
            ('road', 'road')
        ]

        transform_color_key, color_collapse = get_dataset_transform(synthetic_classnames, assoc)
        orig_class = np.arange(0, 10)
        orig_class[-1] = -1
        dest_class_key = orig_class[transform_color_key]
        self._label_to_tensor = aero_adapter(color_collapse)
        self._color_keys = dest_class_key

    def colors(self):
        return synthetic_color_keys

    def classnames(self):
        return synthetic_classnames

    def classes(self):
        return len(synthetic_classnames)

    def pred_to_color_mask(self, true, pred):
        pred_mask = synthetic_color_keys[pred]
        true_mask = synthetic_color_keys[true]
        return true_mask, pred_mask


class Synthetic(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.max_rural = -1
        self.max_uavid = -1
        self.uavid = UAVidToSynthetic(config)
        self.rural = RuralscapesOrigToSynthetic(config)
        self.aero = AeroscapesToSynthetic(config)

    def classes(self):
        return len(synthetic_classnames)

    def classnames(self):
        return synthetic_classnames

    def colors(self):
        return synthetic_color_keys

    def pred_to_color_mask(self, true, pred):
        return self.rural.pred_to_color_mask(true, pred)

    def get_folds(self):
        rural_folds = self.rural.get_folds()
        uavid_folds = self.uavid.get_folds()
        aero_folds = self.aero.get_folds()

        if isinstance(self.config, TrainConfig):
            rural_train = rural_folds[0][0]
            rural_val = rural_folds[0][1]
            self.max_rural = max([max(rural_train), max(rural_val)])

            uavid_train = np.array(uavid_folds[0][0]) + self.max_rural + 1
            uavid_val = np.array(uavid_folds[0][1]) + self.max_rural + 1
            self.max_uavid = int(np.max([uavid_train.max(), uavid_val.max()]))

            aero_train = np.array(aero_folds[0][0]) + self.max_uavid + 1
            aero_val = np.array(aero_folds[0][1]) + self.max_uavid + 1

            train = list(map(lambda x: int(x), np.concatenate([rural_train, uavid_train, aero_train])))
            val = list(map(lambda x: int(x), np.concatenate([rural_val, uavid_val, aero_val])))

            return [(train, val)]

        else:
            rural_test = rural_folds[0]
            self.max_rural = max(rural_test)

            uavid_test = np.array(uavid_folds[0]) + self.max_rural + 1
            self.max_uavid = uavid_test.max()

            aero_test = np.array(aero_folds[0]) + self.max_uavid + 1

            test = list(map(lambda x: int(x), np.concatenate([rural_test, uavid_test, aero_test])))

            return [test]

    def __getitem__(self, index) -> T_co:
        if index > self.max_uavid:
            return self.aero[index - self.max_uavid - 1]
        elif index > self.max_rural:
            return self.uavid[index - self.max_rural - 1]
        else:
            return self.rural[index]
