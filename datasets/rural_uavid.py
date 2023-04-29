from torch.utils.data.dataset import T_co
from config import TrainConfig, TestConfig
from datasets.ruralscapes import RuralscapesOrigSplit
from datasets.ruralscapes import color_keys as rural_color_keys
from datasets.uavid import color_keys as uavid_color_keys
from datasets.ruralscapes import ruralscapes_classnames
from datasets.uavid import UAVid
from datasets.uavid import class_names as uavid_classnames
from datasets.dataset import get_dataset_transform, label_to_tensor_collapse, GenericDataset
import numpy as np


class RuralscapesOrigToUAVid(RuralscapesOrigSplit):
    def __init__(self, config):
        super().__init__(config)
        assoc = [
            ('building', 'building'),
            ('land', 'low veg'),
            ('forest', 'tree'),
            ('sky', None),
            ('fence', 'building'),
            ('road', 'road'),
            ('hill', None),
            ('church', 'building'),
            ('car', 'static car'),
            ('person', 'human'),
            ('haystack', None),
            ('water', None)
        ]

        transform_color_key, color_collapse = get_dataset_transform(uavid_classnames, assoc)
        extended_colors = np.vstack([uavid_color_keys, [-1, -1, -1]])
        dest_colors = extended_colors[transform_color_key]
        self.color_keys = dest_colors
        self.class_names = uavid_classnames
        self._no_classes = len(uavid_classnames)
        self._label_to_tensor = label_to_tensor_collapse(rural_color_keys, color_collapse)


class UAVidToRuralscapes(UAVid):
    def __init__(self, config):
        super().__init__(config)
        assoc = [
            ("background", None),
            ("building", "building"),
            ("road", "road"),
            ("tree", "forest"),
            ("low veg", "land"),
            ("moving car", "car"),
            ("static car", "car"),
            ("human", "person"),
        ]

        transform_color_key, color_collapse = get_dataset_transform(ruralscapes_classnames, assoc)
        extended_colors = np.vstack([rural_color_keys, [-1, -1, -1]])
        dest_colors = extended_colors[transform_color_key]
        self._color_keys = dest_colors
        self._class_names = ruralscapes_classnames
        self._no_classes = len(ruralscapes_classnames)
        self._label_to_tensor = label_to_tensor_collapse(uavid_color_keys, color_collapse)


class UAVid_and_rural(GenericDataset):
    def __init__(self, config):
        super().__init__(config)
        self.max_rural = -1
        self.uavid = UAVidToRuralscapes(config)
        self.rural = RuralscapesOrigSplit(config)

    def classes(self):
        return self.rural.classes()

    def classnames(self):
        return self.rural.classnames()

    def colors(self):
        return self.rural.colors()

    def pred_to_color_mask(self, true, pred):
        return self.rural.pred_to_color_mask(true, pred)

    def get_folds(self):
        rural_folds = self.rural.get_folds()
        uavid_folds = self.uavid.get_folds()

        if isinstance(self.config, TrainConfig):
            rural_train = rural_folds[0][0]
            rural_val = rural_folds[0][1]
            self.max_rural = max([max(rural_train), max(rural_val)])

            uavid_train = np.array(uavid_folds[0][0]) + self.max_rural + 1
            uavid_val = np.array(uavid_folds[0][1]) + self.max_rural + 1

            return [(np.concatenate([rural_train, uavid_train]), np.concatenate([rural_val, uavid_val]))]
        else:
            rural_train = rural_folds[0]
            self.max_rural = max(rural_train)

            uavid_train = np.array(uavid_folds[0]) + self.max_rural + 1

            return [np.concatenate([rural_train, uavid_train])]

    def __getitem__(self, index) -> T_co:
        if index <= self.max_rural:
            return self.rural[index]
        else:
            return self.uavid[index - self.max_rural - 1]
