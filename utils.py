import torch
from torch.utils.data import Sampler
import importlib
import copy
from config import TestConfig
from os import path


class SeqSampler(Sampler[int]):
    def __init__(self, data_source, indices):
        super().__init__(None)
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def set_val(obj, keys, val):
    curr_level = obj
    for key in keys[:-1]:
        curr_level = curr_level[key]

    curr_level[keys[-1]] = val


def init_config(kwargs, clazz):
    if 'name' not in kwargs:
        raise Exception('Provide a name for the execution')

    name = kwargs['name']

    if name == '':
        raise Exception('Name can\'t be an empty string')

    opt = clazz(name=name)
    if 'model_config' not in kwargs:
        raise Exception('Provide model_config parameter')
    if 'dataset_name' not in kwargs:
        raise Exception('Provide dataset_name parameter')
    model_config = kwargs['model_config'] if 'model_config' in kwargs else opt.model_config
    net_conf_args = filter_net_config(kwargs)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    base_net_config = None
    if isinstance(opt, TestConfig):
        exec_info_path = path.join(opt.train_path, 'execution_info')
        if path.exists(exec_info_path):
            info = torch.load(exec_info_path)
            if 'net_config' in info:
                base_net_config = info['net_config']
            else:
                print('WARNING: Old file without net config')

    net_config = generate_net_config(model_config, net_conf_args, opt, base_net_config)
    setattr(opt, 'net_config', net_config)

    return opt


def filter_net_config(input_args):
    items = []
    for key, value in copy.copy(input_args).items():
        if key.startswith('model_config.'):
            items.append((key, value))
            del input_args[key]

    return items


def generate_net_config(model_config, input_args, config, net_config=None):
    if net_config is None:
        net_config = importlib.import_module(f'net_configurations.{model_config}').CONFIG
    for key, value in input_args:
        parts = key[13:].split('.')
        if value == '!CONFIG':
            value = config
        set_val(net_config, parts, value)

    return net_config


def import_class(full_name):
    classname = full_name.split('.')[-1]
    classpath = full_name[:-1 - len(classname)]
    clazz = getattr(importlib.import_module(classpath), classname)

    return clazz
