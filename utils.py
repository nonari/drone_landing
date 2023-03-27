from torch.utils.data import Sampler
import importlib
import copy


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
        curr_level = obj[key]

    curr_level[keys[-1]] = val


def init_config(kwargs, clazz):
    name = kwargs['name']
    opt = clazz(name=name)

    model_config = kwargs['model_config'] if 'model_config' in kwargs else opt.model_config

    net_config = generate_net_config(model_config, kwargs)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    setattr(opt, 'net_config', net_config)

    return opt


def generate_net_config(model_config, input_args):
    net_config = importlib.import_module(f'net_configurations.{model_config}').CONFIG
    for key in copy.copy(input_args):
        if key.startswith('model_config.'):
            parts = key[13:].split('.')
            set_val(net_config, parts, input_args[key])
            del input_args[key]

    return net_config
