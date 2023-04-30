import torch
from torch.utils.data import Sampler
import importlib
import copy
from config import TestConfig
from os import path
import json


from torch import nn

def estimate_memory_training(model, sample_input, optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False, device=0):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    optimizer = optimizer_type(model.parameters(), lr=.001)
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = model(model_input.to(device)).sum()
    c = torch.cuda.memory_allocated(device)
    if use_amp:
        amp_multiplier = .5
    else:
        amp_multiplier = 1
    forward_pass_memory = (c - b)*amp_multiplier
    gradient_memory = model_memory
    if isinstance(optimizer, torch.optim.Adam):
        o = 2
    elif isinstance(optimizer, torch.optim.RMSprop):
        o = 1
    elif isinstance(optimizer, torch.optim.SGD):
        o = 0
    elif isinstance(optimizer, torch.optim.Adagrad):
        o = 1
    else:
        raise ValueError("Unsupported optimizer. Look up how many moments are" +
            "stored by your optimizer and add a case to the optimizer checker.")
    gradient_moment_memory = o*gradient_memory
    total_memory = model_memory + forward_pass_memory + gradient_memory + gradient_moment_memory

    return total_memory

def estimate_memory_inference(model, sample_input, batch_size=1, use_amp=False, device=0):
    """Predict the maximum memory usage of the model.
    Args:
        optimizer_type (Type): the class name of the optimizer to instantiate
        model (nn.Module): the neural network model
        sample_input (torch.Tensor): A sample input to the network. It should be
            a single item, not a batch, and it will be replicated batch_size times.
        batch_size (int): the batch size
        use_amp (bool): whether to estimate based on using mixed precision
        device (torch.device): the device to use
    """
    # Reset model and optimizer
    model.cpu()
    a = torch.cuda.memory_allocated(device)
    model.to(device)
    b = torch.cuda.memory_allocated(device)
    model_memory = b - a
    model_input = sample_input.unsqueeze(0).repeat(batch_size, 1)
    output = model(model_input.to(device)).sum()
    total_memory = model_memory

    return total_memory

def test_memory_training(in_size=100, out_size=10, hidden_size=100, optimizer_type=torch.optim.Adam, batch_size=1, use_amp=False, device=0):
    sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(in_size, hidden_size),
                        *[nn.Linear(hidden_size, hidden_size) for _ in range(200)],
                        nn.Linear(hidden_size, out_size))
    max_mem_est = estimate_memory_training(model, sample_input[0], optimizer_type=optimizer_type, batch_size=batch_size, use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    optimizer = optimizer_type(model.parameters(), lr=.001)
    print("Beginning mem:", torch.cuda.memory_allocated(device), "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    for i in range(3):
        optimizer.zero_grad()
        print("Iteration", i)
        with torch.cuda.amp.autocast(enabled=use_amp):
            a = torch.cuda.memory_allocated(device)
            out = model(sample_input.to(device)).sum() # Taking the sum here just to get a scalar output
            b = torch.cuda.memory_allocated(device)
        print("1 - After forward pass", torch.cuda.memory_allocated(device))
        print("2 - Memory consumed by forward pass", b - a)
        out.backward()
        print("3 - After backward pass", torch.cuda.memory_allocated(device))
        optimizer.step()
        print("4 - After optimizer step", torch.cuda.memory_allocated(device))

def test_memory_inference(in_size=100, out_size=10, hidden_size=100, batch_size=1, use_amp=False, device=0):
    sample_input = torch.randn(batch_size, in_size, dtype=torch.float32)
    model = nn.Sequential(nn.Linear(in_size, hidden_size),
                        *[nn.Linear(hidden_size, hidden_size) for _ in range(200)],
                        nn.Linear(hidden_size, out_size))
    max_mem_est = estimate_memory_inference(model, sample_input[0], batch_size=batch_size, use_amp=use_amp)
    print("Maximum Memory Estimate", max_mem_est)
    print("Beginning mem:", torch.cuda.memory_allocated(device), "Note - this may be higher than 0, which is due to PyTorch caching. Don't worry too much about this number")
    model.to(device)
    print("After model to device:", torch.cuda.memory_allocated(device))
    with torch.no_grad():
        for i in range(3):
            print("Iteration", i)
            with torch.cuda.amp.autocast(enabled=use_amp):
                a = torch.cuda.memory_allocated(device)
                out = model(sample_input.to(device)).sum() # Taking the sum here just to get a scalar output
                b = torch.cuda.memory_allocated(device)
            print("1 - After forward pass", torch.cuda.memory_allocated(device))
            print("2 - Memory consumed by forward pass", b - a)

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


def print_obj(obj):
    obj_dict = copy.copy(obj.__dict__)
    for k in obj.__dict__:
        if k.startswith('_'):
            del obj_dict[k]

    def default(o): return f"<<non-serializable: {type(o).__qualname__}>>"
    print(json.dumps(obj_dict, indent=4, default=default), )
