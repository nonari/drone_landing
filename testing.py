from training import configure_net
import torch
from torch.utils.data.dataloader import DataLoader
from torch import cuda
import importlib
from os import path
from matplotlib import pyplot as plt
import numpy as np


def test_net(config, dataset, fold_info, sampler=None):
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    net.load_state_dict(fold_info['model_state_dict'])
    net.to(device=device)
    net.train(mode=False)

    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=1,
                             num_workers=config.num_threads)

    # for image, label in data_loader:
    #     prediction = net(image)

    data = torch.load(path.join(config.train_path, 'training_results.json'))
    epoch = config.max_epochs if 'epoch' not in data else data[0]['epoch'] + 1
    acc = data[0]['acc'].reshape[-1, data[0]['acc'].shape[0] // epoch]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()