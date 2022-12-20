from training import configure_net
import torch
from torch.utils.data.dataloader import DataLoader
from torch import cuda
import importlib
from os import path
from matplotlib import pyplot as plt
import numpy as np
from dataloader import tugraz_color_keys


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

    for image, label in data_loader:
        prediction = net(image)
        class_mask = prediction.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        label = label.argmax(dim=1).squeeze().detach().cpu().numpy().astype(np.uint)
        color_mask = tugraz_color_keys[class_mask]
        image = image.squeeze().movedim(0, -1).detach().cpu().numpy()
        plt.imshow(color_mask)
        plt.show()

    data = torch.load(path.join(config.train_path, 'training_results.json'), map_location=device)
    epoch = config.max_epochs if 'epoch' not in data else data[config.fold]['epoch'] + 1
    acc = np.asarray(data[config.fold]['acc'])
    acc = acc.reshape((-1, acc.shape[0] // epoch)).mean(axis=1)
    loss = np.asarray(data[config.fold]['loss'])
    loss = loss.reshape((-1, loss.shape[0] // epoch)).mean(axis=1)
    epochs = np.arange(0, epoch)
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_title(f'{config.name}, fold {config.fold}')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('acc', color=color)
    ax1.plot(epochs, acc, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('loss', color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs, loss, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()