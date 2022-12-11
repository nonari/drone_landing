import importlib
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
from torch import nn, optim, cuda


def configure_net(config, net_config):
    net_type = net_config['net']
    if net_type == "unet":
        net = smp.Unet(
            encoder_name=net_config['encoder'],
            encoder_weights='imagenet' if net_config['pretrained'] else None,
            in_channels=3,
            classes=config.classes
        )
    else:
        raise NotImplementedError

    return net


def train_net(config, dataset):
    # TODO: checkpoint loading
    device = torch.device('cpu' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'model_configurations.{config.model_config}').CONFIG
    net = configure_net(config, net_config)
    net.to(device=device)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=net_config['batch_size'],
                             drop_last=True,
                             num_workers=config.num_threads)

    optimizer = eval(net_config['optimizer']['name'])(**net_config['optimizer']['path'])
    criterion = eval(net_config['loss'])()

    curr_epoch = 0

    for epoch in range(curr_epoch, config.max_epochs):
        net.train()
        for image, label in data_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            prediction = net(image)

            loss = criterion(prediction, label)

            loss.backward()
            optimizer.step()
