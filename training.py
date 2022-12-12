import importlib
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
from torch import nn, optim, cuda
from os import path


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


def save_checkpoint(config, net, epoch, loss, idx, best=False):
    if best:
        location = path.join(config.model_path, f'{config.fold}')
    else:
        location = path.join(config.checkpoint_path, f'{config.fold}_{epoch}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'loss': loss,
        'fold': config.fold,
        'idx': idx
    }, location)


def train_net(config, dataset, checkpoint=None):
    device = torch.device('cpu' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'model_configurations.{config.model_config}').CONFIG
    net = configure_net(config, net_config)

    best_loss = 1000
    curr_epoch = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        curr_epoch = checkpoint['epoch']
        best_loss = checkpoint['loss']

    net.to(device=device)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=net_config['batch_size'],
                             drop_last=True,
                             num_workers=config.num_threads)

    optimizer = eval(net_config['optimizer']['name'])(**net_config['optimizer']['path'])
    criterion = eval(net_config['loss'])()

    for epoch in range(curr_epoch, config.max_epochs):
        net.train()

        for image, label in data_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)

            prediction = net(image)

            loss = criterion(prediction, label)

            if loss < best_loss:
                best_loss = loss.item()

            loss.backward()
            optimizer.step()

        if epoch % config.save_every == 0:
            save_checkpoint(config, net, epoch, best_loss, dataset.get_index())
