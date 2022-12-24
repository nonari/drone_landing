import importlib
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
from torch import nn, optim, cuda
from os import path
from tqdm import tqdm
import custom_metrics


def configure_net(net_config, classes):
    net_type = net_config['net']
    if net_type == "unet":
        net = smp.Unet(
            encoder_name=net_config['encoder'],
            encoder_weights='imagenet' if net_config['pretrained'] else None,
            in_channels=3,
            classes=classes
        )
    else:
        raise NotImplementedError

    return net


def save_checkpoint(config, net, epoch, loss, idx_seed, best=False):
    if best:
        location = path.join(config.model_path, f'{config.fold}')
    else:
        location = path.join(config.checkpoint_path, f'{config.fold}_{epoch}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'loss': loss,
        'fold': config.fold,
        'idx_seed': idx_seed
    }, location)


def load_data(config, curr_epoch):
    location = path.join(config.train_path, 'training_results')
    if path.exists(location):
        data = torch.load(location)
        if data[config.fold]['epoch'] >= curr_epoch:
            epoch_len = config.datalen * (config.folds - 1) // config.folds // config.batch_size
            last = epoch_len * curr_epoch
            data[config.fold]['acc'] = data[config.fold]['acc'][:last]
            data[config.fold]['loss'] = data[config.fold]['loss'][:last]
    else:
        data = {'batch_size': config.batch_size}
    return data


def save_data(config, data):
    location = path.join(config.train_path, 'training_results')
    torch.save(data, location)


def add_data(data, config, epoch, acc=None, loss=None):
    if config.fold not in data:
        data[config.fold] = {'acc': [], 'loss': []}

    data[config.fold]['acc'].append(acc)
    data[config.fold]['loss'].append(loss)
    data[config.fold]['epoch'] = epoch


def train_net(config, dataset, idx_seed, sampler=None, checkpoint=None):
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    best_loss = 1000
    curr_epoch = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']

    data = load_data(config, curr_epoch)

    net.to(device=device)

    config.datalen = len(dataset)
    data_loader = DataLoader(dataset=dataset,
                             sampler=sampler,
                             batch_size=config.batch_size,
                             drop_last=True,
                             num_workers=config.num_threads)

    optimizer = eval(net_config['optimizer']['name'])(net.parameters(), **net_config['optimizer']['params'])
    criterion = eval(net_config['loss'])()

    prefix = ''
    if config.folds > 1:
        prefix = f'Fold {config.fold}, '

    net.train()
    for epoch in range(curr_epoch, config.max_epochs):
        loss_epoch = 0
        with tqdm(data_loader, unit="batch") as tq_loader:
            for image, label in tq_loader:
                tq_loader.set_description(f'{prefix}Epoch {epoch}')
                optimizer.zero_grad()
                image, label = image.to(device=device), label.to(device=device)

                prediction = net(image)
                acc, loss = custom_metrics.calc_acc(prediction, label), criterion(prediction, label)
                loss_epoch = loss_epoch + loss.item()
                loss.backward()
                optimizer.step()
                add_data(data, config, epoch, acc, loss.item())
                tq_loader.set_postfix(loss=loss.item(), acc=acc)

        loss_epoch /= tq_loader.__len__()
        if loss_epoch < best_loss and epoch > 0:
            best_loss = loss.item()
            save_checkpoint(config, net, epoch, best_loss, idx_seed, best=True)

        if epoch % config.save_every == 0:
            save_checkpoint(config, net, epoch, best_loss, idx_seed)
        elif epoch == (config.max_epochs - 1):
            save_checkpoint(config, net, epoch, best_loss, idx_seed)

        save_data(config, data)

    del net
