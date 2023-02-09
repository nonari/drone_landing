import importlib
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch
from torch import nn, optim, cuda
from os import path, remove
from tqdm import tqdm
from shutil import copy
import custom_metrics
from mock_scheduler import MockScheduler


def configure_net(net_config, classes):
    net_type = net_config['net']
    if net_type == "unet":
        net = smp.Unet(
            encoder_name=net_config['encoder'],
            encoder_weights='imagenet' if net_config['pretrained'] else None,
            in_channels=3,
            classes=classes
        )
    elif net_type == 'pspnet':
        net = smp.PSPNet(
            encoder_name=net_config['encoder'],
            encoder_weights='imagenet' if net_config['pretrained'] else None,
            in_channels=3,
            classes=classes)
    else:
        raise NotImplementedError

    return net


def save_checkpoint(config, net, epoch, loss, best=False):
    if best:
        location = path.join(config.model_path, f'{config.fold}')
    else:
        location = path.join(config.checkpoint_path, f'{config.fold}_{epoch}')

    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'loss': loss,
        'fold': config.fold
    }, location)


def load_data(config, curr_epoch):
    location = path.join(config.train_path, 'training_results')
    if path.exists(location):
        data = torch.load(location)
        if config.fold in data and data[config.fold]['epoch'] >= curr_epoch:
            last = config.datalen * curr_epoch
            data[config.fold]['acc'] = data[config.fold]['acc'][:last]
            data[config.fold]['loss'] = data[config.fold]['loss'][:last]
            if 'acc_val' in data[config.fold]:
                last = curr_epoch // config.validation_epochs
                data[config.fold]['acc_val'] = data[config.fold]['acc_val'][:last]
                data[config.fold]['loss_val'] = data[config.fold]['loss_val'][:last]
    else:
        data = {}
    return data


def save_data(config, data):
    location = path.join(config.train_path, 'training_results')
    torch.save(data, location)


def add_data(data, config, epoch, acc=None, loss=None, val=False):
    if config.fold not in data:
        data[config.fold] = {'acc': [], 'loss': []}
    if val and 'acc_val' not in data[config.fold]:
        data[config.fold]['acc_val'] = []
        data[config.fold]['loss_val'] = []

    if not val:
        data[config.fold]['acc'].append(acc)
        data[config.fold]['loss'].append(loss)
    else:
        data[config.fold]['acc_val'].append(acc)
        data[config.fold]['loss_val'].append(loss)
    data[config.fold]['epoch'] = epoch


def train_net(config, dataset, train_sampler=None, checkpoint=None):
    device = torch.device('cuda' if cuda.is_available() and config.gpu else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    best_loss = 1000
    curr_epoch = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']

    net.to(device=device)

    data_loader = DataLoader(dataset=dataset,
                             sampler=train_sampler,
                             batch_size=config.batch_size,
                             drop_last=True,
                             num_workers=config.num_threads)

    config.datalen = len(data_loader)
    data = load_data(config, curr_epoch)

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
            best_loss = loss_epoch
            save_checkpoint(config, net, epoch, best_loss, best=True)

        if epoch % config.save_every == 0:
            save_checkpoint(config, net, epoch, best_loss)
        elif epoch == (config.max_epochs - 1):
            save_checkpoint(config, net, epoch, best_loss)

        save_data(config, data)

    del net


def remove_past_checkpoints(config, epoch):
    if epoch // config.validation_epochs > config.stop_after_miss + 1:
        removable_epoch = epoch - config.validation_epochs * (config.stop_after_miss + 1)
        removable_check_path = path.join(config.checkpoint_path, f'{config.fold}_{removable_epoch}')
        if path.exists(removable_check_path):
            remove(removable_check_path)


def train_net_with_validation(config, dataset, train_sampler=None, val_sampler=None, checkpoint=None):
    device = torch.device('cuda' if config.gpu and cuda.is_available() else 'cpu')
    net_config = importlib.import_module(f'net_configurations.{config.model_config}').CONFIG
    net = configure_net(net_config, dataset.classes())

    curr_epoch = 0
    if checkpoint is not None:
        net.load_state_dict(checkpoint['model_state_dict'])
        curr_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']

    net.to(device=device)

    data_loader = DataLoader(dataset=dataset,
                             sampler=train_sampler,
                             batch_size=config.batch_size,
                             drop_last=True,
                             num_workers=config.num_threads)

    config.datalen = len(data_loader)
    data = load_data(config, curr_epoch)

    val_loader = DataLoader(dataset=dataset,
                            sampler=val_sampler,
                            batch_size=config.batch_size,
                            drop_last=False,
                            num_workers=config.num_threads)

    optimizer = eval(net_config['optimizer']['name'])(net.parameters(), **net_config['optimizer']['params'])
    criterion = eval(net_config['loss'])()

    if 'lr_scheduler' in net_config:
        scheduler = eval(net_config['lr_scheduler']['name'])(optimizer, **net_config['lr_scheduler']['params'])
        scheduler.step(8)
    else:
        scheduler = MockScheduler()
    prefix = ''
    if config.folds > 1:
        prefix = f'Fold {config.fold}, '

    for epoch in range(curr_epoch, config.max_epochs):
        net.train()
        scheduler.step()
        config._training = True
        with tqdm(data_loader, unit="batch") as tq_loader:
            for image, label in tq_loader:
                tq_loader.set_description(f'{prefix}Epoch {epoch}')
                optimizer.zero_grad()
                image, label = image.to(device=device), label.to(device=device)

                prediction = net(image)
                acc, loss = custom_metrics.calc_acc(prediction, label), criterion(prediction, label)
                loss.backward()
                optimizer.step()
                add_data(data, config, epoch, acc, loss.item())
                tq_loader.set_postfix(loss=loss.item(), acc=acc)

        save_data(config, data)
        if epoch % config.validation_epochs == 0:
            config._training = False
            save_checkpoint(config, net, epoch, 0)
            net.eval()
            loss_val, acc_val = 0, 0
            with tqdm(val_loader, unit="batch") as tq_loader2:
                for image, label in tq_loader2:
                    tq_loader2.set_description(f'VALIDATION Fold {prefix}Epoch {epoch}')
                    image, label = image.to(device=device), label.to(device=device)
                    prediction = net(image)
                    acc, loss = custom_metrics.calc_acc(prediction, label), criterion(prediction, label)
                    loss_val += loss.item()
                    acc_val += acc
                    tq_loader2.set_postfix(loss=loss.item(), acc=acc)
            loss_val /= val_loader.__len__()
            acc_val /= val_loader.__len__()
            add_data(data, config, epoch, acc_val, loss_val, val=True)
            stop = check_stop(config, data)
            save_data(config, data)
            remove_past_checkpoints(config, epoch)
            if stop or (config.max_epochs - 1 - epoch) < config.validation_epochs:
                copy_good_epoch(config, epoch)
                break

    del net


def copy_good_epoch(config, epoch):
    good_epoch = epoch - config.validation_epochs * config.stop_after_miss
    ori = path.join(config.checkpoint_path, f'{config.fold}_{good_epoch}')
    dst = path.join(config.model_path, f'{config.fold}')
    copy(ori, dst)


def check_stop(config, data):
    max_miss = config.stop_after_miss
    acc_val = data[config.fold]['acc_val']
    if len(acc_val) < max_miss + 1:
        return False

    acc_valid = acc_val[-max_miss - 1:]
    last_acc = acc_valid[-1]
    prev_acc = acc_valid[:-1]
    diffs = list(map(lambda x: last_acc - x, prev_acc))
    valid = list(map(lambda x: x < config.delta, diffs))
    stop = all(valid)

    return stop




