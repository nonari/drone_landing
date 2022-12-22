from os import makedirs, path
from shutil import rmtree
from config import Config
from training import train_net
import fire
from torch.utils.data import SubsetRandomSampler
from dataloader import TUGrazDataset
from glob import glob
from os import path
import torch
from sklearn.model_selection import KFold
import random
from glob import glob
from testing import test_net
import numpy as np


def last_checkpoint(config):
    paths = glob(config.checkpoint_path + '/*')
    paths = sorted(paths, key=lambda p: (path.basename(p).split('_')[0], path.basename(p).split('_')[1]))
    if len(paths) == 0:
        raise Exception(f'No checkpoint found at {config.checkpoint_path}, check path or disable -resume')

    return paths[-1]


def last_executions(config):
    paths = glob(config.checkpoint_path + '/*')
    if len(paths) == 0:
        raise Exception(f'No checkpoint found at {config.checkpoint_path}, check path or disable -resume')

    paths = sorted(paths, key=lambda e: (int(path.basename(e).split('_')[0]), int(path.basename(e).split('_')[1])))

    # Get last of each fold
    lasts = []
    last_fold = int(path.basename(paths[0]).split('_')[0])
    if len(paths) == 1:
        lasts.append(paths[0])
    else:
        for i, p in enumerate(paths[1:]):
            curr_fold = int(path.basename(p).split('_')[0])
            if curr_fold > last_fold:
                lasts.append(paths[i])
            if (i+2) == len(paths):
                lasts.append(paths[i+1])
            last_fold = curr_fold

    return lasts


def folds_strategy(config):
    idx_seed = random.randint(0, 9999)
    config.fold = 0

    dataset = TUGrazDataset(config)
    checkpoint_paths = []
    if config.resume:
        checkpoint_paths = last_executions(config)
        checkpoint = torch.load(checkpoint_paths[0])
        idx_seed = checkpoint['idx_seed']

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=idx_seed)
    folds = list(kfold.split(dataset))

    for fold, (train_idx, _) in enumerate(folds):
        checkpoint = None
        if len(checkpoint_paths) > 0:
            checkpoint = torch.load(checkpoint_paths.pop(0))
            if checkpoint['epoch'] >= config.max_epochs - 1:
                print(f'Fold {fold} was complete.')
                continue
        config.fold = fold
        sampler = SubsetRandomSampler(train_idx)
        train_net(config, dataset, idx_seed, sampler=sampler, checkpoint=checkpoint)


def train(**kwargs):
    name = kwargs['name']
    opt = Config(name=name)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    # create directories
    if not opt.resume:
        if path.exists(opt.save_path):
            if opt.override:
                rmtree(opt.save_path)
            else:
                r = input(f'Will delete {opt.save_path}. Proceed? y/n')
                if r == 'y':
                    rmtree(opt.save_path)
                else:
                    print('!!!! Execution aborted !!!!')
                    exit(0)

    makedirs(opt.save_path, exist_ok=True)
    makedirs(opt.model_path, exist_ok=True)
    makedirs(opt.checkpoint_path, exist_ok=True)
    makedirs(opt.test_path, exist_ok=True)
    makedirs(opt.train_path, exist_ok=True)

    if opt.folds > 1:
        folds_strategy(opt)


def test(**kwargs):
    name = kwargs['name']
    opt = Config(name=name)

    # overwrite options from commandline
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    if opt.folds > 1:
        folds_test(opt)


def folds_test(config):
    dataset = TUGrazDataset(config)
    model_paths = glob(path.join(config.model_path, '*'))
    model_paths = sorted(model_paths, key=lambda p: int(path.basename(p)))

    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu else 'cpu')
    seed_info = torch.load(model_paths[0], map_location=device)

    kfold = KFold(n_splits=config.folds, shuffle=True, random_state=seed_info['idx_seed'])
    folds = list(kfold.split(dataset))

    results = []
    for fold, (_, test_idx) in enumerate(folds):
        config.fold = fold
        sampler = SubsetRandomSampler(test_idx)
        fold_info = torch.load(model_paths[fold], map_location=device)
        result = test_net(config, dataset, fold_info, sampler=sampler)
        results.append(result)
    summarize_results(results)


def summarize_results(results, num_classes, classes=None):
    acc = [r['acc'] for r in results]
    jcc = [r['acc'] for r in results]
    pre = [r['acc'] for r in results]
    f1 = [r['acc'] for r in results]
    conf = [r['acc'] for r in results]

    acc = np.vstack(acc)
    jcc = np.vstack(jcc)
    pre = np.vstack(pre)
    f1 = np.vstack(f1)

    conf = np.dstack(conf)
    conf = conf / conf.astype(np.float).sum(axis=1)


if __name__ == '__main__':
    fire.Fire()
